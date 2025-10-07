"""
Chroma vector database implementation for the Lexora Agentic RAG SDK.

This module provides a Chroma-based vector database implementation with
collection management, persistence, and efficient similarity search.
"""

import asyncio
import json
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import uuid

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.api.models.Collection import Collection
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from .base_vector_db import BaseVectorDB
from ..models.core import Document, SearchResult, CorpusInfo
from ..exceptions import LexoraError, ErrorCode, create_vector_db_error
from ..utils.embeddings import EmbeddingManager
from ..utils.logging import get_logger


class ChromaVectorDB(BaseVectorDB):
    """
    Chroma implementation of vector database.
    
    This implementation uses ChromaDB for efficient similarity search with
    support for local persistence and in-memory operations.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        embedding_manager: EmbeddingManager,
        persist_directory: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize Chroma vector database.
        
        Args:
            config: Configuration dictionary
            embedding_manager: Embedding manager for generating embeddings
            persist_directory: Directory for persistent storage (local mode)
            host: Chroma server host (client mode)
            port: Chroma server port (client mode)
            **kwargs: Additional configuration options
            
        Raises:
            LexoraError: If Chroma is not available or configuration is invalid
        """
        if not CHROMA_AVAILABLE:
            raise create_vector_db_error(
                "ChromaDB library not available. Install with: pip install chromadb",
                "Chroma",
                error_code=ErrorCode.VECTOR_DB_CONNECTION_FAILED
            )
        
        super().__init__(config, **kwargs)
        
        self.embedding_manager = embedding_manager
        self.persist_directory = persist_directory
        self.host = host
        self.port = port
        
        # Chroma client and collections
        self.client: Optional[chromadb.Client] = None
        self._collections: Dict[str, Collection] = {}  # corpus_name -> chroma collection
        self._corpus_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.dimension = embedding_manager.get_dimension()
        self.distance_function = kwargs.get('distance_function', 'cosine')
        self.collection_metadata = kwargs.get('collection_metadata', {})
        
        # Determine client mode
        self.use_server = host is not None and port is not None
        self.use_persistence = persist_directory is not None
        
        self.logger = get_logger(self.__class__.__name__)
    
    async def _connect_impl(self) -> None:
        """Initialize Chroma client and load existing collections."""
        try:
            if self.use_server:
                # Connect to Chroma server
                self.client = chromadb.HttpClient(
                    host=self.host,
                    port=self.port
                )
                self.logger.info(f"Connected to Chroma server at {self.host}:{self.port}")
                
            elif self.use_persistence:
                # Use persistent client
                self.client = chromadb.PersistentClient(
                    path=self.persist_directory
                )
                self.logger.info(f"Connected to persistent Chroma at {self.persist_directory}")
                
            else:
                # Use in-memory client
                self.client = chromadb.Client()
                self.logger.info("Connected to in-memory Chroma client")
            
            # Load existing collections
            await self._load_existing_collections()
            
        except Exception as e:
            raise create_vector_db_error(
                f"Failed to connect to Chroma: {str(e)}",
                "Chroma",
                error_code=ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                original_error=e
            )
    
    async def _disconnect_impl(self) -> None:
        """Clean up Chroma connection."""
        self._collections.clear()
        self._corpus_metadata.clear()
        self.client = None
    
    async def create_corpus(self, name: str, **kwargs) -> bool:
        """
        Create a new Chroma collection (corpus).
        
        Args:
            name: Name of the corpus/collection to create
            **kwargs: Additional collection creation parameters
            
        Returns:
            True if corpus was created successfully
            
        Raises:
            LexoraError: If corpus creation fails
        """
        if not self.client:
            raise create_vector_db_error(
                "Not connected to Chroma",
                "Chroma",
                name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED
            )
        
        if name in self._collections:
            raise create_vector_db_error(
                f"Collection '{name}' already exists",
                "Chroma",
                name,
                ErrorCode.VECTOR_DB_CORPUS_ALREADY_EXISTS
            )        
        try:
            # Prepare collection metadata
            collection_metadata = {
                "created_at": datetime.utcnow().isoformat(),
                "dimension": self.dimension,
                "distance_function": kwargs.get('distance_function', self.distance_function),
                **self.collection_metadata,
                **kwargs.get('metadata', {})
            }
            
            # Create the collection
            collection = self.client.create_collection(
                name=name,
                metadata=collection_metadata,
                embedding_function=None  # We'll provide embeddings manually
            )
            
            # Store collection and metadata
            self._collections[name] = collection
            self._corpus_metadata[name] = {
                "created_at": datetime.utcnow(),
                "dimension": self.dimension,
                "distance_function": kwargs.get('distance_function', self.distance_function),
                "metadata": kwargs
            }
            
            self.logger.info(f"Created Chroma collection '{name}'")
            return True
            
        except Exception as e:
            # Clean up on failure
            if name in self._collections:
                del self._collections[name]
            if name in self._corpus_metadata:
                del self._corpus_metadata[name]
            
            raise create_vector_db_error(
                f"Failed to create Chroma collection '{name}': {str(e)}",
                "Chroma",
                name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                e
            )
    
    async def delete_corpus(self, name: str) -> bool:
        """
        Delete a Chroma collection (corpus).
        
        Args:
            name: Name of the corpus/collection to delete
            
        Returns:
            True if corpus was deleted successfully
            
        Raises:
            LexoraError: If corpus deletion fails
        """
        if not self.client:
            raise create_vector_db_error(
                "Not connected to Chroma",
                "Chroma",
                name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED
            )
        
        if name not in self._collections:
            raise create_vector_db_error(
                f"Collection '{name}' not found",
                "Chroma",
                name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        try:
            # Delete the collection from Chroma
            self.client.delete_collection(name=name)
            
            # Remove from local storage
            del self._collections[name]
            if name in self._corpus_metadata:
                del self._corpus_metadata[name]
            
            self.logger.info(f"Deleted Chroma collection '{name}'")
            return True
            
        except Exception as e:
            raise create_vector_db_error(
                f"Failed to delete Chroma collection '{name}': {str(e)}",
                "Chroma",
                name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                e
            )
    
    async def add_documents(self, corpus_name: str, documents: List[Document]) -> bool:
        """
        Add documents to a Chroma collection.
        
        Args:
            corpus_name: Name of the corpus/collection to add documents to
            documents: List of documents to add
            
        Returns:
            True if documents were added successfully
            
        Raises:
            LexoraError: If document addition fails
        """
        if corpus_name not in self._collections:
            raise create_vector_db_error(
                f"Collection '{corpus_name}' not found",
                "Chroma",
                corpus_name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        if not documents:
            return True
        
        try:
            collection = self._collections[corpus_name]
            
            # Generate embeddings for documents that don't have them
            texts_to_embed = []
            docs_to_process = []
            
            for doc in documents:
                if doc.embedding is None:
                    texts_to_embed.append(doc.content)
                    docs_to_process.append(doc)
                else:
                    docs_to_process.append(doc)
            
            # Generate embeddings in batch if needed
            if texts_to_embed:
                embeddings = await self.embedding_manager.generate_embeddings_batch(texts_to_embed)
                
                # Update documents with embeddings
                embedding_idx = 0
                for doc in docs_to_process:
                    if doc.embedding is None:
                        doc.embedding = embeddings[embedding_idx]
                        embedding_idx += 1
            
            # Prepare data for Chroma
            ids = [doc.id for doc in docs_to_process]
            embeddings = [doc.embedding for doc in docs_to_process]
            documents_content = [doc.content for doc in docs_to_process]
            metadatas = []
            
            for doc in docs_to_process:
                metadata = {
                    "corpus_name": corpus_name,
                    **doc.metadata
                }
                # Ensure all metadata values are strings, numbers, or booleans
                cleaned_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        cleaned_metadata[key] = value
                    else:
                        cleaned_metadata[key] = str(value)
                metadatas.append(cleaned_metadata)
            
            # Add to Chroma collection
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents_content,
                metadatas=metadatas
            )
            
            self.logger.info(f"Added {len(documents)} documents to Chroma collection '{corpus_name}'")
            return True
            
        except Exception as e:
            raise create_vector_db_error(
                f"Failed to add documents to Chroma collection '{corpus_name}': {str(e)}",
                "Chroma",
                corpus_name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                e
            )
    
    async def search(self, corpus_name: str, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search for similar documents in a Chroma collection.
        
        Args:
            corpus_name: Name of the corpus/collection to search in
            query: Search query text
            top_k: Maximum number of results to return
            
        Returns:
            List of search results ordered by relevance
            
        Raises:
            LexoraError: If search fails
        """
        if corpus_name not in self._collections:
            raise create_vector_db_error(
                f"Collection '{corpus_name}' not found",
                "Chroma",
                corpus_name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        try:
            collection = self._collections[corpus_name]
            
            # Generate embedding for query
            query_embedding = await self.embedding_manager.generate_embedding(query)
            
            # Perform search
            search_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Convert results to SearchResult objects
            results = []
            
            if search_results['ids'] and len(search_results['ids']) > 0:
                ids = search_results['ids'][0]
                documents_content = search_results['documents'][0]
                metadatas = search_results['metadatas'][0]
                distances = search_results['distances'][0]
                
                for i in range(len(ids)):
                    doc_id = ids[i]
                    content = documents_content[i]
                    metadata = metadatas[i].copy()
                    distance = distances[i]
                    
                    # Remove corpus_name from metadata
                    stored_corpus_name = metadata.pop('corpus_name', corpus_name)
                    
                    # Get corpus-specific distance function
                    corpus_distance_function = self._corpus_metadata.get(corpus_name, {}).get('distance_function', self.distance_function)
                    
                    # Convert distance to similarity score based on corpus-specific distance function
                    if corpus_distance_function == 'cosine':
                        # For cosine distance: similarity = 1 - distance
                        similarity_score = max(0.0, min(1.0, 1.0 - distance))
                    elif corpus_distance_function == 'l2':
                        # For L2 distance: similarity = 1 / (1 + distance)
                        similarity_score = 1.0 / (1.0 + distance)
                    elif corpus_distance_function == 'ip':
                        # For inner product: apply sigmoid mapping to bound to [0,1]
                        similarity_score = 1.0 / (1.0 + math.exp(-distance))
                    else:
                        # Fallback: use cosine formula
                        similarity_score = max(0.0, min(1.0, 1.0 - distance))
                    
                    # Ensure final similarity is clamped to [0.0, 1.0]
                    similarity_score = max(0.0, min(1.0, similarity_score))
                    
                    # Create Document object
                    document = Document(
                        id=doc_id,
                        content=content,
                        metadata=metadata,
                        embedding=None  # Don't include embedding in search results
                    )
                    
                    # Create SearchResult
                    result = SearchResult(
                        document=document,
                        score=similarity_score,
                        corpus_name=stored_corpus_name
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            raise create_vector_db_error(
                f"Failed to search Chroma collection '{corpus_name}': {str(e)}",
                "Chroma",
                corpus_name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                e
            )
    
    async def list_corpora(self) -> List[str]:
        """
        List all available Chroma collections.
        
        Returns:
            List of corpus/collection names
        """
        return list(self._collections.keys())
    
    async def get_corpus_info(self, name: str) -> CorpusInfo:
        """
        Get information about a specific Chroma collection.
        
        Args:
            name: Name of the corpus/collection
            
        Returns:
            CorpusInfo object with corpus details
            
        Raises:
            LexoraError: If corpus doesn't exist
        """
        if name not in self._collections:
            raise create_vector_db_error(
                f"Collection '{name}' not found",
                "Chroma",
                name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        try:
            collection = self._collections[name]
            
            # Get collection count
            document_count = collection.count()
            
            metadata = self._corpus_metadata.get(name, {})
            
            return CorpusInfo(
                name=name,
                document_count=document_count,
                created_at=metadata.get("created_at", datetime.utcnow()),
                metadata={
                    **metadata.get("metadata", {}),
                    "dimension": metadata.get("dimension", self.dimension),
                    "distance_function": metadata.get("distance_function", self.distance_function),
                    "collection_metadata": collection.metadata
                }
            )
            
        except Exception as e:
            raise create_vector_db_error(
                f"Failed to get info for Chroma collection '{name}': {str(e)}",
                "Chroma",
                name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                e
            )
    
    async def delete_document(self, corpus_name: str, document_id: str) -> bool:
        """
        Delete a specific document from a Chroma collection.
        
        Args:
            corpus_name: Name of the corpus/collection
            document_id: ID of the document to delete
            
        Returns:
            True if document was deleted successfully
            
        Raises:
            LexoraError: If document deletion fails
        """
        if corpus_name not in self._collections:
            raise create_vector_db_error(
                f"Collection '{corpus_name}' not found",
                "Chroma",
                corpus_name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        try:
            collection = self._collections[corpus_name]
            
            # Delete the document by ID
            collection.delete(ids=[document_id])
            
            self.logger.info(f"Deleted document '{document_id}' from Chroma collection '{corpus_name}'")
            return True
            
        except Exception as e:
            raise create_vector_db_error(
                f"Failed to delete document '{document_id}' from Chroma collection '{corpus_name}': {str(e)}",
                "Chroma",
                corpus_name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                e
            )
    
    async def update_document(self, corpus_name: str, document: Document) -> bool:
        """
        Update an existing document in a Chroma collection.
        
        Args:
            corpus_name: Name of the corpus/collection
            document: Updated document
            
        Returns:
            True if document was updated successfully
            
        Raises:
            LexoraError: If document update fails
        """
        if corpus_name not in self._collections:
            raise create_vector_db_error(
                f"Collection '{corpus_name}' not found",
                "Chroma",
                corpus_name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        try:
            collection = self._collections[corpus_name]
            
            # Generate embedding if not provided
            if document.embedding is None:
                document.embedding = await self.embedding_manager.generate_embedding(document.content)
            
            # Prepare metadata
            metadata = {
                "corpus_name": corpus_name,
                **document.metadata
            }
            # Ensure all metadata values are strings, numbers, or booleans
            cleaned_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    cleaned_metadata[key] = value
                else:
                    cleaned_metadata[key] = str(value)
            
            # Update the document (upsert operation)
            collection.upsert(
                ids=[document.id],
                embeddings=[document.embedding],
                documents=[document.content],
                metadatas=[cleaned_metadata]
            )
            
            self.logger.info(f"Updated document '{document.id}' in Chroma collection '{corpus_name}'")
            return True
            
        except Exception as e:
            raise create_vector_db_error(
                f"Failed to update document '{document.id}' in Chroma collection '{corpus_name}': {str(e)}",
                "Chroma",
                corpus_name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                e
            )
    
    async def get_document(self, corpus_name: str, document_id: str) -> Optional[Document]:
        """
        Retrieve a specific document from a Chroma collection.
        
        Args:
            corpus_name: Name of the corpus/collection
            document_id: ID of the document to retrieve
            
        Returns:
            Document if found, None otherwise
            
        Raises:
            LexoraError: If retrieval fails
        """
        if corpus_name not in self._collections:
            raise create_vector_db_error(
                f"Collection '{corpus_name}' not found",
                "Chroma",
                corpus_name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        try:
            collection = self._collections[corpus_name]
            
            # Get the document by ID
            result = collection.get(
                ids=[document_id],
                include=['documents', 'metadatas', 'embeddings']
            )
            
            if not result['ids'] or len(result['ids']) == 0:
                return None
            
            # Extract document information
            content = result['documents'][0]
            metadata = result['metadatas'][0].copy()
            embedding = result['embeddings'][0] if result['embeddings'] else None
            
            # Remove corpus_name from metadata
            metadata.pop('corpus_name', None)
            
            return Document(
                id=document_id,
                content=content,
                metadata=metadata,
                embedding=embedding
            )
            
        except Exception as e:
            raise create_vector_db_error(
                f"Failed to get document '{document_id}' from Chroma collection '{corpus_name}': {str(e)}",
                "Chroma",
                corpus_name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                e
            )
    
    # Chroma-specific utility methods
    
    async def _load_existing_collections(self) -> None:
        """Load existing Chroma collections."""
        try:
            collections = self.client.list_collections()
            
            for collection_info in collections:
                collection_name = collection_info.name
                
                # Get the collection
                collection = self.client.get_collection(name=collection_name)
                self._collections[collection_name] = collection
                
                # Extract metadata from collection
                collection_metadata = collection.metadata or {}
                created_at_str = collection_metadata.get("created_at")
                
                if created_at_str:
                    try:
                        created_at = datetime.fromisoformat(created_at_str)
                    except:
                        created_at = datetime.utcnow()
                else:
                    created_at = datetime.utcnow()
                
                # Store metadata
                self._corpus_metadata[collection_name] = {
                    "created_at": created_at,
                    "dimension": collection_metadata.get("dimension", self.dimension),
                    "distance_function": collection_metadata.get("distance_function", self.distance_function),
                    "metadata": {}
                }
            
        except Exception as e:
            self.logger.warning(f"Failed to load existing Chroma collections: {str(e)}")
    
    def get_collection_info(self, corpus_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a Chroma collection.
        
        Args:
            corpus_name: Name of the corpus/collection
            
        Returns:
            Dictionary with collection information
            
        Raises:
            LexoraError: If corpus doesn't exist
        """
        if corpus_name not in self._collections:
            raise create_vector_db_error(
                f"Collection '{corpus_name}' not found",
                "Chroma",
                corpus_name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        try:
            collection = self._collections[corpus_name]
            
            return {
                "name": corpus_name,
                "id": collection.id,
                "count": collection.count(),
                "metadata": collection.metadata,
                "dimension": self._corpus_metadata[corpus_name].get("dimension", self.dimension),
                "distance_function": self._corpus_metadata[corpus_name].get("distance_function", self.distance_function)
            }
            
        except Exception as e:
            raise create_vector_db_error(
                f"Failed to get info for Chroma collection '{corpus_name}': {str(e)}",
                "Chroma",
                corpus_name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                e
            )
    
    async def peek_collection(self, corpus_name: str, limit: int = 10) -> Dict[str, Any]:
        """
        Peek at a few documents in a Chroma collection.
        
        Args:
            corpus_name: Name of the corpus/collection
            limit: Maximum number of documents to return
            
        Returns:
            Dictionary with sample documents
            
        Raises:
            LexoraError: If corpus doesn't exist
        """
        if corpus_name not in self._collections:
            raise create_vector_db_error(
                f"Collection '{corpus_name}' not found",
                "Chroma",
                corpus_name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        try:
            collection = self._collections[corpus_name]
            
            # Peek at the collection
            result = collection.peek(limit=limit)
            
            return {
                "ids": result.get('ids', []),
                "documents": result.get('documents', []),
                "metadatas": result.get('metadatas', []),
                "count": len(result.get('ids', []))
            }
            
        except Exception as e:
            raise create_vector_db_error(
                f"Failed to peek at Chroma collection '{corpus_name}': {str(e)}",
                "Chroma",
                corpus_name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                e
            )


# Convenience functions

def create_chroma_vector_db(
    embedding_manager: EmbeddingManager,
    persist_directory: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    distance_function: str = 'cosine',
    **kwargs
) -> ChromaVectorDB:
    """
    Create a Chroma vector database with common configuration.
    
    Args:
        embedding_manager: Embedding manager for generating embeddings
        persist_directory: Directory for persistent storage (local mode)
        host: Chroma server host (client mode)
        port: Chroma server port (client mode)
        distance_function: Distance function to use ('cosine', 'l2', 'ip')
        **kwargs: Additional configuration options
        
    Returns:
        Configured Chroma vector database
    """
    config = {
        "provider": "chroma",
        "persist_directory": persist_directory,
        "host": host,
        "port": port,
        "distance_function": distance_function
    }
    
    return ChromaVectorDB(
        config=config,
        embedding_manager=embedding_manager,
        persist_directory=persist_directory,
        host=host,
        port=port,
        distance_function=distance_function,
        **kwargs
    )


def create_persistent_chroma_vector_db(
    embedding_manager: EmbeddingManager,
    persist_directory: str = "./chroma_storage",
    distance_function: str = 'cosine',
    **kwargs
) -> ChromaVectorDB:
    """
    Create a persistent Chroma vector database.
    
    Args:
        embedding_manager: Embedding manager for generating embeddings
        persist_directory: Directory for persistent storage
        distance_function: Distance function to use
        **kwargs: Additional configuration options
        
    Returns:
        Configured persistent Chroma vector database
    """
    return create_chroma_vector_db(
        embedding_manager=embedding_manager,
        persist_directory=persist_directory,
        distance_function=distance_function,
        **kwargs
    )


def create_chroma_server_client(
    embedding_manager: EmbeddingManager,
    host: str = "localhost",
    port: int = 8000,
    distance_function: str = 'cosine',
    **kwargs
) -> ChromaVectorDB:
    """
    Create a Chroma vector database client for server mode.
    
    Args:
        embedding_manager: Embedding manager for generating embeddings
        host: Chroma server host
        port: Chroma server port
        distance_function: Distance function to use
        **kwargs: Additional configuration options
        
    Returns:
        Configured Chroma server client
    """
    return create_chroma_vector_db(
        embedding_manager=embedding_manager,
        host=host,
        port=port,
        distance_function=distance_function,
        **kwargs
    )