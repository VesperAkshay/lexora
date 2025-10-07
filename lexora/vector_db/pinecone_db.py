"""
Pinecone vector database implementation for the Lexora Agentic RAG SDK.

This module provides a Pinecone-based vector database implementation with
connection management, error handling, and efficient similarity search.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import time

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

from .base_vector_db import BaseVectorDB
from ..models.core import Document, SearchResult, CorpusInfo
from ..exceptions import LexoraError, ErrorCode, create_vector_db_error
from ..utils.embeddings import EmbeddingManager
from ..utils.logging import get_logger


class PineconeVectorDB(BaseVectorDB):
    """
    Pinecone implementation of vector database.
    
    This implementation uses Pinecone's managed vector database service for
    scalable similarity search with automatic scaling and management.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        embedding_manager: EmbeddingManager,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Pinecone vector database.
        
        Args:
            config: Configuration dictionary
            embedding_manager: Embedding manager for generating embeddings
            api_key: Pinecone API key
            environment: Pinecone environment (deprecated in newer versions)
            **kwargs: Additional configuration options
            
        Raises:
            LexoraError: If Pinecone is not available or configuration is invalid
        """
        if not PINECONE_AVAILABLE:
            raise create_vector_db_error(
                "Pinecone library not available. Install with: pip install pinecone-client",
                "Pinecone",
                error_code=ErrorCode.VECTOR_DB_CONNECTION_FAILED
            )
        
        super().__init__(config, **kwargs)
        
        self.embedding_manager = embedding_manager
        self.api_key = api_key or config.get('api_key')
        self.environment = environment or config.get('environment')
        
        # Pinecone client and index storage
        self.client: Optional[Pinecone] = None
        self._indices: Dict[str, Any] = {}  # corpus_name -> pinecone index
        self._corpus_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.dimension = embedding_manager.get_dimension()
        self.metric = kwargs.get('metric', 'cosine')
        self.cloud = kwargs.get('cloud', 'aws')
        self.region = kwargs.get('region', 'us-east-1')
        self.pod_type = kwargs.get('pod_type', 'p1.x1')  # For pod-based indexes
        self.replicas = kwargs.get('replicas', 1)
        self.shards = kwargs.get('shards', 1)
        
        # Use serverless by default for new indexes
        self.use_serverless = kwargs.get('use_serverless', True)
        
        self.logger = get_logger(self.__class__.__name__)
        
        if not self.api_key:
            raise create_vector_db_error(
                "Pinecone API key is required",
                "Pinecone",
                error_code=ErrorCode.VECTOR_DB_AUTHENTICATION_FAILED
            )
    
    async def _connect_impl(self) -> None:
        """Initialize Pinecone client and load existing indices."""
        try:
            # Initialize Pinecone client
            self.client = Pinecone(api_key=self.api_key)
            
            # Load existing indices
            await self._load_existing_indices()
            
            self.logger.info(f"Connected to Pinecone with {len(self._indices)} indices")
            
        except Exception as e:
            raise create_vector_db_error(
                f"Failed to connect to Pinecone: {str(e)}",
                "Pinecone",
                error_code=ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                original_error=e
            )
    
    async def _disconnect_impl(self) -> None:
        """Clean up Pinecone connection."""
        self._indices.clear()
        self._corpus_metadata.clear()
        self.client = None
    
    async def create_corpus(self, name: str, **kwargs) -> bool:
        """
        Create a new Pinecone index (corpus).
        
        Args:
            name: Name of the corpus/index to create
            **kwargs: Additional index creation parameters
            
        Returns:
            True if corpus was created successfully
            
        Raises:
            LexoraError: If corpus creation fails
        """
        if not self.client:
            raise create_vector_db_error(
                "Not connected to Pinecone",
                "Pinecone",
                name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED
            )
        
        if name in self._indices:
            raise create_vector_db_error(
                f"Index '{name}' already exists",
                "Pinecone",
                name,
                ErrorCode.VECTOR_DB_CORPUS_ALREADY_EXISTS
            )
        
        try:
            # Check if index already exists in Pinecone
            existing_indexes = self.client.list_indexes()
            if any(idx.name == name for idx in existing_indexes):
                raise create_vector_db_error(
                    f"Index '{name}' already exists in Pinecone",
                    "Pinecone",
                    name,
                    ErrorCode.VECTOR_DB_CORPUS_ALREADY_EXISTS
                )
            
            # Create index specification
            if self.use_serverless:
                spec = ServerlessSpec(
                    cloud=kwargs.get('cloud', self.cloud),
                    region=kwargs.get('region', self.region)
                )
            else:
                # Pod-based index (legacy)
                from pinecone import PodSpec
                spec = PodSpec(
                    environment=self.environment,
                    pod_type=kwargs.get('pod_type', self.pod_type),
                    pods=kwargs.get('pods', 1),
                    replicas=kwargs.get('replicas', self.replicas),
                    shards=kwargs.get('shards', self.shards)
                )
            
            # Create the index
            self.client.create_index(
                name=name,
                dimension=self.dimension,
                metric=kwargs.get('metric', self.metric),
                spec=spec
            )
            
            # Wait for index to be ready
            await self._wait_for_index_ready(name)
            
            # Connect to the index
            index = self.client.Index(name)
            self._indices[name] = index
            
            # Store metadata
            self._corpus_metadata[name] = {
                "created_at": datetime.utcnow(),
                "dimension": self.dimension,
                "metric": self.metric,
                "spec_type": "serverless" if self.use_serverless else "pod",
                "metadata": kwargs
            }
            
            self.logger.info(f"Created Pinecone index '{name}' with dimension {self.dimension}")
            return True
            
        except Exception as e:
            # Clean up on failure
            if name in self._indices:
                del self._indices[name]
            if name in self._corpus_metadata:
                del self._corpus_metadata[name]
            
            raise create_vector_db_error(
                f"Failed to create Pinecone index '{name}': {str(e)}",
                "Pinecone",
                name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                e
            )
    
    async def delete_corpus(self, name: str) -> bool:
        """
        Delete a Pinecone index (corpus).
        
        Args:
            name: Name of the corpus/index to delete
            
        Returns:
            True if corpus was deleted successfully
            
        Raises:
            LexoraError: If corpus deletion fails
        """
        if not self.client:
            raise create_vector_db_error(
                "Not connected to Pinecone",
                "Pinecone",
                name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED
            )
        
        if name not in self._indices:
            raise create_vector_db_error(
                f"Index '{name}' not found",
                "Pinecone",
                name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        try:
            # Delete the index from Pinecone
            self.client.delete_index(name)
            
            # Remove from local storage
            del self._indices[name]
            if name in self._corpus_metadata:
                del self._corpus_metadata[name]
            
            self.logger.info(f"Deleted Pinecone index '{name}'")
            return True
            
        except Exception as e:
            raise create_vector_db_error(
                f"Failed to delete Pinecone index '{name}': {str(e)}",
                "Pinecone",
                name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                e
            )
    
    async def add_documents(self, corpus_name: str, documents: List[Document]) -> bool:
        """
        Add documents to a Pinecone index.
        
        Args:
            corpus_name: Name of the corpus/index to add documents to
            documents: List of documents to add
            
        Returns:
            True if documents were added successfully
            
        Raises:
            LexoraError: If document addition fails
        """
        if corpus_name not in self._indices:
            raise create_vector_db_error(
                f"Index '{corpus_name}' not found",
                "Pinecone",
                corpus_name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        if not documents:
            return True
        
        try:
            index = self._indices[corpus_name]
            
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
            
            # Prepare vectors for Pinecone
            vectors = []
            for doc in docs_to_process:
                vector_data = {
                    "id": doc.id,
                    "values": doc.embedding,
                    "metadata": {
                        "content": doc.content,
                        "corpus_name": corpus_name,
                        **doc.metadata
                    }
                }
                vectors.append(vector_data)
            
            # Upsert vectors to Pinecone (batch operation)
            batch_size = 100  # Pinecone recommended batch size
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                index.upsert(vectors=batch)
            
            self.logger.info(f"Added {len(documents)} documents to Pinecone index '{corpus_name}'")
            return True
            
        except Exception as e:
            raise create_vector_db_error(
                f"Failed to add documents to Pinecone index '{corpus_name}': {str(e)}",
                "Pinecone",
                corpus_name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                e
            )
    
    async def search(self, corpus_name: str, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search for similar documents in a Pinecone index.
        
        Args:
            corpus_name: Name of the corpus/index to search in
            query: Search query text
            top_k: Maximum number of results to return
            
        Returns:
            List of search results ordered by relevance
            
        Raises:
            LexoraError: If search fails
        """
        if corpus_name not in self._indices:
            raise create_vector_db_error(
                f"Index '{corpus_name}' not found",
                "Pinecone",
                corpus_name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        try:
            index = self._indices[corpus_name]
            
            # Generate embedding for query
            query_embedding = await self.embedding_manager.generate_embedding(query)
            
            # Perform search
            search_response = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Convert results to SearchResult objects
            results = []
            for match in search_response.matches:
                # Extract document information from metadata
                metadata = match.metadata
                content = metadata.pop('content', '')
                stored_corpus_name = metadata.pop('corpus_name', corpus_name)
                
                # Create Document object
                document = Document(
                    id=match.id,
                    content=content,
                    metadata=metadata,
                    embedding=None  # Don't include embedding in search results
                )
                
                # Create SearchResult
                result = SearchResult(
                    document=document,
                    score=float(match.score),
                    corpus_name=stored_corpus_name
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            raise create_vector_db_error(
                f"Failed to search Pinecone index '{corpus_name}': {str(e)}",
                "Pinecone",
                corpus_name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                e
            )
    
    async def list_corpora(self) -> List[str]:
        """
        List all available Pinecone indices.
        
        Returns:
            List of corpus/index names
        """
        return list(self._indices.keys())
    
    async def get_corpus_info(self, name: str) -> CorpusInfo:
        """
        Get information about a specific Pinecone index.
        
        Args:
            name: Name of the corpus/index
            
        Returns:
            CorpusInfo object with corpus details
            
        Raises:
            LexoraError: If corpus doesn't exist
        """
        if name not in self._indices:
            raise create_vector_db_error(
                f"Index '{name}' not found",
                "Pinecone",
                name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        try:
            index = self._indices[name]
            
            # Get index statistics
            stats = index.describe_index_stats()
            
            metadata = self._corpus_metadata.get(name, {})
            
            return CorpusInfo(
                name=name,
                document_count=stats.total_vector_count,
                created_at=metadata.get("created_at", datetime.utcnow()),
                metadata={
                    **metadata.get("metadata", {}),
                    "dimension": metadata.get("dimension", self.dimension),
                    "metric": metadata.get("metric", self.metric),
                    "spec_type": metadata.get("spec_type", "unknown"),
                    "namespaces": stats.namespaces,
                    "index_fullness": stats.index_fullness
                }
            )
            
        except Exception as e:
            raise create_vector_db_error(
                f"Failed to get info for Pinecone index '{name}': {str(e)}",
                "Pinecone",
                name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                e
            )
    
    async def delete_document(self, corpus_name: str, document_id: str) -> bool:
        """
        Delete a specific document from a Pinecone index.
        
        Args:
            corpus_name: Name of the corpus/index
            document_id: ID of the document to delete
            
        Returns:
            True if document was deleted successfully
            
        Raises:
            LexoraError: If document deletion fails
        """
        if corpus_name not in self._indices:
            raise create_vector_db_error(
                f"Index '{corpus_name}' not found",
                "Pinecone",
                corpus_name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        try:
            index = self._indices[corpus_name]
            
            # Delete the vector by ID
            index.delete(ids=[document_id])
            
            self.logger.info(f"Deleted document '{document_id}' from Pinecone index '{corpus_name}'")
            return True
            
        except Exception as e:
            raise create_vector_db_error(
                f"Failed to delete document '{document_id}' from Pinecone index '{corpus_name}': {str(e)}",
                "Pinecone",
                corpus_name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                e
            )
    
    async def update_document(self, corpus_name: str, document: Document) -> bool:
        """
        Update an existing document in a Pinecone index.
        
        Args:
            corpus_name: Name of the corpus/index
            document: Updated document
            
        Returns:
            True if document was updated successfully
            
        Raises:
            LexoraError: If document update fails
        """
        if corpus_name not in self._indices:
            raise create_vector_db_error(
                f"Index '{corpus_name}' not found",
                "Pinecone",
                corpus_name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        try:
            # Generate embedding if not provided
            if document.embedding is None:
                document.embedding = await self.embedding_manager.generate_embedding(document.content)
            
            index = self._indices[corpus_name]
            
            # Upsert the updated document (upsert handles both insert and update)
            vector_data = {
                "id": document.id,
                "values": document.embedding,
                "metadata": {
                    "content": document.content,
                    "corpus_name": corpus_name,
                    **document.metadata
                }
            }
            
            index.upsert(vectors=[vector_data])
            
            self.logger.info(f"Updated document '{document.id}' in Pinecone index '{corpus_name}'")
            return True
            
        except Exception as e:
            raise create_vector_db_error(
                f"Failed to update document '{document.id}' in Pinecone index '{corpus_name}': {str(e)}",
                "Pinecone",
                corpus_name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                e
            )
    
    async def get_document(self, corpus_name: str, document_id: str) -> Optional[Document]:
        """
        Retrieve a specific document from a Pinecone index.
        
        Args:
            corpus_name: Name of the corpus/index
            document_id: ID of the document to retrieve
            
        Returns:
            Document if found, None otherwise
            
        Raises:
            LexoraError: If retrieval fails
        """
        if corpus_name not in self._indices:
            raise create_vector_db_error(
                f"Index '{corpus_name}' not found",
                "Pinecone",
                corpus_name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        try:
            index = self._indices[corpus_name]
            
            # Fetch the vector by ID
            fetch_response = index.fetch(ids=[document_id])
            
            if document_id not in fetch_response.vectors:
                return None
            
            vector_data = fetch_response.vectors[document_id]
            metadata = vector_data.metadata
            
            # Extract document information
            content = metadata.pop('content', '')
            metadata.pop('corpus_name', None)  # Remove corpus_name from metadata
            
            return Document(
                id=document_id,
                content=content,
                metadata=metadata,
                embedding=vector_data.values
            )
            
        except Exception as e:
            raise create_vector_db_error(
                f"Failed to get document '{document_id}' from Pinecone index '{corpus_name}': {str(e)}",
                "Pinecone",
                corpus_name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                e
            )
    
    # Pinecone-specific utility methods
    
    async def _wait_for_index_ready(self, index_name: str, timeout: int = 300) -> None:
        """
        Wait for a Pinecone index to be ready.
        
        Args:
            index_name: Name of the index to wait for
            timeout: Maximum time to wait in seconds
            
        Raises:
            LexoraError: If index doesn't become ready within timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                index_description = self.client.describe_index(index_name)
                if index_description.status.ready:
                    return
                
                await asyncio.sleep(5)  # Wait 5 seconds before checking again
                
            except Exception as e:
                self.logger.warning(f"Error checking index status: {str(e)}")
                await asyncio.sleep(5)
        
        raise create_vector_db_error(
            f"Index '{index_name}' did not become ready within {timeout} seconds",
            "Pinecone",
            index_name,
            ErrorCode.VECTOR_DB_OPERATION_TIMEOUT
        )
    
    async def _load_existing_indices(self) -> None:
        """Load existing Pinecone indices."""
        try:
            existing_indexes = self.client.list_indexes()
            
            for index_info in existing_indexes:
                index_name = index_info.name
                
                # Connect to existing index
                index = self.client.Index(index_name)
                self._indices[index_name] = index
                
                # Store basic metadata (we don't have creation time from Pinecone)
                self._corpus_metadata[index_name] = {
                    "created_at": datetime.utcnow(),  # Approximate
                    "dimension": index_info.dimension,
                    "metric": index_info.metric,
                    "spec_type": "serverless" if hasattr(index_info.spec, 'serverless') else "pod",
                    "metadata": {}
                }
            
        except Exception as e:
            self.logger.warning(f"Failed to load existing Pinecone indices: {str(e)}")
    
    def get_index_info(self, corpus_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a Pinecone index.
        
        Args:
            corpus_name: Name of the corpus/index
            
        Returns:
            Dictionary with index information
            
        Raises:
            LexoraError: If corpus doesn't exist
        """
        if corpus_name not in self._indices:
            raise create_vector_db_error(
                f"Index '{corpus_name}' not found",
                "Pinecone",
                corpus_name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        try:
            index = self._indices[corpus_name]
            stats = index.describe_index_stats()
            index_description = self.client.describe_index(corpus_name)
            
            return {
                "name": corpus_name,
                "dimension": index_description.dimension,
                "metric": index_description.metric,
                "total_vector_count": stats.total_vector_count,
                "index_fullness": stats.index_fullness,
                "namespaces": stats.namespaces,
                "spec": index_description.spec,
                "status": index_description.status,
                "host": index_description.host
            }
            
        except Exception as e:
            raise create_vector_db_error(
                f"Failed to get info for Pinecone index '{corpus_name}': {str(e)}",
                "Pinecone",
                corpus_name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                e
            )


# Convenience functions

def create_pinecone_vector_db(
    embedding_manager: EmbeddingManager,
    api_key: str,
    environment: Optional[str] = None,
    metric: str = 'cosine',
    use_serverless: bool = True,
    cloud: str = 'aws',
    region: str = 'us-east-1',
    **kwargs
) -> PineconeVectorDB:
    """
    Create a Pinecone vector database with common configuration.
    
    Args:
        embedding_manager: Embedding manager for generating embeddings
        api_key: Pinecone API key
        environment: Pinecone environment (deprecated)
        metric: Distance metric to use ('cosine', 'euclidean', 'dotproduct')
        use_serverless: Whether to use serverless indexes
        cloud: Cloud provider for serverless ('aws', 'gcp', 'azure')
        region: Region for serverless indexes
        **kwargs: Additional configuration options
        
    Returns:
        Configured Pinecone vector database
    """
    config = {
        "provider": "pinecone",
        "api_key": api_key,
        "environment": environment,
        "metric": metric
    }
    
    return PineconeVectorDB(
        config=config,
        embedding_manager=embedding_manager,
        api_key=api_key,
        environment=environment,
        metric=metric,
        use_serverless=use_serverless,
        cloud=cloud,
        region=region,
        **kwargs
    )