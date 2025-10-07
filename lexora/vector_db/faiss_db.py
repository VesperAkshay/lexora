"""
FAISS vector database implementation for the Lexora Agentic RAG SDK.

This module provides a FAISS-based vector database implementation with
persistence, index management, and efficient similarity search.
"""

import os
import json
import pickle
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from .base_vector_db import BaseVectorDB
from ..models.core import Document, SearchResult, CorpusInfo
from ..exceptions import LexoraError, ErrorCode, create_vector_db_error
from ..utils.embeddings import EmbeddingManager
from ..utils.logging import get_logger


class FAISSVectorDB(BaseVectorDB):
    """
    FAISS implementation of vector database.
    
    This implementation uses Facebook AI Similarity Search (FAISS) for efficient
    vector similarity search with support for persistence and index management.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        embedding_manager: EmbeddingManager,
        storage_path: str = "./faiss_storage",
        index_type: str = "IndexFlatIP",  # Inner Product (cosine similarity for normalized vectors)
        **kwargs
    ):
        """
        Initialize FAISS vector database.
        
        Args:
            config: Configuration dictionary
            embedding_manager: Embedding manager for generating embeddings
            storage_path: Path to store FAISS indices and metadata
            index_type: Type of FAISS index to use
            **kwargs: Additional configuration options
            
        Raises:
            LexoraError: If FAISS is not available or configuration is invalid
        """
        if not FAISS_AVAILABLE:
            raise create_vector_db_error(
                "FAISS library not available. Install with: pip install faiss-cpu or faiss-gpu",
                "FAISS",
                error_code=ErrorCode.VECTOR_DB_CONNECTION_FAILED
            )
        
        super().__init__(config, **kwargs)
        
        self.embedding_manager = embedding_manager
        self.storage_path = Path(storage_path)
        self.index_type = index_type
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage for indices and metadata
        self._indices: Dict[str, faiss.Index] = {}
        self._documents: Dict[str, Dict[str, Document]] = {}  # corpus_name -> {doc_id -> Document}
        self._id_mappings: Dict[str, Dict[int, str]] = {}  # corpus_name -> {faiss_id -> doc_id}
        self._corpus_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.dimension = embedding_manager.get_dimension()
        self.normalize_embeddings = kwargs.get('normalize_embeddings', True)
        self.auto_save = kwargs.get('auto_save', True)
        
        self.logger = get_logger(self.__class__.__name__)
    
    async def _connect_impl(self) -> None:
        """Load existing indices and metadata from disk."""
        try:
            await self._load_all_corpora()
            self.logger.info(f"Loaded {len(self._indices)} corpora from {self.storage_path}")
        except Exception as e:
            self.logger.warning(f"Failed to load existing corpora: {str(e)}")
    
    async def _disconnect_impl(self) -> None:
        """Save all indices and metadata to disk."""
        if self.auto_save:
            try:
                await self._save_all_corpora()
                self.logger.info("Saved all corpora to disk")
            except Exception as e:
                self.logger.warning(f"Failed to save corpora: {str(e)}")
    
    async def create_corpus(self, name: str, **kwargs) -> bool:
        """
        Create a new FAISS corpus.
        
        Args:
            name: Name of the corpus to create
            **kwargs: Additional corpus creation parameters
            
        Returns:
            True if corpus was created successfully
            
        Raises:
            LexoraError: If corpus creation fails
        """
        if name in self._indices:
            raise create_vector_db_error(
                f"Corpus '{name}' already exists",
                "FAISS",
                name,
                ErrorCode.VECTOR_DB_CORPUS_ALREADY_EXISTS  # or appropriate code
            )        
        try:
            # Create FAISS index
            index = self._create_faiss_index()
            
            # Initialize storage structures
            self._indices[name] = index
            self._documents[name] = {}
            self._id_mappings[name] = {}
            self._corpus_metadata[name] = {
                "created_at": datetime.now(timezone.utc),
                "index_type": self.index_type,
                "dimension": self.dimension,
                "metadata": kwargs
            }
            
            # Save to disk if auto_save is enabled
            if self.auto_save:
                await self._save_corpus(name)
            
            self.logger.info(f"Created corpus '{name}' with {self.index_type} index")
            return True
            
        except Exception as e:
            # Clean up on failure
            self._cleanup_corpus(name)
            raise create_vector_db_error(
                f"Failed to create corpus '{name}': {str(e)}",
                "FAISS",
                name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                e
            )
    
    async def delete_corpus(self, name: str) -> bool:
        """
        Delete a FAISS corpus.
        
        Args:
            name: Name of the corpus to delete
            
        Returns:
            True if corpus was deleted successfully
            
        Raises:
            LexoraError: If corpus deletion fails
        """
        if name not in self._indices:
            raise create_vector_db_error(
                f"Corpus '{name}' not found",
                "FAISS",
                name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        try:
            # Remove from memory
            self._cleanup_corpus(name)
            
            # Remove from disk
            await self._delete_corpus_files(name)
            
            self.logger.info(f"Deleted corpus '{name}'")
            return True
            
        except Exception as e:
            raise create_vector_db_error(
                f"Failed to delete corpus '{name}': {str(e)}",
                "FAISS",
                name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                e
            )
    
    async def add_documents(self, corpus_name: str, documents: List[Document]) -> bool:
        """
        Add documents to a FAISS corpus.
        
        Args:
            corpus_name: Name of the corpus to add documents to
            documents: List of documents to add
            
        Returns:
            True if documents were added successfully
            
        Raises:
            LexoraError: If document addition fails
        """
        if corpus_name not in self._indices:
            raise create_vector_db_error(
                f"Corpus '{corpus_name}' not found",
                "FAISS",
                corpus_name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        if not documents:
            return True
        
        try:
            index = self._indices[corpus_name]
            documents_dict = self._documents[corpus_name]
            id_mapping = self._id_mappings[corpus_name]
            
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
            
            # Prepare embeddings for FAISS
            embeddings_array = np.array([doc.embedding for doc in docs_to_process], dtype=np.float32)
            
            # Normalize embeddings if configured
            if self.normalize_embeddings:
                embeddings_array = self._normalize_embeddings(embeddings_array)
            
            # Train index if needed (for IndexIVFFlat and similar)
            if not index.is_trained:
                # Train index on the first batch of vectors
                # Use up to nlist * 39 vectors for training (FAISS recommendation)
                nlist = 100  # Should match the nlist used in index creation
                training_data = embeddings_array[:min(len(embeddings_array), nlist * 39)]
                self.logger.info(f"Training index with {len(training_data)} vectors")
                index.train(training_data)
            
            # Add to FAISS index
            start_id = index.ntotal
            index.add(embeddings_array)
            
            # Update mappings and document storage
            for i, doc in enumerate(docs_to_process):
                faiss_id = start_id + i
                id_mapping[faiss_id] = doc.id
                documents_dict[doc.id] = doc
            
            # Save to disk if auto_save is enabled
            if self.auto_save:
                await self._save_corpus(corpus_name)
            
            self.logger.info(f"Added {len(documents)} documents to corpus '{corpus_name}'")
            return True
            
        except Exception as e:
            raise create_vector_db_error(
                f"Failed to add documents to corpus '{corpus_name}': {str(e)}",
                "FAISS",
                corpus_name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                e
            )
    
    async def search(self, corpus_name: str, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search for similar documents in a FAISS corpus.
        
        Args:
            corpus_name: Name of the corpus to search in
            query: Search query text
            top_k: Maximum number of results to return
            
        Returns:
            List of search results ordered by relevance
            
        Raises:
            LexoraError: If search fails
        """
        if corpus_name not in self._indices:
            raise create_vector_db_error(
                f"Corpus '{corpus_name}' not found",
                "FAISS",
                corpus_name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        try:
            index = self._indices[corpus_name]
            documents_dict = self._documents[corpus_name]
            id_mapping = self._id_mappings[corpus_name]
            
            if index.ntotal == 0:
                return []  # Empty corpus
            
            # Generate embedding for query
            query_embedding = await self.embedding_manager.generate_embedding(query)
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Normalize query embedding if configured
            if self.normalize_embeddings:
                query_vector = self._normalize_embeddings(query_vector)
            
            # Perform search
            top_k = min(top_k, index.ntotal)  # Don't search for more than available
            scores, indices = index.search(query_vector, top_k)
            
            # Convert results to SearchResult objects
            results = []
            for i in range(len(indices[0])):
                faiss_id = indices[0][i]
                score = float(scores[0][i])
                
                # Skip invalid results (FAISS returns -1 for invalid indices)
                if faiss_id == -1:
                    continue
                
                # Get document ID and document
                doc_id = id_mapping.get(faiss_id)
                if doc_id and doc_id in documents_dict:
                    document = documents_dict[doc_id]
                    
                    # Convert FAISS score to normalized similarity based on index type
                    # Different index types use different distance/similarity metrics
                    if self.index_type == "IndexFlatIP" and self.normalize_embeddings:
                        # Inner Product with normalized embeddings: score is cosine similarity [-1, 1]
                        # Map to [0, 1] range where 1 is most similar
                        normalized_score = max(0.0, min(1.0, (score + 1.0) / 2.0))
                    elif self.index_type == "IndexFlatL2" or self.index_type == "IndexIVFFlat":
                        # L2 distance: smaller distances mean higher similarity
                        # Convert distance to similarity using inverse decay: 1 / (1 + distance)
                        normalized_score = 1.0 / (1.0 + score)
                        # Clamp to [0, 1] range
                        normalized_score = max(0.0, min(1.0, normalized_score))
                    else:
                        # Unknown or other index types: return raw score clamped to [0, 1]
                        # Note: No normalization applied for unknown index types
                        normalized_score = max(0.0, min(1.0, score))
                    
                    results.append(SearchResult(
                        document=document,
                        score=normalized_score,
                        corpus_name=corpus_name
                    ))
            
            return results
            
        except Exception as e:
            raise create_vector_db_error(
                f"Failed to search corpus '{corpus_name}': {str(e)}",
                "FAISS",
                corpus_name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                e
            )
    
    async def list_corpora(self) -> List[str]:
        """
        List all available FAISS corpora.
        
        Returns:
            List of corpus names
        """
        return list(self._indices.keys())
    
    async def get_corpus_info(self, name: str) -> CorpusInfo:
        """
        Get information about a specific FAISS corpus.
        
        Args:
            name: Name of the corpus
            
        Returns:
            CorpusInfo object with corpus details
            
        Raises:
            LexoraError: If corpus doesn't exist
        """
        if name not in self._indices:
            raise create_vector_db_error(
                f"Corpus '{name}' not found",
                "FAISS",
                name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        metadata = self._corpus_metadata[name]
        document_count = len(self._documents[name])
        
        return CorpusInfo(
            name=name,
            document_count=document_count,
            created_at=metadata["created_at"],
            metadata={
                **metadata["metadata"],
                "index_type": metadata["index_type"],
                "dimension": metadata["dimension"],
                "faiss_total": self._indices[name].ntotal
            }
        )
    
    async def delete_document(self, corpus_name: str, document_id: str) -> bool:
        """
        Delete a specific document from a FAISS corpus.
        
        Note: FAISS doesn't support efficient single document deletion.
        This implementation removes the document from metadata but keeps
        the vector in the index for performance reasons.
        
        Args:
            corpus_name: Name of the corpus
            document_id: ID of the document to delete
            
        Returns:
            True if document was deleted successfully
            
        Raises:
            LexoraError: If document deletion fails
        """
        if corpus_name not in self._indices:
            raise create_vector_db_error(
                f"Corpus '{corpus_name}' not found",
                "FAISS",
                corpus_name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        documents_dict = self._documents[corpus_name]
        
        if document_id not in documents_dict:
            raise create_vector_db_error(
                f"Document '{document_id}' not found in corpus '{corpus_name}'",
                "FAISS",
                corpus_name,
                ErrorCode.VECTOR_DB_DOCUMENT_NOT_FOUND
            )
        
        try:
            # Remove from document storage
            del documents_dict[document_id]
            
            # Note: We don't remove from FAISS index or id_mapping for performance
            # The vector remains in the index but won't be returned in search results
            # since the document is no longer in documents_dict
            
            # Save to disk if auto_save is enabled
            if self.auto_save:
                await self._save_corpus(corpus_name)
            
            self.logger.info(f"Deleted document '{document_id}' from corpus '{corpus_name}'")
            return True
            
        except Exception as e:
            raise create_vector_db_error(
                f"Failed to delete document '{document_id}' from corpus '{corpus_name}': {str(e)}",
                "FAISS",
                corpus_name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                e
            )
    
    async def update_document(self, corpus_name: str, document: Document) -> bool:
        """
        Update an existing document in a FAISS corpus.
        
        Note: FAISS doesn't support efficient updates. This implementation
        updates the document metadata but doesn't update the vector in the index.
        For vector updates, delete and re-add the document.
        
        Args:
            corpus_name: Name of the corpus
            document: Updated document
            
        Returns:
            True if document was updated successfully
            
        Raises:
            LexoraError: If document update fails
        """
        if corpus_name not in self._indices:
            raise create_vector_db_error(
                f"Corpus '{corpus_name}' not found",
                "FAISS",
                corpus_name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        documents_dict = self._documents[corpus_name]
        
        if document.id not in documents_dict:
            raise create_vector_db_error(
                f"Document '{document.id}' not found in corpus '{corpus_name}'",
                "FAISS",
                corpus_name,
                ErrorCode.VECTOR_DB_DOCUMENT_NOT_FOUND
            )
        
        try:
            # Update document in storage
            documents_dict[document.id] = document
            
            # Save to disk if auto_save is enabled
            if self.auto_save:
                await self._save_corpus(corpus_name)
            
            self.logger.info(f"Updated document '{document.id}' in corpus '{corpus_name}'")
            return True
            
        except Exception as e:
            raise create_vector_db_error(
                f"Failed to update document '{document.id}' in corpus '{corpus_name}': {str(e)}",
                "FAISS",
                corpus_name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                e
            )
    
    async def get_document(self, corpus_name: str, document_id: str) -> Optional[Document]:
        """
        Retrieve a specific document from a FAISS corpus.
        
        Args:
            corpus_name: Name of the corpus
            document_id: ID of the document to retrieve
            
        Returns:
            Document if found, None otherwise
            
        Raises:
            LexoraError: If retrieval fails
        """
        if corpus_name not in self._indices:
            raise create_vector_db_error(
                f"Corpus '{corpus_name}' not found",
                "FAISS",
                corpus_name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        documents_dict = self._documents[corpus_name]
        return documents_dict.get(document_id)
    
    # FAISS-specific utility methods
    
    def _create_faiss_index(self):
        """
        Create a FAISS index based on configuration.
        
        Returns:
            Configured FAISS index
        """
        if self.index_type == "IndexFlatIP":
            # Inner Product index (good for normalized vectors)
            return faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IndexFlatL2":
            # L2 distance index
            return faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IndexIVFFlat":
            # IVF (Inverted File) index for faster search on large datasets
            # Note: This index requires training with sample data before use
            # Training should be done in add_documents when first vectors are added
            quantizer = faiss.IndexFlatL2(self.dimension)
            nlist = 100  # Number of clusters
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            # Index will need training before first use - caller must check index.is_trained
            return index
        else:
            # Default to flat inner product
            return faiss.IndexFlatIP(self.dimension)
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings to unit length.
        
        Args:
            embeddings: Array of embeddings to normalize
            
        Returns:
            Normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms
    
    def _cleanup_corpus(self, name: str) -> None:
        """Clean up corpus data from memory."""
        if name in self._indices:
            del self._indices[name]
        if name in self._documents:
            del self._documents[name]
        if name in self._id_mappings:
            del self._id_mappings[name]
        if name in self._corpus_metadata:
            del self._corpus_metadata[name]
    
    # Persistence methods
    
    async def _save_corpus(self, name: str) -> None:
        """Save a corpus to disk."""
        corpus_dir = self.storage_path / name
        corpus_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = corpus_dir / "index.faiss"
        faiss.write_index(self._indices[name], str(index_path))
        
        # Save metadata
        metadata_path = corpus_dir / "metadata.json"
        metadata = {
            **self._corpus_metadata[name],
            "created_at": self._corpus_metadata[name]["created_at"].isoformat()
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save documents and ID mappings
        data_path = corpus_dir / "data.pkl"
        data = {
            "documents": self._documents[name],
            "id_mappings": self._id_mappings[name]
        }
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
    
    async def _load_corpus(self, name: str) -> None:
        """Load a corpus from disk."""
        corpus_dir = self.storage_path / name
        
        if not corpus_dir.exists():
            return
        
        try:
            # Load FAISS index
            index_path = corpus_dir / "index.faiss"
            if index_path.exists():
                self._indices[name] = faiss.read_index(str(index_path))
            
            # Load metadata
            metadata_path = corpus_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    # Convert created_at back to datetime
                    metadata["created_at"] = datetime.fromisoformat(metadata["created_at"])
                    self._corpus_metadata[name] = metadata
            
            # Load documents and ID mappings
            data_path = corpus_dir / "data.pkl"
            if data_path.exists():
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                    self._documents[name] = data["documents"]
                    self._id_mappings[name] = data["id_mappings"]
            
        except Exception as e:
            self.logger.warning(f"Failed to load corpus '{name}': {str(e)}")
            self._cleanup_corpus(name)
    
    async def _save_all_corpora(self) -> None:
        """Save all corpora to disk."""
        for name in self._indices.keys():
            await self._save_corpus(name)
    
    async def _load_all_corpora(self) -> None:
        """Load all corpora from disk."""
        if not self.storage_path.exists():
            return
        
        for corpus_dir in self.storage_path.iterdir():
            if corpus_dir.is_dir():
                await self._load_corpus(corpus_dir.name)
    
    async def _delete_corpus_files(self, name: str) -> None:
        """Delete corpus files from disk."""
        corpus_dir = self.storage_path / name
        if corpus_dir.exists():
            import shutil
            shutil.rmtree(corpus_dir)
    
    # Additional FAISS-specific methods
    
    def get_index_info(self, corpus_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a FAISS index.
        
        Args:
            corpus_name: Name of the corpus
            
        Returns:
            Dictionary with index information
            
        Raises:
            LexoraError: If corpus doesn't exist
        """
        if corpus_name not in self._indices:
            raise create_vector_db_error(
                f"Corpus '{corpus_name}' not found",
                "FAISS",
                corpus_name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        index = self._indices[corpus_name]
        
        return {
            "index_type": self.index_type,
            "dimension": index.d,
            "total_vectors": index.ntotal,
            "is_trained": index.is_trained,
            "metric_type": index.metric_type,
            "documents_count": len(self._documents[corpus_name]),
            "storage_path": str(self.storage_path / corpus_name)
        }
    
    async def rebuild_corpus(self, corpus_name: str) -> bool:
        """
        Rebuild a corpus index (useful for cleaning up deleted documents).
        
        Args:
            corpus_name: Name of the corpus to rebuild
            
        Returns:
            True if rebuild was successful
            
        Raises:
            LexoraError: If rebuild fails
        """
        if corpus_name not in self._indices:
            raise create_vector_db_error(
                f"Corpus '{corpus_name}' not found",
                "FAISS",
                corpus_name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        try:
            documents = list(self._documents[corpus_name].values())
            
            # Create new index
            new_index = self._create_faiss_index()
            
            # Clear existing mappings
            self._indices[corpus_name] = new_index
            self._id_mappings[corpus_name] = {}
            
            # Re-add all documents
            if documents:
                # Clear embeddings to force regeneration
                for doc in documents:
                    doc.embedding = None
                
                await self.add_documents(corpus_name, documents)
            
            self.logger.info(f"Rebuilt corpus '{corpus_name}' with {len(documents)} documents")
            return True
            
        except Exception as e:
            raise create_vector_db_error(
                f"Failed to rebuild corpus '{corpus_name}': {str(e)}",
                "FAISS",
                corpus_name,
                ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                e
            )


# Convenience functions

def create_faiss_vector_db(
    embedding_manager: EmbeddingManager,
    storage_path: str = "./faiss_storage",
    index_type: str = "IndexFlatIP",
    normalize_embeddings: bool = True,
    auto_save: bool = True,
    **kwargs
) -> FAISSVectorDB:
    """
    Create a FAISS vector database with common configuration.
    
    Args:
        embedding_manager: Embedding manager for generating embeddings
        storage_path: Path to store FAISS indices and metadata
        index_type: Type of FAISS index to use
        normalize_embeddings: Whether to normalize embeddings
        auto_save: Whether to automatically save changes to disk
        **kwargs: Additional configuration options
        
    Returns:
        Configured FAISS vector database
    """
    config = {
        "provider": "faiss",
        "storage_path": storage_path,
        "index_type": index_type
    }
    
    return FAISSVectorDB(
        config=config,
        embedding_manager=embedding_manager,
        storage_path=storage_path,
        index_type=index_type,
        normalize_embeddings=normalize_embeddings,
        auto_save=auto_save,
        **kwargs
    )