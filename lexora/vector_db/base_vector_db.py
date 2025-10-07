"""
Base vector database interface for the Lexora Agentic RAG SDK.

This module provides the abstract base class that all vector database providers
must implement, ensuring consistent interfaces across different vector database services.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import asyncio
import time
from datetime import datetime

from ..models.core import Document, SearchResult, CorpusInfo
from ..exceptions import LexoraError, ErrorCode, create_vector_db_error
from ..utils.logging import get_logger


class BaseVectorDB(ABC):
    """
    Abstract base class for vector database providers.
    
    This class defines the interface that all vector database providers must implement
    to ensure consistent behavior across different vector database services.
    """
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        """
        Initialize the vector database provider.
        
        Args:
            config: Configuration dictionary for the provider
            **kwargs: Additional provider-specific configuration options
        """
        self.config = config
        self.provider_config = kwargs
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize connection state
        self._connected = False
        self._connection = None
    
    @abstractmethod
    async def create_corpus(self, name: str, **kwargs) -> bool:
        """
        Create a new document corpus.
        
        Args:
            name: Name of the corpus to create
            **kwargs: Additional corpus creation parameters
            
        Returns:
            True if corpus was created successfully
            
        Raises:
            LexoraError: If corpus creation fails
        """
        pass
    
    @abstractmethod
    async def delete_corpus(self, name: str) -> bool:
        """
        Delete an existing corpus.
        
        Args:
            name: Name of the corpus to delete
            
        Returns:
            True if corpus was deleted successfully
            
        Raises:
            LexoraError: If corpus deletion fails
        """
        pass
    
    @abstractmethod
    async def add_documents(self, corpus_name: str, documents: List[Document]) -> bool:
        """
        Add documents to a corpus.
        
        Args:
            corpus_name: Name of the corpus to add documents to
            documents: List of documents to add
            
        Returns:
            True if documents were added successfully
            
        Raises:
            LexoraError: If document addition fails
        """
        pass
    
    @abstractmethod
    async def search(self, corpus_name: str, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search for similar documents in a corpus.
        
        Args:
            corpus_name: Name of the corpus to search in
            query: Search query text
            top_k: Maximum number of results to return
            
        Returns:
            List of search results ordered by relevance
            
        Raises:
            LexoraError: If search fails
        """
        pass
    
    @abstractmethod
    async def list_corpora(self) -> List[str]:
        """
        List all available corpora.
        
        Returns:
            List of corpus names
            
        Raises:
            LexoraError: If listing fails
        """
        pass
    
    @abstractmethod
    async def get_corpus_info(self, name: str) -> CorpusInfo:
        """
        Get information about a specific corpus.
        
        Args:
            name: Name of the corpus
            
        Returns:
            CorpusInfo object with corpus details
            
        Raises:
            LexoraError: If corpus doesn't exist or info retrieval fails
        """
        pass
    
    # Additional abstract methods for document management
    
    @abstractmethod
    async def delete_document(self, corpus_name: str, document_id: str) -> bool:
        """
        Delete a specific document from a corpus.
        
        Args:
            corpus_name: Name of the corpus
            document_id: ID of the document to delete
            
        Returns:
            True if document was deleted successfully
            
        Raises:
            LexoraError: If document deletion fails
        """
        pass
    
    @abstractmethod
    async def update_document(self, corpus_name: str, document: Document) -> bool:
        """
        Update an existing document in a corpus.
        
        Args:
            corpus_name: Name of the corpus
            document: Updated document
            
        Returns:
            True if document was updated successfully
            
        Raises:
            LexoraError: If document update fails
        """
        pass
    
    @abstractmethod
    async def get_document(self, corpus_name: str, document_id: str) -> Optional[Document]:
        """
        Retrieve a specific document from a corpus.
        
        Args:
            corpus_name: Name of the corpus
            document_id: ID of the document to retrieve
            
        Returns:
            Document if found, None otherwise
            
        Raises:
            LexoraError: If retrieval fails
        """
        pass
    
    # Connection management methods
    
    async def connect(self) -> None:
        """
        Establish connection to the vector database.
        
        Raises:
            LexoraError: If connection fails
        """
        if self._connected:
            return
        
        try:
            await self._connect_impl()
            self._connected = True
            self.logger.info(f"Connected to {self.__class__.__name__}")
        except Exception as e:
            raise create_vector_db_error(
                f"Failed to connect to vector database: {str(e)}",
                self.__class__.__name__,
                error_code=ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                original_error=e
            )
    
    async def disconnect(self) -> None:
        """
        Close connection to the vector database.
        
        Raises:
            LexoraError: If disconnection fails
        """
        if not self._connected:
            return
        
        try:
            await self._disconnect_impl()
            self._connected = False
            self.logger.info(f"Disconnected from {self.__class__.__name__}")
        except Exception as e:
            self.logger.warning(f"Error during disconnection: {str(e)}")
    
    @abstractmethod
    async def _connect_impl(self) -> None:
        """
        Provider-specific connection implementation.
        
        Raises:
            Exception: If connection fails
        """
        pass
    
    @abstractmethod
    async def _disconnect_impl(self) -> None:
        """
        Provider-specific disconnection implementation.
        
        Raises:
            Exception: If disconnection fails
        """
        pass
    
    # Health check and status methods
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the vector database.
        
        Returns:
            Dictionary with health status information
        """
        try:
            start_time = time.time()
            
            # Test basic connectivity
            if not self._connected:
                await self.connect()
            
            # Test basic operations
            corpora = await self.list_corpora()
            
            duration = time.time() - start_time
            
            return {
                "status": "healthy",
                "connected": self._connected,
                "response_time": duration,
                "corpora_count": len(corpora),
                "provider": self.__class__.__name__,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": self._connected,
                "error": str(e),
                "provider": self.__class__.__name__,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def is_connected(self) -> bool:
        """
        Check if the database is connected.
        
        Returns:
            True if connected, False otherwise
        """
        return self._connected
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()
    
    def get_provider_name(self) -> str:
        """
        Get the name of the vector database provider.
        
        Returns:
            Provider name
        """
        return self.__class__.__name__
    
    # Utility methods with retry logic
    
    async def search_with_retry(
        self,
        corpus_name: str,
        query: str,
        top_k: int = 10,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0
    ) -> List[SearchResult]:
        """
        Search with automatic retry logic.
        
        Args:
            corpus_name: Name of the corpus to search in
            query: Search query text
            top_k: Maximum number of results to return
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries in seconds
            backoff_factor: Multiplier for delay after each retry
            
        Returns:
            List of search results
            
        Raises:
            LexoraError: If all retry attempts fail
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                results = await self.search(corpus_name, query, top_k)
                duration = time.time() - start_time
                
                self.logger.log_vector_operation(
                    "search",
                    corpus_name,
                    document_count=len(results),
                    duration=duration
                )
                
                return results
                
            except LexoraError as e:
                last_error = e
                
                # Don't retry on certain error types
                if e.error_code in (ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND, ErrorCode.VECTOR_DB_AUTHENTICATION_FAILED):
                    raise e
                
                if attempt < max_retries:
                    delay = retry_delay * (backoff_factor ** attempt)
                    self.logger.warning(
                        f"Vector search failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.1f}s",
                        error=str(e),
                        attempt=attempt + 1,
                        max_retries=max_retries + 1
                    )
                    await asyncio.sleep(delay)
                else:
                    self.logger.log_vector_operation(
                        "search",
                        corpus_name,
                        error=e
                    )
        
        # If we get here, all retries failed
        raise create_vector_db_error(
            f"Vector search failed after {max_retries + 1} attempts: {str(last_error)}",
            self.__class__.__name__,
            corpus_name,
            ErrorCode.VECTOR_DB_OPERATION_TIMEOUT,
            last_error
        )
    
    async def add_documents_batch(
        self,
        corpus_name: str,
        documents: List[Document],
        batch_size: int = 100
    ) -> bool:
        """
        Add documents in batches for better performance.
        
        Args:
            corpus_name: Name of the corpus to add documents to
            documents: List of documents to add
            batch_size: Number of documents per batch
            
        Returns:
            True if all documents were added successfully
            
        Raises:
            LexoraError: If any batch fails
        """
        if not documents:
            return True
        
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            try:
                start_time = time.time()
                success = await self.add_documents(corpus_name, batch)
                duration = time.time() - start_time
                
                if success:
                    self.logger.info(
                        f"Added batch {batch_num}/{total_batches} ({len(batch)} documents) to corpus '{corpus_name}'",
                        batch_size=len(batch),
                        batch_number=batch_num,
                        total_batches=total_batches,
                        duration=duration
                    )
                else:
                    raise create_vector_db_error(
                        f"Failed to add batch {batch_num}/{total_batches} to corpus '{corpus_name}'",
                        self.__class__.__name__,
                        corpus_name,
                        ErrorCode.VECTOR_DB_OPERATION_TIMEOUT
                    )
                    
            except Exception as e:
                self.logger.log_vector_operation(
                    "add_documents_batch",
                    corpus_name,
                    document_count=len(batch),
                    error=e
                )
                raise
        
        return True
    
    def validate_config(self) -> None:
        """
        Validate the provider configuration.
        
        Raises:
            LexoraError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise create_vector_db_error(
                "Configuration must be a dictionary",
                self.__class__.__name__,
                error_code=ErrorCode.INVALID_CONFIG
            )
    
    def __str__(self) -> str:
        """Return string representation of the vector database provider."""
        return f"{self.__class__.__name__}(connected={self._connected})"
    
    def __repr__(self) -> str:
        """Return detailed representation of the vector database provider."""
        return f"{self.__class__.__name__}(config={self.config}, connected={self._connected})"
    
    # Context manager support
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


class MockVectorDB(BaseVectorDB):
    """
    Mock vector database provider for testing and development.
    
    This provider stores data in memory and provides deterministic behavior
    for testing purposes without requiring external services.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize mock vector database provider.
        
        Args:
            config: Configuration dictionary (optional)
            **kwargs: Additional configuration options
        """
        super().__init__(config or {}, **kwargs)
        
        # In-memory storage
        self._corpora: Dict[str, Dict[str, Document]] = {}
        self._corpus_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.simulate_delay = kwargs.get('simulate_delay', 0.01)
        self.fail_probability = kwargs.get('fail_probability', 0.0)
    
    async def _connect_impl(self) -> None:
        """Mock connection implementation."""
        if self.simulate_delay > 0:
            await asyncio.sleep(self.simulate_delay)
    
    async def _disconnect_impl(self) -> None:
        """Mock disconnection implementation."""
        if self.simulate_delay > 0:
            await asyncio.sleep(self.simulate_delay)
    
    async def create_corpus(self, name: str, **kwargs) -> bool:
        """Create a mock corpus."""
        await self._simulate_operation()
        
        if name in self._corpora:
            raise create_vector_db_error(
                f"Corpus '{name}' already exists",
                self.__class__.__name__,
                name,
                ErrorCode.VECTOR_DB_CORPUS_ALREADY_EXISTS # OR APPROPIRATE ERROR CODE
            )
        
        self._corpora[name] = {}
        self._corpus_metadata[name] = {
            "created_at": datetime.utcnow(),
            "metadata": kwargs
        }
        
        return True
    
    async def delete_corpus(self, name: str) -> bool:
        """Delete a mock corpus."""
        await self._simulate_operation()
        
        if name not in self._corpora:
            raise create_vector_db_error(
                f"Corpus '{name}' not found",
                self.__class__.__name__,
                name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        del self._corpora[name]
        del self._corpus_metadata[name]
        
        return True
    
    async def add_documents(self, corpus_name: str, documents: List[Document]) -> bool:
        """Add documents to a mock corpus."""
        await self._simulate_operation()
        
        if corpus_name not in self._corpora:
            raise create_vector_db_error(
                f"Corpus '{corpus_name}' not found",
                self.__class__.__name__,
                corpus_name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        for doc in documents:
            self._corpora[corpus_name][doc.id] = doc
        
        return True
    
    async def search(self, corpus_name: str, query: str, top_k: int = 10) -> List[SearchResult]:
        """Perform mock search."""
        await self._simulate_operation()
        
        if corpus_name not in self._corpora:
            raise create_vector_db_error(
                f"Corpus '{corpus_name}' not found",
                self.__class__.__name__,
                corpus_name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        # Simple mock search: return documents that contain query terms
        results = []
        query_lower = query.lower()
        
        for doc in self._corpora[corpus_name].values():
            if query_lower in doc.content.lower():
                # Mock similarity score based on query length and content match
                score = min(0.95, len(query) / len(doc.content) + 0.5)
                results.append(SearchResult(
                    document=doc,
                    score=score,
                    corpus_name=corpus_name
                ))
        
        # Sort by score and limit results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    async def list_corpora(self) -> List[str]:
        """List all mock corpora."""
        await self._simulate_operation()
        return list(self._corpora.keys())
    
    async def get_corpus_info(self, name: str) -> CorpusInfo:
        """Get mock corpus information."""
        await self._simulate_operation()
        
        if name not in self._corpora:
            raise create_vector_db_error(
                f"Corpus '{name}' not found",
                self.__class__.__name__,
                name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        metadata = self._corpus_metadata[name]
        return CorpusInfo(
            name=name,
            document_count=len(self._corpora[name]),
            created_at=metadata["created_at"],
            metadata=metadata["metadata"]
        )
    
    async def delete_document(self, corpus_name: str, document_id: str) -> bool:
        """Delete a document from mock corpus."""
        await self._simulate_operation()
        
        if corpus_name not in self._corpora:
            raise create_vector_db_error(
                f"Corpus '{corpus_name}' not found",
                self.__class__.__name__,
                corpus_name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        if document_id not in self._corpora[corpus_name]:
            raise create_vector_db_error(
                f"Document '{document_id}' not found in corpus '{corpus_name}'",
                self.__class__.__name__,
                corpus_name,
                ErrorCode.VECTOR_DB_DOCUMENT_NOT_FOUND
            )
        
        del self._corpora[corpus_name][document_id]
        return True
    
    async def update_document(self, corpus_name: str, document: Document) -> bool:
        """Update a document in mock corpus."""
        await self._simulate_operation()
        
        if corpus_name not in self._corpora:
            raise create_vector_db_error(
                f"Corpus '{corpus_name}' not found",
                self.__class__.__name__,
                corpus_name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        self._corpora[corpus_name][document.id] = document
        return True
    
    async def get_document(self, corpus_name: str, document_id: str) -> Optional[Document]:
        """Get a document from mock corpus."""
        await self._simulate_operation()
        
        if corpus_name not in self._corpora:
            raise create_vector_db_error(
                f"Corpus '{corpus_name}' not found",
                self.__class__.__name__,
                corpus_name,
                ErrorCode.VECTOR_DB_CORPUS_NOT_FOUND
            )
        
        return self._corpora[corpus_name].get(document_id)
    
    async def _simulate_operation(self) -> None:
        """Simulate operation delay and potential failures."""
        # Simulate processing delay
        if self.simulate_delay > 0:
            await asyncio.sleep(self.simulate_delay)
        
        # Simulate random failures if configured
        if self.fail_probability > 0:
            import random
            if random.random() < self.fail_probability:
                raise create_vector_db_error(
                    "Simulated vector database failure",
                    self.__class__.__name__,
                    error_code=ErrorCode.VECTOR_DB_CONNECTION_FAILED
                )


# Utility functions for vector database management

def create_mock_vector_db(
    simulate_delay: float = 0.01,
    fail_probability: float = 0.0,
    **kwargs
) -> MockVectorDB:
    """
    Create a mock vector database for testing.
    
    Args:
        simulate_delay: Delay to simulate processing time
        fail_probability: Probability of simulating failures (0.0-1.0)
        **kwargs: Additional configuration options
        
    Returns:
        Configured mock vector database
    """
    return MockVectorDB(
        simulate_delay=simulate_delay,
        fail_probability=fail_probability,
        **kwargs
    )


def validate_vector_db_provider(provider: BaseVectorDB) -> None:
    """
    Validate that an object implements the BaseVectorDB interface correctly.
    
    Args:
        provider: Vector database provider to validate
        
    Raises:
        LexoraError: If provider doesn't implement the interface correctly
    """
    if not isinstance(provider, BaseVectorDB):
        raise LexoraError(
            f"Provider must inherit from BaseVectorDB, got {type(provider).__name__}",
            ErrorCode.INVALID_CONFIG
        )
    
    # Check that required methods are implemented
    required_methods = [
        'create_corpus', 'delete_corpus', 'add_documents', 'search',
        'list_corpora', 'get_corpus_info', 'delete_document',
        'update_document', 'get_document'
    ]
    
    for method_name in required_methods:
        if not hasattr(provider, method_name):
            raise LexoraError(
                f"Provider missing required method: {method_name}",
                ErrorCode.INVALID_CONFIG
            )
        
        method = getattr(provider, method_name)
        if not callable(method):
            raise LexoraError(
                f"Provider method {method_name} is not callable",
                ErrorCode.INVALID_CONFIG
            )
    
    # Validate configuration
    provider.validate_config()