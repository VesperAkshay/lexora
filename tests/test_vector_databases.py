"""
Unit tests for vector database implementations.

This module tests all vector database implementations including:
- BaseVectorDB interface compliance
- FAISS vector database operations
- Pinecone vector database operations
- ChromaDB vector database operations
- Mock vector database operations
"""

import asyncio
import os
import tempfile
import shutil
import time
from typing import List, Dict, Any
from unittest.mock import Mock, patch, AsyncMock
import pytest

from models.core import Document, SearchResult, CorpusInfo
from utils.embeddings import MockEmbeddingProvider, EmbeddingManager
from vector_db.base_vector_db import BaseVectorDB, MockVectorDB
from vector_db.faiss_db import FAISSVectorDB
from vector_db.pinecone_db import PineconeVectorDB
from vector_db.chroma_db import ChromaVectorDB
from exceptions import LexoraError, VectorDBError


class TestBaseVectorDB:
    """Test the BaseVectorDB abstract interface."""
    
    def test_base_vector_db_is_abstract(self):
        """Test that BaseVectorDB cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseVectorDB()
    
    def test_base_vector_db_interface_methods(self):
        """Test that BaseVectorDB defines the required interface methods."""
        required_methods = [
            'connect', 'disconnect', 'create_corpus', 'delete_corpus',
            'list_corpora', 'get_corpus_info', 'add_documents', 
            'delete_document', 'search', 'health_check'
        ]
        
        for method_name in required_methods:
            assert hasattr(BaseVectorDB, method_name)
            method = getattr(BaseVectorDB, method_name)
            assert callable(method)


class TestMockVectorDB:
    """Test the MockVectorDB implementation."""
    
    @pytest.fixture
    def mock_embedding_manager(self):
        """Create a mock embedding manager for testing."""
        provider = MockEmbeddingProvider(dimension=384)
        return EmbeddingManager(provider=provider)
    
    @pytest.fixture
    def mock_vector_db(self, mock_embedding_manager):
        """Create a MockVectorDB instance for testing."""
        config = {'provider': 'mock'}
        return MockVectorDB(config=config, embedding_manager=mock_embedding_manager, dimension=384)
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                id="doc1",
                content="This is the first test document about machine learning.",
                metadata={"source": "test", "category": "ml"}
            ),
            Document(
                id="doc2", 
                content="This is the second test document about artificial intelligence.",
                metadata={"source": "test", "category": "ai"}
            ),
            Document(
                id="doc3",
                content="This is the third test document about natural language processing.",
                metadata={"source": "test", "category": "nlp"}
            )
        ]
    
    @pytest.mark.asyncio
    async def test_mock_vector_db_lifecycle(self, mock_vector_db):
        """Test the complete lifecycle of MockVectorDB operations."""
        # Test connection
        await mock_vector_db.connect()
        assert mock_vector_db.is_connected()
        
        # Test corpus creation
        await mock_vector_db.create_corpus("test_corpus")
        corpora = await mock_vector_db.list_corpora()
        assert "test_corpus" in corpora
        
        # Test corpus info
        corpus_info = await mock_vector_db.get_corpus_info("test_corpus")
        assert corpus_info.name == "test_corpus"
        assert corpus_info.document_count == 0
        
        # Test disconnection
        await mock_vector_db.disconnect()
        assert not mock_vector_db.is_connected()
    
    @pytest.mark.asyncio
    async def test_mock_vector_db_document_operations(self, mock_vector_db, sample_documents):
        """Test document CRUD operations in MockVectorDB."""
        await mock_vector_db.connect()
        await mock_vector_db.create_corpus("test_corpus")
        
        # Test adding documents
        await mock_vector_db.add_documents("test_corpus", sample_documents)
        
        # Verify corpus info updated
        corpus_info = await mock_vector_db.get_corpus_info("test_corpus")
        assert corpus_info.document_count == 3
        
        # Test search functionality
        results = await mock_vector_db.search("test_corpus", "machine learning", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(result, SearchResult) for result in results)
        
        # Test document deletion
        await mock_vector_db.delete_document("test_corpus", "doc1")
        corpus_info = await mock_vector_db.get_corpus_info("test_corpus")
        assert corpus_info.document_count == 2
        
        await mock_vector_db.disconnect()
    
    @pytest.mark.asyncio
    async def test_mock_vector_db_error_handling(self, mock_vector_db):
        """Test error handling in MockVectorDB."""
        await mock_vector_db.connect()
        
        # Test operations on non-existent corpus
        with pytest.raises(VectorDBError):
            await mock_vector_db.get_corpus_info("nonexistent_corpus")
        
        with pytest.raises(VectorDBError):
            await mock_vector_db.add_documents("nonexistent_corpus", [])
        
        with pytest.raises(VectorDBError):
            await mock_vector_db.search("nonexistent_corpus", "query")
        
        # Test duplicate corpus creation
        await mock_vector_db.create_corpus("test_corpus")
        with pytest.raises(VectorDBError):
            await mock_vector_db.create_corpus("test_corpus")
        
        await mock_vector_db.disconnect()
    
    @pytest.mark.asyncio
    async def test_mock_vector_db_health_check(self, mock_embedding_manager):
        """Test health check functionality."""
        # Create a fresh instance for this test
        config = {'provider': 'mock'}
        mock_db = MockVectorDB(config=config, embedding_manager=mock_embedding_manager, dimension=384)
        
        # Health check automatically connects if needed
        health = await mock_db.health_check()
        assert health["connected"]  # Health check auto-connects
        assert health["provider"] == "MockVectorDB"
        assert health["status"] == "healthy"
        assert "response_time" in health
        assert "corpora_count" in health
        
        await mock_db.disconnect()


class TestFAISSVectorDB:
    """Test the FAISSVectorDB implementation."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for FAISS index files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_embedding_manager(self):
        """Create a mock embedding manager for testing."""
        provider = MockEmbeddingProvider(dimension=384)
        return EmbeddingManager(provider=provider)
    
    @pytest.fixture
    def faiss_vector_db(self, mock_embedding_manager, temp_dir):
        """Create a FAISSVectorDB instance for testing."""
        config = {
            'provider': 'faiss',
            'index_path': os.path.join(temp_dir, 'test_index')
        }
        return FAISSVectorDB(
            config=config,
            embedding_manager=mock_embedding_manager,
            dimension=384
        )
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                id="faiss_doc1",
                content="FAISS is a library for efficient similarity search.",
                metadata={"source": "faiss_test", "type": "info"}
            ),
            Document(
                id="faiss_doc2",
                content="Vector databases enable semantic search capabilities.",
                metadata={"source": "faiss_test", "type": "info"}
            )
        ]
    
    @pytest.mark.asyncio
    async def test_faiss_vector_db_lifecycle(self, faiss_vector_db):
        """Test the complete lifecycle of FAISSVectorDB operations."""
        # Test connection
        await faiss_vector_db.connect()
        assert faiss_vector_db.is_connected()
        
        # Test corpus creation with unique name
        corpus_name = f"faiss_lifecycle_corpus_{int(time.time())}"
        await faiss_vector_db.create_corpus(corpus_name)
        corpora = await faiss_vector_db.list_corpora()
        assert corpus_name in corpora
        
        # Test disconnection
        await faiss_vector_db.disconnect()
        assert not faiss_vector_db.is_connected()
    
    @pytest.mark.asyncio
    async def test_faiss_vector_db_document_operations(self, faiss_vector_db, sample_documents):
        """Test document operations in FAISSVectorDB."""
        await faiss_vector_db.connect()
        
        # Use unique corpus name
        corpus_name = f"faiss_docs_corpus_{int(time.time())}"
        await faiss_vector_db.create_corpus(corpus_name)
        
        # Test adding documents
        await faiss_vector_db.add_documents(corpus_name, sample_documents)
        
        # Test search
        results = await faiss_vector_db.search(corpus_name, "similarity search", top_k=1)
        assert len(results) >= 1
        assert isinstance(results[0], SearchResult)
        
        # Test corpus info
        corpus_info = await faiss_vector_db.get_corpus_info(corpus_name)
        assert corpus_info.document_count == 2
        
        await faiss_vector_db.disconnect()
    
    @pytest.mark.asyncio
    async def test_faiss_vector_db_persistence(self, faiss_vector_db, sample_documents, temp_dir):
        """Test FAISS index persistence."""
        # Add documents and disconnect
        await faiss_vector_db.connect()
        
        # Use unique corpus name
        corpus_name = f"persistent_corpus_{int(time.time())}"
        await faiss_vector_db.create_corpus(corpus_name)
        await faiss_vector_db.add_documents(corpus_name, sample_documents)
        await faiss_vector_db.disconnect()
        
        # Create new instance and verify data persisted
        config = {
            'provider': 'faiss',
            'index_path': os.path.join(temp_dir, 'test_index')
        }
        provider = MockEmbeddingProvider(dimension=384)
        embedding_manager = EmbeddingManager(provider=provider)
        
        new_faiss_db = FAISSVectorDB(
            config=config,
            embedding_manager=embedding_manager,
            dimension=384
        )
        
        await new_faiss_db.connect()
        corpus_info = await new_faiss_db.get_corpus_info(corpus_name)
        assert corpus_info.document_count == 2
        
        await new_faiss_db.disconnect()
    
    @pytest.mark.asyncio
    async def test_faiss_vector_db_health_check(self, mock_embedding_manager):
        """Test FAISS health check functionality."""
        # Create a fresh instance for this test
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'provider': 'faiss',
                'index_path': os.path.join(temp_dir, 'health_test_index')
            }
            faiss_db = FAISSVectorDB(
                config=config,
                embedding_manager=mock_embedding_manager,
                dimension=384
            )
            
            # Health check automatically connects if needed
            health = await faiss_db.health_check()
            assert health["connected"]  # Health check auto-connects
            assert health["provider"] == "FAISSVectorDB"
            assert health["status"] == "healthy"
            
            await faiss_db.disconnect()


class TestPineconeVectorDB:
    """Test the PineconeVectorDB implementation."""
    
    @pytest.fixture
    def mock_embedding_manager(self):
        """Create a mock embedding manager for testing."""
        provider = MockEmbeddingProvider(dimension=384)
        return EmbeddingManager(provider=provider)
    
    @pytest.fixture
    def pinecone_vector_db(self, mock_embedding_manager):
        """Create a PineconeVectorDB instance for testing."""
        config = {
            'provider': 'pinecone',
            'api_key': 'test-api-key',
            'environment': 'test-env'
        }
        return PineconeVectorDB(
            config=config,
            embedding_manager=mock_embedding_manager,
            api_key='test-api-key'
        )
    
    @pytest.mark.asyncio
    async def test_pinecone_vector_db_initialization(self, pinecone_vector_db):
        """Test PineconeVectorDB initialization."""
        assert pinecone_vector_db.api_key == 'test-api-key'
        assert not pinecone_vector_db.is_connected()
    
    @pytest.mark.asyncio
    async def test_pinecone_vector_db_health_check(self, pinecone_vector_db):
        """Test Pinecone health check functionality."""
        health = await pinecone_vector_db.health_check()
        assert "connected" in health
        assert health["provider"] == "PineconeVectorDB"
    
    @pytest.mark.asyncio
    @patch('vector_db.pinecone_db.Pinecone')
    async def test_pinecone_vector_db_connection_mock(self, mock_pinecone_class, pinecone_vector_db):
        """Test Pinecone connection with mocked Pinecone client."""
        # Mock the Pinecone client
        mock_client = Mock()
        mock_pinecone_class.return_value = mock_client
        
        # Test connection
        await pinecone_vector_db.connect()
        assert pinecone_vector_db.is_connected()
        
        # Verify Pinecone client was initialized
        mock_pinecone_class.assert_called_once_with(api_key='test-api-key')
        
        await pinecone_vector_db.disconnect()


class TestChromaVectorDB:
    """Test the ChromaVectorDB implementation."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for ChromaDB persistence."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_embedding_manager(self):
        """Create a mock embedding manager for testing."""
        provider = MockEmbeddingProvider(dimension=384)
        return EmbeddingManager(provider=provider)
    
    @pytest.fixture
    def chroma_vector_db(self, mock_embedding_manager, temp_dir):
        """Create a ChromaVectorDB instance for testing."""
        config = {
            'provider': 'chroma',
            'persist_directory': temp_dir
        }
        return ChromaVectorDB(
            config=config,
            embedding_manager=mock_embedding_manager,
            persist_directory=temp_dir
        )
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                id="chroma_doc1",
                content="ChromaDB is an open-source vector database.",
                metadata={"source": "chroma_test", "category": "database"}
            ),
            Document(
                id="chroma_doc2",
                content="Vector embeddings enable semantic similarity search.",
                metadata={"source": "chroma_test", "category": "ml"}
            )
        ]
    
    @pytest.mark.asyncio
    async def test_chroma_vector_db_lifecycle(self, chroma_vector_db):
        """Test the complete lifecycle of ChromaVectorDB operations."""
        # Test connection
        await chroma_vector_db.connect()
        assert chroma_vector_db.is_connected()
        
        # Test corpus creation
        await chroma_vector_db.create_corpus("chroma_test_corpus")
        corpora = await chroma_vector_db.list_corpora()
        assert "chroma_test_corpus" in corpora
        
        # Test disconnection
        await chroma_vector_db.disconnect()
        assert not chroma_vector_db.is_connected()
    
    @pytest.mark.asyncio
    async def test_chroma_vector_db_document_operations(self, chroma_vector_db, sample_documents):
        """Test document operations in ChromaVectorDB."""
        await chroma_vector_db.connect()
        await chroma_vector_db.create_corpus("chroma_test_corpus")
        
        # Test adding documents
        await chroma_vector_db.add_documents("chroma_test_corpus", sample_documents)
        
        # Test search
        results = await chroma_vector_db.search("chroma_test_corpus", "vector database", top_k=2)
        assert len(results) >= 1
        assert isinstance(results[0], SearchResult)
        
        # Test corpus info
        corpus_info = await chroma_vector_db.get_corpus_info("chroma_test_corpus")
        assert corpus_info.document_count == 2
        
        await chroma_vector_db.disconnect()
    
    @pytest.mark.asyncio
    async def test_chroma_vector_db_persistence(self, chroma_vector_db, sample_documents, temp_dir):
        """Test ChromaDB persistence functionality."""
        # Add documents and disconnect
        await chroma_vector_db.connect()
        await chroma_vector_db.create_corpus("persistent_chroma_corpus")
        await chroma_vector_db.add_documents("persistent_chroma_corpus", sample_documents)
        await chroma_vector_db.disconnect()
        
        # Create new instance and verify data persisted
        config = {
            'provider': 'chroma',
            'persist_directory': temp_dir
        }
        provider = MockEmbeddingProvider(dimension=384)
        embedding_manager = EmbeddingManager(provider=provider)
        
        new_chroma_db = ChromaVectorDB(
            config=config,
            embedding_manager=embedding_manager,
            persist_directory=temp_dir
        )
        
        await new_chroma_db.connect()
        corpus_info = await new_chroma_db.get_corpus_info("persistent_chroma_corpus")
        assert corpus_info.document_count == 2
        
        await new_chroma_db.disconnect()
    
    @pytest.mark.asyncio
    async def test_chroma_vector_db_health_check(self, mock_embedding_manager):
        """Test ChromaDB health check functionality."""
        # Create a fresh instance for this test
        temp_dir = tempfile.mkdtemp()
        try:
            config = {
                'provider': 'chroma',
                'persist_directory': temp_dir
            }
            chroma_db = ChromaVectorDB(
                config=config,
                embedding_manager=mock_embedding_manager,
                persist_directory=temp_dir
            )
            
            # Health check automatically connects if needed
            health = await chroma_db.health_check()
            assert health["connected"]  # Health check auto-connects
            assert health["provider"] == "ChromaVectorDB"
            assert health["status"] == "healthy"
            
            await chroma_db.disconnect()
            
            # Give some time for cleanup
            import time
            time.sleep(0.1)
            
        finally:
            # Manual cleanup with retry for Windows
            import time
            for _ in range(3):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    break
                except (OSError, PermissionError):
                    time.sleep(0.1)


class TestVectorDBConvenienceFunctions:
    """Test convenience functions for vector database creation."""
    
    @pytest.fixture
    def mock_embedding_manager(self):
        """Create a mock embedding manager for testing."""
        provider = MockEmbeddingProvider(dimension=384)
        return EmbeddingManager(provider=provider)
    
    def test_create_mock_vector_db(self, mock_embedding_manager):
        """Test create_mock_vector_db convenience function."""
        from vector_db import create_mock_vector_db
        
        mock_db = create_mock_vector_db(
            config={'provider': 'mock'},
            embedding_manager=mock_embedding_manager,
            dimension=384
        )
        
        assert isinstance(mock_db, MockVectorDB)
        assert mock_db.simulate_delay == 0.01
    
    def test_create_faiss_vector_db(self, mock_embedding_manager):
        """Test create_faiss_vector_db convenience function."""
        from vector_db import create_faiss_vector_db
        
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = os.path.join(temp_dir, 'test_index')
            
            faiss_db = create_faiss_vector_db(
                embedding_manager=mock_embedding_manager,
                dimension=384,
                index_path=index_path
            )
            
            assert isinstance(faiss_db, FAISSVectorDB)
            assert faiss_db.dimension == 384
    
    def test_create_chroma_vector_db(self, mock_embedding_manager):
        """Test create_chroma_vector_db convenience function."""
        from vector_db import create_chroma_vector_db
        
        with tempfile.TemporaryDirectory() as temp_dir:
            chroma_db = create_chroma_vector_db(
                embedding_manager=mock_embedding_manager,
                persist_directory=temp_dir
            )
            
            assert isinstance(chroma_db, ChromaVectorDB)
    
    def test_validate_vector_db_provider(self, mock_embedding_manager):
        """Test validate_vector_db_provider function."""
        from vector_db import validate_vector_db_provider
        
        # Test valid provider (MockVectorDB instance)
        config = {'provider': 'mock'}
        mock_db = MockVectorDB(config=config, embedding_manager=mock_embedding_manager, dimension=384)
        
        # Should not raise exception for valid provider
        validate_vector_db_provider(mock_db)
        
        # Test invalid provider (string instead of BaseVectorDB instance)
        with pytest.raises(LexoraError):
            validate_vector_db_provider("invalid_provider")


class TestVectorDBErrorHandling:
    """Test error handling across all vector database implementations."""
    
    @pytest.fixture
    def mock_embedding_manager(self):
        """Create a mock embedding manager for testing."""
        provider = MockEmbeddingProvider(dimension=384)
        return EmbeddingManager(provider=provider)
    
    @pytest.mark.asyncio
    async def test_operations_without_connection(self, mock_embedding_manager):
        """Test that operations work even when not explicitly connected (MockVectorDB behavior)."""
        config = {'provider': 'mock'}
        mock_db = MockVectorDB(config=config, embedding_manager=mock_embedding_manager, dimension=384)
        
        # MockVectorDB allows operations without explicit connection
        # This is by design for testing purposes
        await mock_db.create_corpus("test_corpus")
        corpora = await mock_db.list_corpora()
        assert len(corpora) == 1
    
    @pytest.mark.asyncio
    async def test_invalid_corpus_operations(self, mock_embedding_manager):
        """Test operations on invalid/non-existent corpora."""
        config = {'provider': 'mock'}
        mock_db = MockVectorDB(config=config, embedding_manager=mock_embedding_manager, dimension=384)
        
        await mock_db.connect()
        
        # Test operations on non-existent corpus
        with pytest.raises(VectorDBError):
            await mock_db.get_corpus_info("nonexistent_corpus")
        
        with pytest.raises(VectorDBError):
            await mock_db.delete_corpus("nonexistent_corpus")
        
        await mock_db.disconnect()
    
    @pytest.mark.asyncio
    async def test_empty_document_list_handling(self, mock_embedding_manager):
        """Test handling of empty document lists."""
        config = {'provider': 'mock'}
        mock_db = MockVectorDB(config=config, embedding_manager=mock_embedding_manager, dimension=384)
        
        await mock_db.connect()
        await mock_db.create_corpus("test_corpus")
        
        # Test adding empty document list
        await mock_db.add_documents("test_corpus", [])
        
        # Verify corpus is still empty
        corpus_info = await mock_db.get_corpus_info("test_corpus")
        assert corpus_info.document_count == 0
        
        await mock_db.disconnect()


if __name__ == "__main__":
    pytest.main([__file__])