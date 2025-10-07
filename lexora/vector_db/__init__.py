"""
Vector Database module - Vector storage abstraction layer.

This module provides:
- BaseVectorDB: Abstract base class for vector database providers
- FAISSVectorDB: FAISS implementation
- PineconeVectorDB: Pinecone implementation  
- ChromaVectorDB: Chroma implementation
"""

from .base_vector_db import BaseVectorDB, MockVectorDB, create_mock_vector_db, validate_vector_db_provider
from .faiss_db import FAISSVectorDB, create_faiss_vector_db
from .pinecone_db import PineconeVectorDB, create_pinecone_vector_db
from .chroma_db import ChromaVectorDB, create_chroma_vector_db, create_persistent_chroma_vector_db, create_chroma_server_client

__all__ = [
    "BaseVectorDB",
    "MockVectorDB",
    "FAISSVectorDB",
    "PineconeVectorDB",
    "ChromaVectorDB",
    "create_mock_vector_db",
    "create_faiss_vector_db",
    "create_pinecone_vector_db",
    "create_chroma_vector_db",
    "create_persistent_chroma_vector_db",
    "create_chroma_server_client",
    "validate_vector_db_provider",
]