"""
Embedding utilities for the Lexora Agentic RAG SDK.

This module provides embedding generation and caching functionality with support
for different embedding models and providers.
"""

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod
import asyncio
from functools import lru_cache
import time

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from ..exceptions import LexoraError, ErrorCode, create_llm_error


class EmbeddingCache:
    """
    Simple in-memory cache for embeddings with optional disk persistence.
    
    This cache helps avoid redundant embedding generation for the same text,
    improving performance and reducing API costs.
    """
    
    def __init__(self, max_size: int = 10000, cache_file: Optional[str] = None):
        """
        Initialize the embedding cache.
        
        Args:
            max_size: Maximum number of embeddings to cache in memory
            cache_file: Optional file path for persistent caching
        """
        self.max_size = max_size
        self.cache_file = cache_file
        self._cache: Dict[str, List[float]] = {}
        self._access_times: Dict[str, float] = {}
        
        # Load from disk if cache file exists
        if cache_file and os.path.exists(cache_file):
            self._load_from_disk()
    
    def _generate_key(self, text: str, model: str) -> str:
        """Generate a cache key for the given text and model."""
        content = f"{model}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text: str, model: str) -> Optional[List[float]]:
        """
        Get embedding from cache if available.
        
        Args:
            text: Text to get embedding for
            model: Embedding model name
            
        Returns:
            Cached embedding vector or None if not found
        """
        key = self._generate_key(text, model)
        if key in self._cache:
            self._access_times[key] = time.time()
            return self._cache[key]
        return None
    
    def put(self, text: str, model: str, embedding: List[float]) -> None:
        """
        Store embedding in cache.
        
        Args:
            text: Text that was embedded
            model: Embedding model name
            embedding: Generated embedding vector
        """
        key = self._generate_key(text, model)
        
        # Evict oldest entries if cache is full
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_oldest()
        
        self._cache[key] = embedding
        self._access_times[key] = time.time()
        
        # Save to disk if cache file is configured
        if self.cache_file:
            self._save_to_disk()
    
    def _evict_oldest(self) -> None:
        """Evict the least recently used cache entry."""
        if not self._access_times:
            return
        
        oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        del self._cache[oldest_key]
        del self._access_times[oldest_key]
    
    def _load_from_disk(self) -> None:
        """Load cache from disk file."""
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
                self._cache = data.get('cache', {})
                # Reset access times since we don't persist them
                self._access_times = {k: time.time() for k in self._cache.keys()}
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            # If loading fails, start with empty cache
            self._cache = {}
            self._access_times = {}
    
    def _save_to_disk(self) -> None:
        """Save cache to disk file."""
        try:
            cache_dir = os.path.dirname(self.cache_file)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump({'cache': self._cache}, f)
        except (OSError, json.JSONEncodeError):
            # If saving fails, continue without persistence
            pass    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()
        self._access_times.clear()
        if self.cache_file and os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
            except OSError:
                pass
    
    def size(self) -> int:
        """Get the current cache size."""
        return len(self._cache)


class BaseEmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.
    
    This defines the interface that all embedding providers must implement.
    """
    
    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            LexoraError: If embedding generation fails
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this provider.
        
        Returns:
            Embedding dimension
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name of the embedding model.
        
        Returns:
            Model name
        """
        pass


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    OpenAI embedding provider using the OpenAI API.
    
    Supports various OpenAI embedding models including text-embedding-ada-002
    and newer text-embedding-3-small/large models.
    """
    
    # Model dimensions mapping
    MODEL_DIMENSIONS = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }
    
    def __init__(self, model: str = "text-embedding-ada-002", api_key: Optional[str] = None):
        """
        Initialize OpenAI embedding provider.
        
        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key (if not provided, uses OPENAI_API_KEY env var)
            
        Raises:
            LexoraError: If OpenAI is not available or model is not supported
        """
        if not OPENAI_AVAILABLE:
            raise create_llm_error(
                "OpenAI library not available. Install with: pip install openai",
                "openai",
                model,
                ErrorCode.LLM_CONNECTION_FAILED
            )
        
        if model not in self.MODEL_DIMENSIONS:
            raise create_llm_error(
                f"Unsupported OpenAI embedding model: {model}. "
                f"Supported models: {list(self.MODEL_DIMENSIONS.keys())}",
                "openai",
                model,
                ErrorCode.LLM_MODEL_NOT_FOUND
            )
        
        self.model = model
        self.client = openai.AsyncOpenAI(api_key=api_key)
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding using OpenAI API.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            LexoraError: If API call fails
        """
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        
        except openai.RateLimitError as e:
            raise create_llm_error(
                f"OpenAI rate limit exceeded: {str(e)}",
                "openai",
                self.model,
                ErrorCode.LLM_RATE_LIMIT_EXCEEDED,
                e
            )
        
        except openai.AuthenticationError as e:
            raise create_llm_error(
                f"OpenAI authentication failed: {str(e)}",
                "openai",
                self.model,
                ErrorCode.LLM_AUTHENTICATION_FAILED,
                e
            )
        
        except openai.APITimeoutError as e:
            raise create_llm_error(
                f"OpenAI API timeout: {str(e)}",
                "openai",
                self.model,
                ErrorCode.LLM_TIMEOUT,
                e
            )
        
        except Exception as e:
            raise create_llm_error(
                f"OpenAI embedding generation failed: {str(e)}",
                "openai",
                self.model,
                ErrorCode.LLM_INVALID_RESPONSE,
                e
            )
    
    def get_dimension(self) -> int:
        """Get the dimension of embeddings for this model."""
        return self.MODEL_DIMENSIONS[self.model]
    
    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """
    Mock embedding provider for testing and development.
    
    Generates deterministic embeddings based on text hash for consistent testing.
    """
    
    def __init__(self, dimension: int = 1536, model_name: str = "mock-embedding"):
        """
        Initialize mock embedding provider.
        
        Args:
            dimension: Dimension of generated embeddings
            model_name: Name to use for the mock model
        """
        self.dimension = dimension
        self.model_name = model_name
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate deterministic mock embedding.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Mock embedding vector
        """
        # Generate deterministic embedding based on text hash
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to numbers and normalize
        embedding = []
        for i in range(0, len(text_hash), 2):
            hex_pair = text_hash[i:i+2]
            value = int(hex_pair, 16) / 255.0 * 2.0 - 1.0  # Normalize to [-1, 1]
            embedding.append(value)
        
        # Pad or truncate to desired dimension
        while len(embedding) < self.dimension:
            embedding.extend(embedding[:min(len(embedding), self.dimension - len(embedding))])
        
        return embedding[:self.dimension]
    
    def get_dimension(self) -> int:
        """Get the dimension of mock embeddings."""
        return self.dimension
    
    def get_model_name(self) -> str:
        """Get the mock model name."""
        return self.model_name


class EmbeddingManager:
    """
    High-level embedding manager that handles caching and provider management.
    
    This is the main interface for generating embeddings in the RAG system.
    """
    
    def __init__(
        self,
        provider: BaseEmbeddingProvider,
        cache: Optional[EmbeddingCache] = None,
        enable_caching: bool = True
    ):
        """
        Initialize embedding manager.
        
        Args:
            provider: Embedding provider to use
            cache: Optional custom cache instance
            enable_caching: Whether to enable embedding caching
        """
        self.provider = provider
        self.enable_caching = enable_caching
        
        if enable_caching:
            self.cache = cache or EmbeddingCache()
        else:
            self.cache = None
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text with caching.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector
            
        Raises:
            LexoraError: If embedding generation fails
        """
        if not text or not text.strip():
            raise LexoraError(
                "Cannot generate embedding for empty text",
                ErrorCode.TOOL_INVALID_PARAMETERS
            )
        
        text = text.strip()
        
        # Check cache first
        if self.cache:
            cached_embedding = self.cache.get(text, self.provider.get_model_name())
            if cached_embedding is not None:
                return cached_embedding
        
        # Generate new embedding
        embedding = await self.provider.generate_embedding(text)
        
        # Cache the result
        if self.cache:
            self.cache.put(text, self.provider.get_model_name(), embedding)
        
        return embedding
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors
            
        Raises:
            LexoraError: If any embedding generation fails
        """
        if not texts:
            return []
        
        # Generate embeddings concurrently
        tasks = [self.generate_embedding(text) for text in texts]
        return await asyncio.gather(*tasks)
    
    def get_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.provider.get_dimension()
    
    def get_model_name(self) -> str:
        """Get the embedding model name."""
        return self.provider.get_model_name()
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self.cache:
            self.cache.clear()
    
    def get_cache_size(self) -> int:
        """Get the current cache size."""
        if self.cache:
            return self.cache.size()
        return 0


# Convenience functions for creating embedding managers

def create_openai_embedding_manager(
    model: str = "text-embedding-ada-002",
    api_key: Optional[str] = None,
    enable_caching: bool = True,
    cache_file: Optional[str] = None
) -> EmbeddingManager:
    """
    Create an embedding manager with OpenAI provider.
    
    Args:
        model: OpenAI embedding model name
        api_key: OpenAI API key
        enable_caching: Whether to enable caching
        cache_file: Optional cache file path
        
    Returns:
        Configured embedding manager
        
    Raises:
        LexoraError: If OpenAI setup fails
    """
    provider = OpenAIEmbeddingProvider(model=model, api_key=api_key)
    cache = EmbeddingCache(cache_file=cache_file) if enable_caching else None
    return EmbeddingManager(provider=provider, cache=cache, enable_caching=enable_caching)


def create_mock_embedding_manager(
    dimension: int = 1536,
    model_name: str = "mock-embedding",
    enable_caching: bool = True
) -> EmbeddingManager:
    """
    Create an embedding manager with mock provider for testing.
    
    Args:
        dimension: Embedding dimension
        model_name: Mock model name
        enable_caching: Whether to enable caching
        
    Returns:
        Configured embedding manager with mock provider
    """
    provider = MockEmbeddingProvider(dimension=dimension, model_name=model_name)
    cache = EmbeddingCache() if enable_caching else None
    return EmbeddingManager(provider=provider, cache=cache, enable_caching=enable_caching)


# Utility functions

def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity score between -1 and 1
        
    Raises:
        LexoraError: If embeddings have different dimensions
    """
    if len(embedding1) != len(embedding2):
        raise LexoraError(
            f"Embedding dimensions don't match: {len(embedding1)} vs {len(embedding2)}",
            ErrorCode.VECTOR_DB_INVALID_DIMENSION
        )
    
    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
    
    # Calculate magnitudes
    magnitude1 = sum(a * a for a in embedding1) ** 0.5
    magnitude2 = sum(b * b for b in embedding2) ** 0.5
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


def normalize_embedding(embedding: List[float]) -> List[float]:
    """
    Normalize an embedding vector to unit length.
    
    Args:
        embedding: Embedding vector to normalize
        
    Returns:
        Normalized embedding vector
    """
    magnitude = sum(x * x for x in embedding) ** 0.5
    if magnitude == 0:
        return embedding
    
    return [x / magnitude for x in embedding]