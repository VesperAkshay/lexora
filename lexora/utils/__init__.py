"""
Utils module - Utility functions and helpers.

This module provides:
- Embedding utilities for text embedding generation
- Text chunking utilities for document processing
- Logging configuration and utilities
- Validation helpers for data validation
"""

from .embeddings import EmbeddingManager, EmbeddingCache
from .chunking import TextChunker, ChunkingStrategy, TextChunk
from .logging import setup_logging, get_logger, configure_logging
from .validation import validate_parameters, validate_schema, ValidationResult

__all__ = [
    "EmbeddingManager",
    "EmbeddingCache",
    "TextChunker",
    "ChunkingStrategy",
    "TextChunk",
    "setup_logging",
    "get_logger",
    "configure_logging",
    "validate_parameters",
    "validate_schema",
    "ValidationResult",
]