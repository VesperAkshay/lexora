"""
Text chunking utilities for the Lexora Agentic RAG SDK.

This module provides various text chunking strategies for breaking down large documents
into smaller, manageable pieces for embedding and retrieval.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from enum import Enum

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from ..exceptions import LexoraError, ErrorCode


class ChunkingStrategy(Enum):
    """
    Enumeration of available chunking strategies.
    """
    FIXED_SIZE = "fixed_size"
    SENTENCE_BASED = "sentence_based"
    PARAGRAPH_BASED = "paragraph_based"
    TOKEN_BASED = "token_based"
    SEMANTIC_BASED = "semantic_based"


class TextChunk:
    """
    Represents a chunk of text with metadata.
    
    Attributes:
        content: The text content of the chunk
        start_index: Starting character index in the original document
        end_index: Ending character index in the original document
        chunk_index: Sequential index of this chunk in the document
        metadata: Additional metadata about the chunk
    """
    
    def __init__(
        self,
        content: str,
        start_index: int,
        end_index: int,
        chunk_index: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a text chunk.
        
        Args:
            content: The text content of the chunk
            start_index: Starting character index in the original document
            end_index: Ending character index in the original document
            chunk_index: Sequential index of this chunk
            metadata: Additional metadata about the chunk
        """
        self.content = content
        self.start_index = start_index
        self.end_index = end_index
        self.chunk_index = chunk_index
        self.metadata = metadata or {}
    
    def __str__(self) -> str:
        """Return string representation of the chunk."""
        return f"TextChunk(index={self.chunk_index}, length={len(self.content)}, start={self.start_index})"
    
    def __repr__(self) -> str:
        """Return detailed representation of the chunk."""
        return (f"TextChunk(content='{self.content[:50]}...', "
                f"start_index={self.start_index}, end_index={self.end_index}, "
                f"chunk_index={self.chunk_index})")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the chunk to a dictionary representation.
        
        Returns:
            Dictionary representation of the chunk
        """
        return {
            "content": self.content,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TextChunk':
        """
        Create a TextChunk from a dictionary.
        
        Args:
            data: Dictionary containing chunk data
            
        Returns:
            TextChunk instance
        """
        return cls(
            content=data["content"],
            start_index=data["start_index"],
            end_index=data["end_index"],
            chunk_index=data["chunk_index"],
            metadata=data.get("metadata", {})
        )


class BaseChunker(ABC):
    """
    Abstract base class for text chunkers.
    
    All chunking strategies must implement this interface.
    """
    
    @abstractmethod
    def chunk_text(self, text: str, **kwargs) -> List[TextChunk]:
        """
        Chunk the given text into smaller pieces.
        
        Args:
            text: Text to be chunked
            **kwargs: Additional parameters specific to the chunking strategy
            
        Returns:
            List of text chunks
            
        Raises:
            LexoraError: If chunking fails
        """
        pass


class FixedSizeChunker(BaseChunker):
    """
    Chunks text into fixed-size pieces with optional overlap.
    
    This is the simplest chunking strategy that splits text into chunks
    of a specified character length with configurable overlap.
    """
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        """
        Initialize fixed-size chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Raises:
            LexoraError: If parameters are invalid
        """
        if chunk_size <= 0:
            raise LexoraError(
                "Chunk size must be positive",
                ErrorCode.TOOL_INVALID_PARAMETERS
            )
        
        if overlap < 0:
            raise LexoraError(
                "Overlap cannot be negative",
                ErrorCode.TOOL_INVALID_PARAMETERS
            )
        
        if overlap >= chunk_size:
            raise LexoraError(
                "Overlap must be less than chunk size",
                ErrorCode.TOOL_INVALID_PARAMETERS
            )
        
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, **kwargs) -> List[TextChunk]:
        """
        Chunk text into fixed-size pieces.
        
        Args:
            text: Text to be chunked
            **kwargs: Additional parameters (ignored for this strategy)
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        chunks = []
        chunk_index = 0
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_content = text[start:end]
            
            # Skip empty chunks
            if chunk_content.strip():
                chunk = TextChunk(
                    content=chunk_content,
                    start_index=start,
                    end_index=end,
                    chunk_index=chunk_index,
                    metadata={"strategy": "fixed_size", "chunk_size": self.chunk_size}
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position considering overlap
            start = end - self.overlap
            
            # Avoid infinite loop if overlap is too large
            if start <= chunks[-1].start_index if chunks else False:
                start = end
        
        return chunks


class SentenceBasedChunker(BaseChunker):
    """
    Chunks text based on sentence boundaries with size limits.
    
    This chunker tries to keep sentences intact while respecting
    maximum chunk size limits.
    """
    
    def __init__(self, max_chunk_size: int = 1000, overlap_sentences: int = 1):
        """
        Initialize sentence-based chunker.
        
        Args:
            max_chunk_size: Maximum size of each chunk in characters
            overlap_sentences: Number of sentences to overlap between chunks
            
        Raises:
            LexoraError: If parameters are invalid
        """
        if max_chunk_size <= 0:
            raise LexoraError(
                "Max chunk size must be positive",
                ErrorCode.TOOL_INVALID_PARAMETERS
            )
        
        if overlap_sentences < 0:
            raise LexoraError(
                "Overlap sentences cannot be negative",
                ErrorCode.TOOL_INVALID_PARAMETERS
            )
        
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences
        
        # Sentence boundary pattern
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    
    def chunk_text(self, text: str, **kwargs) -> List[TextChunk]:
        """
        Chunk text based on sentence boundaries.
        
        Args:
            text: Text to be chunked
            **kwargs: Additional parameters (ignored for this strategy)
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Split into sentences
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        
        chunks = []
        chunk_index = 0
        current_sentences = []
        current_length = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed max size, create a chunk
            if current_sentences and current_length + sentence_length > self.max_chunk_size:
                chunk = self._create_chunk_from_sentences(
                    current_sentences, chunk_index, text
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_sentences) - self.overlap_sentences)
                current_sentences = current_sentences[overlap_start:]
                current_length = sum(len(s) for s in current_sentences)
            
            current_sentences.append(sentence)
            current_length += sentence_length
            i += 1
        
        # Add remaining sentences as final chunk
        if current_sentences:
            chunk = self._create_chunk_from_sentences(
                current_sentences, chunk_index, text
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = self.sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_chunk_from_sentences(
        self, sentences: List[str], chunk_index: int, original_text: str
    ) -> TextChunk:
        """Create a TextChunk from a list of sentences."""
        content = ' '.join(sentences)
        
        # Find start and end indices in original text
        start_index = original_text.find(sentences[0])
        if start_index == -1:
            start_index = 0
        
        end_index = start_index + len(content)
        
        return TextChunk(
            content=content,
            start_index=start_index,
            end_index=end_index,
            chunk_index=chunk_index,
            metadata={
                "strategy": "sentence_based",
                "sentence_count": len(sentences),
                "max_chunk_size": self.max_chunk_size
            }
        )


class ParagraphBasedChunker(BaseChunker):
    """
    Chunks text based on paragraph boundaries.
    
    This chunker splits text at paragraph breaks and can combine
    small paragraphs to meet minimum size requirements.
    """
    
    def __init__(self, min_chunk_size: int = 200, max_chunk_size: int = 1000):
        """
        Initialize paragraph-based chunker.
        
        Args:
            min_chunk_size: Minimum size for chunks (small paragraphs will be combined)
            max_chunk_size: Maximum size for chunks (large paragraphs will be split)
            
        Raises:
            LexoraError: If parameters are invalid
        """
        if min_chunk_size <= 0 or max_chunk_size <= 0:
            raise LexoraError(
                "Chunk sizes must be positive",
                ErrorCode.TOOL_INVALID_PARAMETERS
            )
        
        if min_chunk_size >= max_chunk_size:
            raise LexoraError(
                "Min chunk size must be less than max chunk size",
                ErrorCode.TOOL_INVALID_PARAMETERS
            )
        
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def chunk_text(self, text: str, **kwargs) -> List[TextChunk]:
        """
        Chunk text based on paragraph boundaries.
        
        Args:
            text: Text to be chunked
            **kwargs: Additional parameters (ignored for this strategy)
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Split into paragraphs
        paragraphs = self._split_paragraphs(text)
        if not paragraphs:
            return []
        
        chunks = []
        chunk_index = 0
        current_content = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph_length = len(paragraph)
            
            # If paragraph is too large, split it
            if paragraph_length > self.max_chunk_size:
                # First, add any accumulated content as a chunk
                if current_content:
                    chunk = self._create_chunk_from_content(
                        current_content, chunk_index, text
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_content = []
                    current_length = 0
                
                # Split large paragraph using fixed-size chunker
                fixed_chunker = FixedSizeChunker(
                    chunk_size=self.max_chunk_size,
                    overlap=100
                )
                para_chunks = fixed_chunker.chunk_text(paragraph)
                
                for para_chunk in para_chunks:
                    para_chunk.chunk_index = chunk_index
                    para_chunk.metadata.update({
                        "strategy": "paragraph_based",
                        "split_paragraph": True
                    })
                    chunks.append(para_chunk)
                    chunk_index += 1
                
                continue
            
            # If adding this paragraph would exceed max size, create a chunk
            if (current_content and 
                current_length + paragraph_length > self.max_chunk_size):
                chunk = self._create_chunk_from_content(
                    current_content, chunk_index, text
                )
                chunks.append(chunk)
                chunk_index += 1
                current_content = []
                current_length = 0
            
            current_content.append(paragraph)
            current_length += paragraph_length
            
            # If we've reached minimum size, we can create a chunk
            if current_length >= self.min_chunk_size:
                chunk = self._create_chunk_from_content(
                    current_content, chunk_index, text
                )
                chunks.append(chunk)
                chunk_index += 1
                current_content = []
                current_length = 0
        
        # Add remaining content as final chunk
        if current_content:
            chunk = self._create_chunk_from_content(
                current_content, chunk_index, text
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _create_chunk_from_content(
        self, content_list: List[str], chunk_index: int, original_text: str
    ) -> TextChunk:
        """Create a TextChunk from a list of content pieces."""
        content = '\n\n'.join(content_list)
        
        # Find start and end indices in original text
        start_index = original_text.find(content_list[0])
        if start_index == -1:
            start_index = 0
        
        end_index = start_index + len(content)
        
        return TextChunk(
            content=content,
            start_index=start_index,
            end_index=end_index,
            chunk_index=chunk_index,
            metadata={
                "strategy": "paragraph_based",
                "paragraph_count": len(content_list),
                "min_chunk_size": self.min_chunk_size,
                "max_chunk_size": self.max_chunk_size
            }
        )


class TokenBasedChunker(BaseChunker):
    """
    Chunks text based on token count using tiktoken.
    
    This chunker is useful when working with language models that have
    token limits, as it ensures chunks don't exceed token limits.
    """
    
    def __init__(
        self,
        max_tokens: int = 500,
        overlap_tokens: int = 50,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize token-based chunker.
        
        Args:
            max_tokens: Maximum number of tokens per chunk
            overlap_tokens: Number of tokens to overlap between chunks
            encoding_name: Tiktoken encoding name to use
            
        Raises:
            LexoraError: If tiktoken is not available or parameters are invalid
        """
        if not TIKTOKEN_AVAILABLE:
            raise LexoraError(
                "tiktoken library not available. Install with: pip install tiktoken",
                ErrorCode.TOOL_EXECUTION_FAILED
            )
        
        if max_tokens <= 0:
            raise LexoraError(
                "Max tokens must be positive",
                ErrorCode.TOOL_INVALID_PARAMETERS
            )
        
        if overlap_tokens < 0:
            raise LexoraError(
                "Overlap tokens cannot be negative",
                ErrorCode.TOOL_INVALID_PARAMETERS
            )
        
        if overlap_tokens >= max_tokens:
            raise LexoraError(
                "Overlap tokens must be less than max tokens",
                ErrorCode.TOOL_INVALID_PARAMETERS
            )
        
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            raise LexoraError(
                f"Failed to load tiktoken encoding '{encoding_name}': {str(e)}",
                ErrorCode.TOOL_EXECUTION_FAILED,
                original_error=e
            )
    
    def chunk_text(self, text: str, **kwargs) -> List[TextChunk]:
        """
        Chunk text based on token count.
        
        Args:
            text: Text to be chunked
            **kwargs: Additional parameters (ignored for this strategy)
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Encode the entire text
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= self.max_tokens:
            # Text fits in one chunk
            return [TextChunk(
                content=text,
                start_index=0,
                end_index=len(text),
                chunk_index=0,
                metadata={
                    "strategy": "token_based",
                    "token_count": len(tokens),
                    "max_tokens": self.max_tokens
                }
            )]
        
        chunks = []
        chunk_index = 0
        start_token = 0
        
        while start_token < len(tokens):
            end_token = min(start_token + self.max_tokens, len(tokens))
            chunk_tokens = tokens[start_token:end_token]
            
            # Decode tokens back to text
            chunk_content = self.encoding.decode(chunk_tokens)
            
            # Find character indices (approximate)
            start_char = self._estimate_char_index(tokens[:start_token], text)
            end_char = self._estimate_char_index(tokens[:end_token], text)
            
            chunk = TextChunk(
                content=chunk_content,
                start_index=start_char,
                end_index=end_char,
                chunk_index=chunk_index,
                metadata={
                    "strategy": "token_based",
                    "token_count": len(chunk_tokens),
                    "max_tokens": self.max_tokens
                }
            )
            chunks.append(chunk)
            chunk_index += 1
            
            # Move start position considering overlap
            start_token = end_token - self.overlap_tokens
        
        return chunks
    
    def _estimate_char_index(self, tokens: List[int], original_text: str) -> int:
        """Estimate character index from token position."""
        if not tokens:
            return 0
        
        # Decode tokens and find length
        decoded = self.encoding.decode(tokens)
        return min(len(decoded), len(original_text))


class TextChunker:
    """
    High-level text chunker that supports multiple chunking strategies.
    
    This is the main interface for chunking text in the RAG system.
    """
    
    def __init__(self, strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE, **kwargs):
        """
        Initialize text chunker with specified strategy.
        
        Args:
            strategy: Chunking strategy to use
            **kwargs: Strategy-specific parameters
            
        Raises:
            LexoraError: If strategy is not supported
        """
        self.strategy = strategy
        self.chunker = self._create_chunker(strategy, **kwargs)
    
    def _create_chunker(self, strategy: ChunkingStrategy, **kwargs) -> BaseChunker:
        """Create appropriate chunker based on strategy."""
        if strategy == ChunkingStrategy.FIXED_SIZE:
            return FixedSizeChunker(
                chunk_size=kwargs.get('chunk_size', 1000),
                overlap=kwargs.get('overlap', 100)
            )
        
        elif strategy == ChunkingStrategy.SENTENCE_BASED:
            return SentenceBasedChunker(
                max_chunk_size=kwargs.get('max_chunk_size', 1000),
                overlap_sentences=kwargs.get('overlap_sentences', 1)
            )
        
        elif strategy == ChunkingStrategy.PARAGRAPH_BASED:
            return ParagraphBasedChunker(
                min_chunk_size=kwargs.get('min_chunk_size', 200),
                max_chunk_size=kwargs.get('max_chunk_size', 1000)
            )
        
        elif strategy == ChunkingStrategy.TOKEN_BASED:
            return TokenBasedChunker(
                max_tokens=kwargs.get('max_tokens', 500),
                overlap_tokens=kwargs.get('overlap_tokens', 50),
                encoding_name=kwargs.get('encoding_name', 'cl100k_base')
            )
        
        else:
            raise LexoraError(
                f"Unsupported chunking strategy: {strategy}",
                ErrorCode.TOOL_INVALID_PARAMETERS
            )
    
    def chunk_text(self, text: str) -> List[TextChunk]:
        """
        Chunk the given text using the configured strategy.
        
        Args:
            text: Text to be chunked
            
        Returns:
            List of text chunks
            
        Raises:
            LexoraError: If chunking fails
        """
        if not text or not text.strip():
            return []
        
        return self.chunker.chunk_text(text.strip())
    
    def chunk_document(self, document_content: str, document_id: str) -> List[TextChunk]:
        """
        Chunk a document and add document metadata to chunks.
        
        Args:
            document_content: Content of the document to chunk
            document_id: Unique identifier for the document
            
        Returns:
            List of text chunks with document metadata
        """
        chunks = self.chunk_text(document_content)
        
        # Add document metadata to each chunk
        for chunk in chunks:
            chunk.metadata.update({
                "document_id": document_id,
                "total_chunks": len(chunks)
            })
        
        return chunks
    
    def get_strategy(self) -> ChunkingStrategy:
        """Get the current chunking strategy."""
        return self.strategy


# Convenience functions for creating chunkers

def create_fixed_size_chunker(chunk_size: int = 1000, overlap: int = 100) -> TextChunker:
    """
    Create a fixed-size text chunker.
    
    Args:
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        Configured text chunker
    """
    return TextChunker(
        strategy=ChunkingStrategy.FIXED_SIZE,
        chunk_size=chunk_size,
        overlap=overlap
    )


def create_sentence_chunker(max_chunk_size: int = 1000, overlap_sentences: int = 1) -> TextChunker:
    """
    Create a sentence-based text chunker.
    
    Args:
        max_chunk_size: Maximum size of each chunk in characters
        overlap_sentences: Number of sentences to overlap between chunks
        
    Returns:
        Configured text chunker
    """
    return TextChunker(
        strategy=ChunkingStrategy.SENTENCE_BASED,
        max_chunk_size=max_chunk_size,
        overlap_sentences=overlap_sentences
    )


def create_paragraph_chunker(min_chunk_size: int = 200, max_chunk_size: int = 1000) -> TextChunker:
    """
    Create a paragraph-based text chunker.
    
    Args:
        min_chunk_size: Minimum size for chunks
        max_chunk_size: Maximum size for chunks
        
    Returns:
        Configured text chunker
    """
    return TextChunker(
        strategy=ChunkingStrategy.PARAGRAPH_BASED,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size
    )


def create_token_chunker(
    max_tokens: int = 500,
    overlap_tokens: int = 50,
    encoding_name: str = "cl100k_base"
) -> TextChunker:
    """
    Create a token-based text chunker.
    
    Args:
        max_tokens: Maximum number of tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks
        encoding_name: Tiktoken encoding name to use
        
    Returns:
        Configured text chunker
        
    Raises:
        LexoraError: If tiktoken is not available
    """
    return TextChunker(
        strategy=ChunkingStrategy.TOKEN_BASED,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        encoding_name=encoding_name
    )