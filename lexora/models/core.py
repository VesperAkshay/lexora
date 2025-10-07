"""
Core data models for the Lexora Agentic RAG SDK.

This module contains the fundamental data structures used throughout the system
for representing documents, search results, and corpus information.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator


class Document(BaseModel):
    """
    Represents a document in the system.

    This model encapsulates all information about a document including its content,
    metadata, and optional embedding vector for similarity search.

    Attributes:
        id: Unique identifier for the document
        content: The actual text content of the document
        metadata: Additional key-value pairs for document metadata
        embedding: Optional vector representation for similarity search
    """

    id: str = Field(..., description="Unique identifier for the document")
    content: str = Field(..., description="The actual text content of the document")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value pairs for document metadata",
    )
    embedding: Optional[List[float]] = Field(
        default=None, description="Optional vector representation for similarity search"
    )

    @validator("id")
    def validate_id(cls, v):
        """Validate that document ID is not empty."""
        if not v or not v.strip():
            raise ValueError("Document ID cannot be empty")
        return v.strip()

    @validator("content")
    def validate_content(cls, v):
        """Validate that document content is not empty."""
        if not v or not v.strip():
            raise ValueError("Document content cannot be empty")
        return v.strip()
    @validator("embedding")
    def validate_embedding(cls, v):
        """Validate embedding vector if provided."""
        if v is not None:
            if not isinstance(v, list):
                raise ValueError("Embedding must be a list of floats")
            if len(v) == 0:
                raise ValueError("Embedding cannot be empty")
            # Convert to floats and validate
            try:
                v = [float(x) for x in v]
            except (ValueError, TypeError):
                raise ValueError("All embedding values must be numeric")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the document to a dictionary representation.

        Returns:
            Dictionary representation of the document
        """
        return self.dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """
        Create a Document instance from a dictionary.

        Args:
            data: Dictionary containing document data

        Returns:
            Document instance
        """
        return cls(**data)


class SearchResult(BaseModel):
    """
    Represents a search result from vector database.

    This model encapsulates a document returned from a similarity search
    along with its relevance score and source corpus information.

    Attributes:
        document: The document that matched the search query
        score: Similarity score (higher values indicate better matches)
        corpus_name: Name of the corpus where the document was found
    """

    document: Document = Field(
        ..., description="The document that matched the search query"
    )
    score: float = Field(
        ..., description="Similarity score (higher values indicate better matches)"
    )
    corpus_name: str = Field(
        ..., description="Name of the corpus where the document was found"
    )

    @validator("score")
    def validate_score(cls, v):
        """Validate that score is a valid float."""
        if not isinstance(v, (int, float)):
            raise ValueError("Score must be a numeric value")
        return float(v)

    @validator("corpus_name")
    def validate_corpus_name(cls, v):
        """Validate that corpus name is not empty."""
        if not v or not v.strip():
            raise ValueError("Corpus name cannot be empty")
        return v.strip()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the search result to a dictionary representation.

        Returns:
            Dictionary representation of the search result
        """
        return self.dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResult":
        """
        Create a SearchResult instance from a dictionary.

        Args:
            data: Dictionary containing search result data

        Returns:
            SearchResult instance
        """
        return cls(**data)


class CorpusInfo(BaseModel):
    """
    Information about a document corpus.

    This model provides metadata and statistics about a corpus,
    including document count, creation time, and additional metadata.

    Attributes:
        name: Name of the corpus
        document_count: Number of documents in the corpus
        created_at: Timestamp when the corpus was created
        metadata: Additional key-value pairs for corpus metadata
    """

    name: str = Field(..., description="Name of the corpus")
    document_count: int = Field(..., description="Number of documents in the corpus")
    created_at: datetime = Field(
        ..., description="Timestamp when the corpus was created"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value pairs for corpus metadata",
    )

    @validator("name")
    def validate_name(cls, v):
        """Validate that corpus name is not empty."""
        if not v or not v.strip():
            raise ValueError("Corpus name cannot be empty")
        return v.strip()

    @validator("document_count")
    def validate_document_count(cls, v):
        """Validate that document count is non-negative."""
        if not isinstance(v, int):
            raise ValueError("Document count must be an integer")
        if v < 0:
            raise ValueError("Document count cannot be negative")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the corpus info to a dictionary representation.

        Returns:
            Dictionary representation of the corpus info
        """
        return self.dict()

    @classmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the corpus info to a dictionary representation.

        Returns:
            Dictionary representation of the corpus info
        """
        result = self.dict()
        # Convert datetime to ISO format string for JSON serialization
        result["created_at"] = result["created_at"].isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CorpusInfo":
        """
        Create a CorpusInfo instance from a dictionary.

        Args:
            data: Dictionary containing corpus info data

        Returns:
            CorpusInfo instance
        """
        # Handle datetime conversion if it's a string
        if "created_at" in data and isinstance(data["created_at"], str):
            data = data.copy()
            try:
                data["created_at"] = datetime.fromisoformat(data["created_at"])
            except ValueError as e:
                raise ValueError(f"Invalid datetime format for created_at: {e}")
        return cls(**data)