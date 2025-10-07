"""
Add Data Tool for the Lexora Agentic RAG SDK.

This tool allows users to add documents to existing corpora with automatic
embedding generation, text chunking, and batch processing support.
"""

import uuid
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone

from .base_tool import BaseTool, ToolParameter, ParameterType
from ..models.core import Document
from ..vector_db.base_vector_db import BaseVectorDB
from ..utils.embeddings import EmbeddingManager
from ..utils.chunking import TextChunker
from ..exceptions import LexoraError, ErrorCode, create_tool_error
from ..utils.logging import get_logger


class AddDataTool(BaseTool):
    """
    Tool for adding documents to corpora with automatic processing.
    
    This tool handles document processing including text chunking, embedding
    generation, and batch insertion into vector databases. It supports both
    single documents and batch operations.
    """
    
    def __init__(
        self,
        vector_db: BaseVectorDB,
        embedding_manager: EmbeddingManager,
        text_chunker: Optional[TextChunker] = None,
        **kwargs
    ):
        """
        Initialize the add data tool.
        
        Args:
            vector_db: Vector database instance for document storage
            embedding_manager: Embedding manager for generating embeddings
            text_chunker: Optional text chunker for splitting large documents
            **kwargs: Additional configuration options
            
        Raises:
            LexoraError: If required dependencies are not provided or invalid
        """
        super().__init__(**kwargs)
        
        if not isinstance(vector_db, BaseVectorDB):
            raise create_tool_error(
                "vector_db must be an instance of BaseVectorDB",
                "add_data",
                ErrorCode.TOOL_INVALID_PARAMETERS
            )
        
        if not isinstance(embedding_manager, EmbeddingManager):
            raise create_tool_error(
                "embedding_manager must be an instance of EmbeddingManager",
                "add_data",
                ErrorCode.TOOL_INVALID_PARAMETERS
            )
        
        self.vector_db = vector_db
        self.embedding_manager = embedding_manager
        self.text_chunker = text_chunker
        self.logger = get_logger(self.__class__.__name__)
        
        # Configuration
        self.max_batch_size = kwargs.get('max_batch_size', 100)
        self.auto_generate_ids = kwargs.get('auto_generate_ids', True)
    
    @property
    def name(self) -> str:
        """Tool name identifier."""
        return "add_data"
    
    @property
    def description(self) -> str:
        """Tool description for users and LLMs."""
        return (
            "Add documents to an existing corpus with automatic embedding generation. "
            "Supports single documents, multiple documents, and automatic text chunking "
            "for large documents. Documents will be processed and stored in the vector database."
        )
    
    @property
    def version(self) -> str:
        """Tool version for compatibility tracking."""
        return "1.0.0"
    
    def _setup_parameters(self) -> None:
        """Set up tool parameters."""
        self._parameters = [
            ToolParameter(
                name="corpus_name",
                type=ParameterType.STRING,
                description="Name of the corpus to add documents to",
                required=True
            ),
            ToolParameter(
                name="documents",
                type=ParameterType.ARRAY,
                description="List of documents to add. Each document should be a dict with 'content' and optionally 'id' and 'metadata'",
                required=False
            ),
            ToolParameter(
                name="content",
                type=ParameterType.STRING,
                description="Single document content (alternative to documents array)",
                required=False
            ),
            ToolParameter(
                name="document_id",
                type=ParameterType.STRING,
                description="ID for single document (used with content parameter)",
                required=False
            ),
            ToolParameter(
                name="metadata",
                type=ParameterType.OBJECT,
                description="Metadata for single document (used with content parameter)",
                required=False,
                default={}
            ),
            ToolParameter(
                name="chunk_documents",
                type=ParameterType.BOOLEAN,
                description="Whether to automatically chunk large documents",
                required=False,
                default=True
            ),
            ToolParameter(
                name="chunk_size",
                type=ParameterType.INTEGER,
                description="Maximum size of text chunks in characters",
                required=False,
                default=1000,
                minimum=100,
                maximum=10000
            ),
            ToolParameter(
                name="chunk_overlap",
                type=ParameterType.INTEGER,
                description="Number of characters to overlap between chunks",
                required=False,
                default=100,
                minimum=0,
                maximum=1000
            ),
            ToolParameter(
                name="batch_size",
                type=ParameterType.INTEGER,
                description="Number of documents to process in each batch",
                required=False,
                default=50,
                minimum=1,
                maximum=500
            ),
            ToolParameter(
                name="generate_embeddings",
                type=ParameterType.BOOLEAN,
                description="Whether to generate embeddings for documents",
                required=False,
                default=True
            )
        ]
    
    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute document addition.
        
        Args:
            **kwargs: Validated parameters for document addition
            
        Returns:
            Dictionary containing addition results and statistics
            
        Raises:
            LexoraError: If document addition fails
        """
        corpus_name = kwargs["corpus_name"]
        documents_data = kwargs.get("documents")
        single_content = kwargs.get("content")
        single_id = kwargs.get("document_id")
        single_metadata = kwargs.get("metadata", {})
        chunk_documents = kwargs.get("chunk_documents", True)
        chunk_size = kwargs.get("chunk_size", 1000)
        chunk_overlap = kwargs.get("chunk_overlap", 100)
        batch_size = kwargs.get("batch_size", 50)
        generate_embeddings = kwargs.get("generate_embeddings", True)
        
        try:
            # Validate input - must have either documents array or single content
            if not documents_data and not single_content:
                raise create_tool_error(
                    "Must provide either 'documents' array or 'content' for single document",
                    self.name,
                    ErrorCode.TOOL_INVALID_PARAMETERS
                )
            
            if documents_data and single_content:
                raise create_tool_error(
                    "Cannot provide both 'documents' array and single 'content' - choose one",
                    self.name,
                    ErrorCode.TOOL_INVALID_PARAMETERS
                )
            
            # Ensure vector database is connected
            if not self.vector_db.is_connected():
                await self.vector_db.connect()
            
            # Verify corpus exists
            existing_corpora = await self.vector_db.list_corpora()
            if corpus_name not in existing_corpora:
                raise create_tool_error(
                    f"Corpus '{corpus_name}' does not exist. Create it first using the create_corpus tool.",
                    self.name,
                    {"corpus_name": corpus_name, "existing_corpora": existing_corpora},
                    ErrorCode.TOOL_VALIDATION_FAILED
                )
            
            # Prepare documents list
            raw_documents = []
            
            if single_content:
                # Handle single document
                doc_data = {
                    "content": single_content,
                    "metadata": single_metadata
                }
                if single_id:
                    doc_data["id"] = single_id
                raw_documents.append(doc_data)
            else:
                # Handle documents array
                raw_documents = documents_data
            
            # Process and validate documents
            processed_documents = []
            total_chunks = 0
            
            for i, doc_data in enumerate(raw_documents):
                if not isinstance(doc_data, dict):
                    raise create_tool_error(
                        f"Document {i} must be a dictionary with 'content' field",
                        self.name,
                        ErrorCode.TOOL_INVALID_PARAMETERS
                    )
                
                if "content" not in doc_data:
                    raise create_tool_error(
                        f"Document {i} missing required 'content' field",
                        self.name,
                        ErrorCode.TOOL_INVALID_PARAMETERS
                    )
                
                content = doc_data["content"]
                if not content or not content.strip():
                    self.logger.warning(f"Skipping document {i} with empty content")
                    continue
                
                doc_id = doc_data.get("id")
                if not doc_id and self.auto_generate_ids:
                    doc_id = f"doc_{uuid.uuid4().hex[:8]}_{int(datetime.now(timezone.utc).timestamp())}"                
                doc_metadata = doc_data.get("metadata", {})
                
                # Add processing metadata
                processing_metadata = {
                    **doc_metadata,
                    "added_at": datetime.utcnow().isoformat(),
                    "added_by": "lexora_add_data_tool",
                    "original_length": len(content),
                    "embedding_model": self.embedding_manager.get_model_name() if generate_embeddings else None
                }
                
                # Handle text chunking
                if chunk_documents and len(content) > chunk_size:
                    # Use injected chunker if available, otherwise create new one
                    if self.text_chunker is not None:
                        # Use the provided chunker
                        # Note: chunk_size and chunk_overlap parameters are ignored when using custom chunker
                        chunker = self.text_chunker
                    else:
                        # Create a new chunker with the specified parameters
                        from ..utils.chunking import TextChunker, ChunkingStrategy
                        chunker = TextChunker(
                            strategy=ChunkingStrategy.FIXED_SIZE,
                            chunk_size=chunk_size,
                            overlap=chunk_overlap
                        )
                    
                    # Chunk the document
                    chunks = chunker.chunk_text(content)
                    
                    # Create document for each chunk
                    for j, chunk in enumerate(chunks):
                        chunk_id = f"{doc_id}_chunk_{j}" if doc_id else f"chunk_{uuid.uuid4().hex[:8]}"
                        chunk_metadata = {
                            **processing_metadata,
                            "chunk_index": j,
                            "total_chunks": len(chunks),
                            "parent_document_id": doc_id,
                            "is_chunk": True
                        }
                        
                        processed_documents.append(Document(
                            id=chunk_id,
                            content=chunk.content.strip(),  # TextChunk has a content attribute
                            metadata=chunk_metadata
                        ))
                    
                    total_chunks += len(chunks)
                    
                else:
                    # Use document as-is
                    if not doc_id:
                        raise create_tool_error(
                            f"Document {i} missing 'id' field and auto_generate_ids is disabled",
                            self.name,
                            ErrorCode.TOOL_INVALID_PARAMETERS
                        )
                    
                    processing_metadata["is_chunk"] = False
                    processed_documents.append(Document(
                        id=doc_id,
                        content=content.strip(),
                        metadata=processing_metadata
                    ))
            
            if not processed_documents:
                return {
                    "corpus_name": corpus_name,
                    "documents_added": 0,
                    "chunks_created": 0,
                    "embeddings_generated": 0,
                    "batches_processed": 0,
                    "message": "No valid documents to add"
                }
            
            # Generate embeddings if requested
            embeddings_generated = 0
            if generate_embeddings:
                self.logger.info(f"Generating embeddings for {len(processed_documents)} documents")
                
                for doc in processed_documents:
                    try:
                        embedding = await self.embedding_manager.generate_embedding(doc.content)
                        doc.embedding = embedding
                        embeddings_generated += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to generate embedding for document {doc.id}: {e}")
                        # Continue without embedding for this document
            
            # Add documents in batches
            batches_processed = 0
            documents_added = 0
            
            for i in range(0, len(processed_documents), batch_size):
                batch = processed_documents[i:i + batch_size]
                batch_num = batches_processed + 1
                
                try:
                    self.logger.info(f"Adding batch {batch_num} ({len(batch)} documents) to corpus '{corpus_name}'")
                    success = await self.vector_db.add_documents(corpus_name, batch)
                    
                    if success:
                        documents_added += len(batch)
                        batches_processed += 1
                    else:
                        raise create_tool_error(
                            f"Failed to add batch {batch_num} to corpus '{corpus_name}'",
                            self.name,
                            {"batch_size": len(batch), "batch_number": batch_num},
                            ErrorCode.TOOL_EXECUTION_FAILED
                        )
                        
                except Exception as e:
                    self.logger.error(f"Error adding batch {batch_num}: {e}")
                    raise create_tool_error(
                        f"Failed to add batch {batch_num} to corpus '{corpus_name}': {str(e)}",
                        self.name,
                        {"batch_size": len(batch), "batch_number": batch_num},
                        ErrorCode.TOOL_EXECUTION_FAILED,
                        e
                    )
            
            # Get updated corpus info
            try:
                corpus_info = await self.vector_db.get_corpus_info(corpus_name)
                final_document_count = corpus_info.document_count
            except Exception as e:
                self.logger.warning(f"Could not retrieve updated corpus info: {e}")
                final_document_count = None
            
            # Return success response
            return {
                "corpus_name": corpus_name,
                "documents_added": documents_added,
                "chunks_created": total_chunks,
                "embeddings_generated": embeddings_generated,
                "batches_processed": batches_processed,
                "final_document_count": final_document_count,
                "processing_summary": {
                    "chunking_enabled": chunk_documents,
                    "chunk_size": chunk_size if chunk_documents else None,
                    "chunk_overlap": chunk_overlap if chunk_documents else None,
                    "embedding_model": self.embedding_manager.get_model_name() if generate_embeddings else None,
                    "batch_size": batch_size
                },
                "message": f"Successfully added {documents_added} documents to corpus '{corpus_name}'"
            }
            
        except LexoraError:
            # Re-raise LexoraErrors as-is
            raise
            
        except Exception as e:
            # Wrap other exceptions
            raise create_tool_error(
                f"Unexpected error adding data to corpus '{corpus_name}': {str(e)}",
                self.name,
                {"corpus_name": corpus_name, "error_type": type(e).__name__},
                ErrorCode.TOOL_EXECUTION_FAILED,
                e
            )


# Convenience function for creating the tool
def create_add_data_tool(
    vector_db: BaseVectorDB,
    embedding_manager: EmbeddingManager,
    text_chunker: Optional[TextChunker] = None,
    **kwargs
) -> AddDataTool:
    """
    Create an AddDataTool instance.
    
    Args:
        vector_db: Vector database instance
        embedding_manager: Embedding manager instance
        text_chunker: Optional text chunker instance
        **kwargs: Additional configuration options
        
    Returns:
        Configured AddDataTool instance
    """
    return AddDataTool(
        vector_db=vector_db,
        embedding_manager=embedding_manager,
        text_chunker=text_chunker,
        **kwargs
    )