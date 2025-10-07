"""
Bulk Add Data Tool for the Lexora Agentic RAG SDK.

This tool provides efficient batch processing for adding large datasets to corpora
with optimized performance, progress tracking, and error recovery capabilities.
"""

import asyncio
import uuid
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime
import time

from .base_tool import BaseTool, ToolParameter, ParameterType
from ..models.core import Document
from ..vector_db.base_vector_db import BaseVectorDB
from ..utils.embeddings import EmbeddingManager
from ..utils.chunking import TextChunker, ChunkingStrategy
from ..exceptions import LexoraError, ErrorCode, create_tool_error
from ..utils.logging import get_logger


class BulkAddDataTool(BaseTool):
    """
    Tool for efficient bulk addition of large datasets to corpora.
    
    This tool is optimized for processing large volumes of documents with
    features like parallel processing, progress tracking, error recovery,
    and memory-efficient streaming operations.
    """
    
    def __init__(
        self,
        vector_db: BaseVectorDB,
        embedding_manager: EmbeddingManager,
        text_chunker: Optional[TextChunker] = None,
        **kwargs
    ):
        """
        Initialize the bulk add data tool.
        
        Args:
            vector_db: Vector database instance for document storage
            embedding_manager: Embedding manager for generating embeddings
            text_chunker: Optional text chunker for splitting large documents
            **kwargs: Additional configuration options
            
        Raises:
            LexoraError: If required dependencies are not provided or invalid
        """
        if not isinstance(vector_db, BaseVectorDB):
            raise create_tool_error(
                "vector_db must be an instance of BaseVectorDB",
                "bulk_add_data",
                None,
                ErrorCode.TOOL_INVALID_PARAMETERS
            )
        
        if not isinstance(embedding_manager, EmbeddingManager):
            raise create_tool_error(
                "embedding_manager must be an instance of EmbeddingManager",
                "bulk_add_data",
                None,
                ErrorCode.TOOL_INVALID_PARAMETERS
            )
        
        # Configuration for bulk operations (set before calling super().__init__)
        self.default_batch_size = kwargs.get('default_batch_size', 100)
        self.max_batch_size = kwargs.get('max_batch_size', 1000)
        self.max_concurrent_embeddings = kwargs.get('max_concurrent_embeddings', 10)
        self.progress_report_interval = kwargs.get('progress_report_interval', 100)
        self.auto_generate_ids = kwargs.get('auto_generate_ids', True)
        self.continue_on_error = kwargs.get('continue_on_error', True)
        
        self.vector_db = vector_db
        self.embedding_manager = embedding_manager
        self.text_chunker = text_chunker
        
        super().__init__(**kwargs)
        
        self.logger = get_logger(self.__class__.__name__)
    
    @property
    def name(self) -> str:
        """Tool name identifier."""
        return "bulk_add_data"
    
    @property
    def description(self) -> str:
        """Tool description for users and LLMs."""
        return (
            "Efficiently add large datasets to corpora with optimized batch processing. "
            "Supports parallel embedding generation, progress tracking, error recovery, "
            "and memory-efficient streaming operations. Ideal for processing thousands "
            "of documents with automatic chunking and embedding generation."
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
                description="List of documents to add in bulk. Each document should be a dict with 'content' and optionally 'id' and 'metadata'",
                required=True
            ),
            ToolParameter(
                name="batch_size",
                type=ParameterType.INTEGER,
                description="Number of documents to process in each batch",
                required=False,
                default=self.default_batch_size,
                minimum=1,
                maximum=self.max_batch_size
            ),
            ToolParameter(
                name="max_concurrent_embeddings",
                type=ParameterType.INTEGER,
                description="Maximum number of concurrent embedding generation tasks",
                required=False,
                default=self.max_concurrent_embeddings,
                minimum=1,
                maximum=50
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
                name="generate_embeddings",
                type=ParameterType.BOOLEAN,
                description="Whether to generate embeddings for documents",
                required=False,
                default=True
            ),
            ToolParameter(
                name="continue_on_error",
                type=ParameterType.BOOLEAN,
                description="Whether to continue processing if individual documents fail",
                required=False,
                default=True
            ),
            ToolParameter(
                name="progress_reporting",
                type=ParameterType.BOOLEAN,
                description="Whether to include detailed progress information",
                required=False,
                default=True
            ),
            ToolParameter(
                name="validate_corpus",
                type=ParameterType.BOOLEAN,
                description="Whether to validate that the corpus exists before processing",
                required=False,
                default=True
            ),
            ToolParameter(
                name="deduplicate_ids",
                type=ParameterType.BOOLEAN,
                description="Whether to remove duplicate document IDs from the input",
                required=False,
                default=True
            )
        ]
    
    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute bulk document addition.
        
        Args:
            **kwargs: Validated parameters for bulk document addition
            
        Returns:
            Dictionary containing bulk processing results and statistics
            
        Raises:
            LexoraError: If bulk processing fails
        """
        corpus_name = kwargs["corpus_name"]
        documents_data = kwargs["documents"]
        batch_size = kwargs.get("batch_size", self.default_batch_size)
        max_concurrent = kwargs.get("max_concurrent_embeddings", self.max_concurrent_embeddings)
        chunk_documents = kwargs.get("chunk_documents", True)
        chunk_size = kwargs.get("chunk_size", 1000)
        chunk_overlap = kwargs.get("chunk_overlap", 100)
        generate_embeddings = kwargs.get("generate_embeddings", True)
        continue_on_error = kwargs.get("continue_on_error", True)
        progress_reporting = kwargs.get("progress_reporting", True)
        validate_corpus = kwargs.get("validate_corpus", True)
        deduplicate_ids = kwargs.get("deduplicate_ids", True)
        
        start_time = datetime.utcnow()
        
        try:
            # Validate input
            if not documents_data or len(documents_data) == 0:
                raise create_tool_error(
                    "Documents list cannot be empty",
                    self.name,
                    {"corpus_name": corpus_name},
                    ErrorCode.TOOL_INVALID_PARAMETERS
                )
            
            # Ensure vector database is connected
            if not self.vector_db.is_connected():
                await self.vector_db.connect()
            
            # Validate corpus exists if requested
            if validate_corpus:
                existing_corpora = await self.vector_db.list_corpora()
                if corpus_name not in existing_corpora:
                    raise create_tool_error(
                        f"Corpus '{corpus_name}' does not exist. Available corpora: {existing_corpora}",
                        self.name,
                        {"corpus_name": corpus_name, "existing_corpora": existing_corpora},
                        ErrorCode.TOOL_VALIDATION_FAILED
                    )
            
            self.logger.info(
                f"Starting bulk processing: {len(documents_data)} documents to corpus '{corpus_name}' "
                f"(batch_size={batch_size}, max_concurrent={max_concurrent})"
            )
            
            # Initialize tracking variables
            total_documents = len(documents_data)
            processed_documents = 0
            successful_additions = 0
            failed_documents = 0
            total_chunks_created = 0
            total_embeddings_generated = 0
            processing_errors = []
            
            # Process documents in batches
            async for batch_result in self._process_documents_in_batches(
                corpus_name=corpus_name,
                documents_data=documents_data,
                batch_size=batch_size,
                max_concurrent=max_concurrent,
                chunk_documents=chunk_documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                generate_embeddings=generate_embeddings,
                continue_on_error=continue_on_error,
                deduplicate_ids=deduplicate_ids
            ):
                # Update counters
                processed_documents += batch_result["processed_count"]
                successful_additions += batch_result["successful_count"]
                failed_documents += batch_result["failed_count"]
                total_chunks_created += batch_result["chunks_created"]
                total_embeddings_generated += batch_result["embeddings_generated"]
                processing_errors.extend(batch_result["errors"])
                
                # Report progress if enabled
                if progress_reporting and processed_documents % self.progress_report_interval == 0:
                    progress_pct = (processed_documents / total_documents) * 100
                    self.logger.info(
                        f"Bulk processing progress: {processed_documents}/{total_documents} "
                        f"({progress_pct:.1f}%) - {successful_additions} successful, {failed_documents} failed"
                    )
            
            # Calculate final statistics
            end_time = datetime.utcnow()
            total_duration = (end_time - start_time).total_seconds()
            
            # Get updated corpus info
            try:
                corpus_info = await self.vector_db.get_corpus_info(corpus_name)
                final_document_count = corpus_info.document_count
            except Exception as e:
                self.logger.warning(f"Could not retrieve updated corpus info: {e}")
                final_document_count = None
            
            # Prepare response
            result = {
                "corpus_name": corpus_name,
                "bulk_processing_summary": {
                    "total_input_documents": total_documents,
                    "processed_documents": processed_documents,
                    "successful_additions": successful_additions,
                    "failed_documents": failed_documents,
                    "chunks_created": total_chunks_created,
                    "embeddings_generated": total_embeddings_generated,
                    "processing_errors": len(processing_errors)
                },
                "performance_metrics": {
                    "total_duration_seconds": total_duration,
                    "documents_per_second": processed_documents / total_duration if total_duration > 0 else 0,
                    "successful_rate": (successful_additions / processed_documents) * 100 if processed_documents > 0 else 0,
                    "average_batch_size": batch_size,
                    "concurrent_embeddings": max_concurrent
                },
                "final_corpus_state": {
                    "document_count": final_document_count,
                    "corpus_name": corpus_name
                }
            }
            
            # Add error details if there were failures
            if processing_errors:
                result["processing_errors"] = processing_errors[:50]  # Limit error list size
                if len(processing_errors) > 50:
                    result["additional_errors_count"] = len(processing_errors) - 50
            
            # Add execution metadata
            result["execution_info"] = {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "tool_version": self.version,
                "vector_db_provider": self.vector_db.get_provider_name(),
                "embedding_model": self.embedding_manager.get_model_name() if generate_embeddings else None,
                "parameters_used": {
                    "batch_size": batch_size,
                    "max_concurrent_embeddings": max_concurrent,
                    "chunk_documents": chunk_documents,
                    "generate_embeddings": generate_embeddings,
                    "continue_on_error": continue_on_error
                }
            }
            
            # Generate appropriate message
            if failed_documents == 0:
                result["message"] = f"Successfully processed all {successful_additions} documents in {total_duration:.2f}s"
            elif successful_additions > 0:
                result["message"] = f"Partially successful: {successful_additions} documents added, {failed_documents} failed in {total_duration:.2f}s"
            else:
                result["message"] = f"Bulk processing failed: no documents were successfully added"
            
            # If all documents failed and continue_on_error is False, this is an error
            if failed_documents > 0 and successful_additions == 0 and not continue_on_error:
                raise create_tool_error(
                    f"Bulk processing failed: all {failed_documents} documents failed to process",
                    self.name,
                    {"corpus_name": corpus_name, "failed_count": failed_documents},
                    ErrorCode.TOOL_EXECUTION_FAILED
                )
            
            return result
            
        except LexoraError:
            # Re-raise LexoraErrors as-is
            raise
            
        except Exception as e:
            # Wrap other exceptions
            raise create_tool_error(
                f"Unexpected error during bulk processing for corpus '{corpus_name}': {str(e)}",
                self.name,
                {"corpus_name": corpus_name, "error_type": type(e).__name__},
                ErrorCode.TOOL_EXECUTION_FAILED,
                e
            )
    
    async def _process_documents_in_batches(
        self,
        corpus_name: str,
        documents_data: List[Dict[str, Any]],
        batch_size: int,
        max_concurrent: int,
        chunk_documents: bool,
        chunk_size: int,
        chunk_overlap: int,
        generate_embeddings: bool,
        continue_on_error: bool,
        deduplicate_ids: bool
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process documents in batches with async generator for memory efficiency.
        
        Args:
            corpus_name: Target corpus name
            documents_data: List of document data
            batch_size: Size of each processing batch
            max_concurrent: Maximum concurrent embedding tasks
            chunk_documents: Whether to chunk large documents
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            generate_embeddings: Whether to generate embeddings
            continue_on_error: Whether to continue on individual failures
            deduplicate_ids: Whether to remove duplicate IDs
            
        Yields:
            Dictionary containing batch processing results
        """
        # Deduplicate documents if requested
        if deduplicate_ids:
            seen_ids = set()
            deduplicated_docs = []
            for doc_data in documents_data:
                doc_id = doc_data.get("id")
                if doc_id and doc_id in seen_ids:
                    self.logger.warning(f"Skipping duplicate document ID: {doc_id}")
                    continue
                if doc_id:
                    seen_ids.add(doc_id)
                deduplicated_docs.append(doc_data)
            documents_data = deduplicated_docs
        
        # Process documents in batches
        for i in range(0, len(documents_data), batch_size):
            batch = documents_data[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            try:
                batch_result = await self._process_single_batch(
                    corpus_name=corpus_name,
                    batch_data=batch,
                    batch_num=batch_num,
                    max_concurrent=max_concurrent,
                    chunk_documents=chunk_documents,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    generate_embeddings=generate_embeddings,
                    continue_on_error=continue_on_error
                )
                
                yield batch_result
                
            except Exception as e:
                self.logger.error(f"Batch {batch_num} processing failed: {e}")
                
                # Yield error result for this batch
                yield {
                    "batch_number": batch_num,
                    "processed_count": len(batch),
                    "successful_count": 0,
                    "failed_count": len(batch),
                    "chunks_created": 0,
                    "embeddings_generated": 0,
                    "errors": [{"batch_error": str(e), "affected_documents": len(batch)}]
                }
                
                if not continue_on_error:
                    raise
    
    async def _process_single_batch(
        self,
        corpus_name: str,
        batch_data: List[Dict[str, Any]],
        batch_num: int,
        max_concurrent: int,
        chunk_documents: bool,
        chunk_size: int,
        chunk_overlap: int,
        generate_embeddings: bool,
        continue_on_error: bool
    ) -> Dict[str, Any]:
        """
        Process a single batch of documents.
        
        Args:
            corpus_name: Target corpus name
            batch_data: List of documents in this batch
            batch_num: Batch number for logging
            max_concurrent: Maximum concurrent embedding tasks
            chunk_documents: Whether to chunk documents
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            generate_embeddings: Whether to generate embeddings
            continue_on_error: Whether to continue on failures
            
        Returns:
            Dictionary containing batch processing results
        """
        batch_start_time = time.time()
        processed_documents = []
        batch_errors = []
        chunks_created = 0
        embeddings_generated = 0
        
        # Process each document in the batch
        for i, doc_data in enumerate(batch_data):
            try:
                # Validate document data
                if not isinstance(doc_data, dict):
                    raise ValueError(f"Document {i} must be a dictionary")
                
                if "content" not in doc_data:
                    raise ValueError(f"Document {i} missing required 'content' field")
                
                content = doc_data["content"]
                if not content or not content.strip():
                    self.logger.warning(f"Skipping document {i} with empty content")
                    continue
                
                doc_id = doc_data.get("id")
                if not doc_id and self.auto_generate_ids:
                    doc_id = f"bulk_doc_{uuid.uuid4().hex[:8]}_{int(time.time())}"
                
                doc_metadata = doc_data.get("metadata", {})
                
                # Add processing metadata
                processing_metadata = {
                    **doc_metadata,
                    "added_at": datetime.utcnow().isoformat(),
                    "added_by": "lexora_bulk_add_data_tool",
                    "batch_number": batch_num,
                    "original_length": len(content),
                    "embedding_model": self.embedding_manager.get_model_name() if generate_embeddings else None
                }
                
                # Handle text chunking
                if chunk_documents and self.text_chunker and len(content) > chunk_size:
                    # Create a chunker with the specified parameters
                    from ..utils.chunking import TextChunker, ChunkingStrategy
                    dynamic_chunker = TextChunker(
                        strategy=ChunkingStrategy.FIXED_SIZE,
                        chunk_size=chunk_size,
                        overlap=chunk_overlap
                    )
                    
                    # Chunk the document
                    chunks = dynamic_chunker.chunk_text(content)
                    
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
                            content=chunk.content.strip(),
                            metadata=chunk_metadata
                        ))
                    
                    chunks_created += len(chunks)
                
                else:
                    # Use document as-is
                    if not doc_id:
                        raise ValueError(f"Document {i} missing 'id' field and auto_generate_ids is disabled")
                    
                    processing_metadata["is_chunk"] = False
                    processed_documents.append(Document(
                        id=doc_id,
                        content=content.strip(),
                        metadata=processing_metadata
                    ))
                
            except Exception as e:
                error_info = {
                    "document_index": i,
                    "error": str(e),
                    "document_id": doc_data.get("id", "unknown")
                }
                batch_errors.append(error_info)
                self.logger.warning(f"Failed to process document {i} in batch {batch_num}: {e}")
                
                if not continue_on_error:
                    raise
        
        # Generate embeddings concurrently if requested
        if generate_embeddings and processed_documents:
            try:
                embeddings_generated = await self._generate_embeddings_concurrent(
                    processed_documents, max_concurrent
                )
            except Exception as e:
                self.logger.error(f"Embedding generation failed for batch {batch_num}: {e}")
                if not continue_on_error:
                    raise
        
        # Add documents to vector database
        successful_count = 0
        if processed_documents:
            try:
                # Use batch addition if available
                if hasattr(self.vector_db, 'add_documents_batch'):
                    success = await self.vector_db.add_documents_batch(
                        corpus_name, processed_documents, batch_size=len(processed_documents)
                    )
                else:
                    success = await self.vector_db.add_documents(corpus_name, processed_documents)
                
                if success:
                    successful_count = len(processed_documents)
                else:
                    batch_errors.append({
                        "batch_error": "Vector database returned False for batch addition",
                        "affected_documents": len(processed_documents)
                    })
                    
            except Exception as e:
                batch_errors.append({
                    "batch_error": f"Failed to add documents to vector database: {str(e)}",
                    "affected_documents": len(processed_documents)
                })
                self.logger.error(f"Failed to add batch {batch_num} to vector database: {e}")
                
                if not continue_on_error:
                    raise
        
        batch_duration = time.time() - batch_start_time
        
        self.logger.info(
            f"Batch {batch_num} completed: {successful_count}/{len(batch_data)} successful "
            f"in {batch_duration:.2f}s"
        )
        
        return {
            "batch_number": batch_num,
            "processed_count": len(batch_data),
            "successful_count": successful_count,
            "failed_count": len(batch_data) - successful_count,
            "chunks_created": chunks_created,
            "embeddings_generated": embeddings_generated,
            "errors": batch_errors,
            "batch_duration": batch_duration
        }
    
    async def _generate_embeddings_concurrent(
        self, documents: List[Document], max_concurrent: int
    ) -> int:
        """
        Generate embeddings for documents with controlled concurrency.
        
        Args:
            documents: List of documents to generate embeddings for
            max_concurrent: Maximum number of concurrent embedding tasks
            
        Returns:
            Number of embeddings successfully generated
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        embeddings_generated = 0
        
        async def generate_single_embedding(doc: Document) -> bool:
            async with semaphore:
                try:
                    embedding = await self.embedding_manager.generate_embedding(doc.content)
                    doc.embedding = embedding
                    return True
                except Exception as e:
                    self.logger.warning(f"Failed to generate embedding for document {doc.id}: {e}")
                    return False
        
        # Generate embeddings concurrently
        tasks = [generate_single_embedding(doc) for doc in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful embeddings
        for result in results:
            if result is True:
                embeddings_generated += 1
        
        return embeddings_generated


# Convenience function for creating the tool
def create_bulk_add_data_tool(
    vector_db: BaseVectorDB,
    embedding_manager: EmbeddingManager,
    text_chunker: Optional[TextChunker] = None,
    **kwargs
) -> BulkAddDataTool:
    """
    Create a BulkAddDataTool instance.
    
    Args:
        vector_db: Vector database instance
        embedding_manager: Embedding manager instance
        text_chunker: Optional text chunker instance
        **kwargs: Additional configuration options
        
    Returns:
        Configured BulkAddDataTool instance
    """
    return BulkAddDataTool(
        vector_db=vector_db,
        embedding_manager=embedding_manager,
        text_chunker=text_chunker,
        **kwargs
    )