"""
Delete Document Tool for the Lexora Agentic RAG SDK.

This tool allows users to remove specific documents from corpora using
document IDs, with proper validation and error handling.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from .base_tool import BaseTool, ToolParameter, ParameterType
from ..vector_db.base_vector_db import BaseVectorDB
from ..exceptions import LexoraError, ErrorCode, create_tool_error
from ..utils.logging import get_logger


class DeleteDocumentTool(BaseTool):
    """
    Tool for deleting specific documents from corpora.
    
    This tool provides safe document deletion with validation,
    confirmation options, and detailed reporting of the deletion
    operation results.
    """
    
    def __init__(self, vector_db: BaseVectorDB, **kwargs):
        """
        Initialize the delete document tool.
        
        Args:
            vector_db: Vector database instance for document deletion
            **kwargs: Additional configuration options
            
        Raises:
            LexoraError: If vector_db is not provided or invalid
        """
        super().__init__(**kwargs)
        
        if not isinstance(vector_db, BaseVectorDB):
            raise create_tool_error(
                "vector_db must be an instance of BaseVectorDB",
                "delete_document",
                None,
                ErrorCode.TOOL_INVALID_PARAMETERS
            )
        
        self.vector_db = vector_db
        self.logger = get_logger(self.__class__.__name__)
        
        # Configuration
        self.require_confirmation = kwargs.get('require_confirmation', False)
        self.allow_batch_delete = kwargs.get('allow_batch_delete', True)
    
    @property
    def name(self) -> str:
        """Tool name identifier."""
        return "delete_document"
    
    @property
    def description(self) -> str:
        """Tool description for users and LLMs."""
        return (
            "Delete specific documents from a corpus using document IDs. "
            "Supports both single document deletion and batch operations. "
            "Provides validation to ensure documents exist before deletion "
            "and returns detailed results about the deletion operation."
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
                description="Name of the corpus containing the document(s) to delete",
                required=True
            ),
            ToolParameter(
                name="document_id",
                type=ParameterType.STRING,
                description="ID of the document to delete (for single document deletion)",
                required=False
            ),
            ToolParameter(
                name="document_ids",
                type=ParameterType.ARRAY,
                description="List of document IDs to delete (for batch deletion)",
                required=False
            ),
            ToolParameter(
                name="confirm_deletion",
                type=ParameterType.BOOLEAN,
                description="Explicit confirmation that deletion should proceed",
                required=False,
                default=True
            ),
            ToolParameter(
                name="dry_run",
                type=ParameterType.BOOLEAN,
                description="If true, validate deletion but don't actually delete documents",
                required=False,
                default=False
            ),
            ToolParameter(
                name="ignore_missing",
                type=ParameterType.BOOLEAN,
                description="If true, don't fail if some documents don't exist",
                required=False,
                default=False
            ),
            ToolParameter(
                name="return_deleted_info",
                type=ParameterType.BOOLEAN,
                description="Whether to return information about deleted documents",
                required=False,
                default=True
            )
        ]
    
    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute document deletion.
        
        Args:
            **kwargs: Validated parameters for document deletion
            
        Returns:
            Dictionary containing deletion results and statistics
            
        Raises:
            LexoraError: If document deletion fails
        """
        corpus_name = kwargs["corpus_name"]
        single_document_id = kwargs.get("document_id")
        document_ids_list = kwargs.get("document_ids")
        confirm_deletion = kwargs.get("confirm_deletion", True)
        dry_run = kwargs.get("dry_run", False)
        ignore_missing = kwargs.get("ignore_missing", False)
        return_deleted_info = kwargs.get("return_deleted_info", True)
        
        try:
            # Validate input - must have either single document_id or document_ids list
            if not single_document_id and not document_ids_list:
                raise create_tool_error(
                    "Must provide either 'document_id' for single deletion or 'document_ids' for batch deletion",
                    self.name,
                    {"corpus_name": corpus_name},
                    ErrorCode.TOOL_INVALID_PARAMETERS
                )
            
            if single_document_id and document_ids_list:
                raise create_tool_error(
                    "Cannot provide both 'document_id' and 'document_ids' - choose one deletion method",
                    self.name,
                    {"corpus_name": corpus_name},
                    ErrorCode.TOOL_INVALID_PARAMETERS
                )
            
            # Prepare document IDs list
            if single_document_id:
                document_ids = [single_document_id]
                operation_type = "single"
            else:
                document_ids = document_ids_list
                operation_type = "batch"
            
            # Validate document IDs
            if not document_ids or len(document_ids) == 0:
                raise create_tool_error(
                    "Document IDs list cannot be empty",
                    self.name,
                    {"corpus_name": corpus_name},
                    ErrorCode.TOOL_INVALID_PARAMETERS
                )
            
            # Check for duplicates
            unique_ids = list(set(document_ids))
            if len(unique_ids) != len(document_ids):
                self.logger.warning(f"Duplicate document IDs found, removing duplicates")
                document_ids = unique_ids
            
            # Ensure vector database is connected
            if not self.vector_db.is_connected():
                await self.vector_db.connect()
            
            # Verify corpus exists
            existing_corpora = await self.vector_db.list_corpora()
            if corpus_name not in existing_corpora:
                raise create_tool_error(
                    f"Corpus '{corpus_name}' does not exist. Available corpora: {existing_corpora}",
                    self.name,
                    {"corpus_name": corpus_name, "existing_corpora": existing_corpora},
                    ErrorCode.TOOL_VALIDATION_FAILED
                )
            
            # Check confirmation
            if not confirm_deletion and not dry_run:
                raise create_tool_error(
                    "Deletion not confirmed. Set confirm_deletion=true to proceed with deletion",
                    self.name,
                    {"corpus_name": corpus_name, "document_count": len(document_ids)},
                    ErrorCode.TOOL_VALIDATION_FAILED
                )
            
            self.logger.info(
                f"{'Dry run' if dry_run else 'Executing'} {operation_type} document deletion: "
                f"{len(document_ids)} documents from corpus '{corpus_name}'"
            )
            
            # Validate documents exist and collect information
            existing_documents = []
            missing_documents = []
            deleted_info = []
            
            for doc_id in document_ids:
                try:
                    # Try to get document info before deletion
                    if hasattr(self.vector_db, 'get_document'):
                        doc = await self.vector_db.get_document(corpus_name, doc_id)
                        if doc:
                            existing_documents.append(doc_id)
                            if return_deleted_info:
                                deleted_info.append({
                                    "document_id": doc.id,
                                    "content_length": len(doc.content),
                                    "metadata": doc.metadata,
                                    "had_embedding": doc.embedding is not None
                                })
                        else:
                            missing_documents.append(doc_id)
                    else:
                        # Fallback: assume document exists if we can't check
                        existing_documents.append(doc_id)
                        if return_deleted_info:
                            deleted_info.append({
                                "document_id": doc_id,
                                "content_length": None,
                                "metadata": {},
                                "had_embedding": None,
                                "note": "Document info not available before deletion"
                            })
                except Exception as e:
                    # Check if this is a "not found" error vs other failures
                    error_msg = str(e).lower()
                    if "not found" in error_msg or "does not exist" in error_msg:
                        self.logger.warning(f"Document {doc_id} not found: {e}")
                        missing_documents.append(doc_id)
                    else:
                        # Re-raise non-"not found" errors as they indicate real problems
                        self.logger.error(f"Error verifying document {doc_id}: {e}")
                        raise            
            # Handle missing documents
            if missing_documents and not ignore_missing:
                raise create_tool_error(
                    f"Documents not found in corpus '{corpus_name}': {missing_documents}. "
                    f"Use ignore_missing=true to skip missing documents.",
                    self.name,
                    {"corpus_name": corpus_name, "missing_documents": missing_documents},
                    ErrorCode.TOOL_VALIDATION_FAILED
                )
            
            # Perform deletion (unless dry run)
            deletion_results = []
            successful_deletions = []
            failed_deletions = []
            
            if not dry_run and existing_documents:
                for doc_id in existing_documents:
                    try:
# At the top of lexora/tools/delete_document.py
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone                        if success:
                            successful_deletions.append(doc_id)
                            deletion_results.append({
                                "document_id": doc_id,
                                "status": "deleted",
                                "timestamp": datetime.utcnow().isoformat()
                            })
                        else:
                            failed_deletions.append(doc_id)
                            deletion_results.append({
                                "document_id": doc_id,
                                "status": "failed",
                                "error": "Vector database returned False",
                                "timestamp": datetime.utcnow().isoformat()
                            })
                    except Exception as e:
                        failed_deletions.append(doc_id)
                        deletion_results.append({
                            "document_id": doc_id,
                            "status": "failed",
                            "error": str(e),
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        self.logger.error(f"Failed to delete document {doc_id}: {e}")
            
            # Get updated corpus info
            try:
                updated_corpus_info = await self.vector_db.get_corpus_info(corpus_name)
                final_document_count = updated_corpus_info.document_count
            except Exception as e:
                self.logger.warning(f"Could not retrieve updated corpus info: {e}")
                final_document_count = None
            
            # Prepare response
            result = {
                "corpus_name": corpus_name,
                "operation_type": operation_type,
                "dry_run": dry_run,
                "requested_deletions": len(document_ids),
                "documents_found": len(existing_documents),
                "documents_missing": len(missing_documents),
                "successful_deletions": len(successful_deletions),
                "failed_deletions": len(failed_deletions),
                "final_document_count": final_document_count
            }
            
            # Add detailed information based on parameters
            if missing_documents:
                result["missing_documents"] = missing_documents
            
            if return_deleted_info and deleted_info:
                result["deleted_documents_info"] = deleted_info
            
            if deletion_results:
                result["deletion_results"] = deletion_results
            
            if failed_deletions:
                result["failed_document_ids"] = failed_deletions
            
            # Add execution metadata
            result["execution_info"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "tool_version": self.version,
                "vector_db_provider": self.vector_db.get_provider_name(),
                "parameters_used": {
                    "confirm_deletion": confirm_deletion,
                    "dry_run": dry_run,
                    "ignore_missing": ignore_missing,
                    "return_deleted_info": return_deleted_info
                }
            }
            
            # Generate appropriate message
            if dry_run:
                result["message"] = f"Dry run completed: {len(existing_documents)} documents would be deleted from corpus '{corpus_name}'"
            elif len(successful_deletions) == len(existing_documents):
                result["message"] = f"Successfully deleted {len(successful_deletions)} documents from corpus '{corpus_name}'"
            elif len(successful_deletions) > 0:
                result["message"] = f"Partially successful: deleted {len(successful_deletions)} of {len(existing_documents)} documents from corpus '{corpus_name}'"
            else:
                result["message"] = f"No documents were deleted from corpus '{corpus_name}'"
            
            return result
            
        except LexoraError:
            # Re-raise LexoraErrors as-is
            raise
            
        except Exception as e:
            # Wrap other exceptions
            raise create_tool_error(
                f"Unexpected error deleting documents from corpus '{corpus_name}': {str(e)}",
                self.name,
                {"corpus_name": corpus_name, "document_ids": document_ids if 'document_ids' in locals() else [], "error_type": type(e).__name__},
                ErrorCode.TOOL_EXECUTION_FAILED,
                e
            )


# Convenience function for creating the tool
def create_delete_document_tool(vector_db: BaseVectorDB, **kwargs) -> DeleteDocumentTool:
    """
    Create a DeleteDocumentTool instance.
    
    Args:
        vector_db: Vector database instance
        **kwargs: Additional configuration options
        
    Returns:
        Configured DeleteDocumentTool instance
    """
    return DeleteDocumentTool(vector_db=vector_db, **kwargs)