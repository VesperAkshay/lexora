"""
Delete Corpus Tool for the Lexora Agentic RAG SDK.

This tool allows users to completely remove document corpora from the vector
database with proper validation, confirmation, and safety measures.
"""

from typing import Any, Dict, List, Optional
 from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from .base_tool import BaseTool, ToolParameter, ParameterType
from ..models.core import CorpusInfo
from ..vector_db.base_vector_db import BaseVectorDB
from ..exceptions import LexoraError, ErrorCode, create_tool_error
from ..utils.logging import get_logger


class DeleteCorpusTool(BaseTool):
    """
    Tool for deleting entire document corpora.
    
    This tool provides safe corpus deletion with multiple confirmation
    mechanisms, backup options, and detailed reporting. It includes
    safety measures to prevent accidental deletion of important data.
    """
    
    def __init__(self, vector_db: BaseVectorDB, **kwargs):
        """
        Initialize the delete corpus tool.
        
        Args:
            vector_db: Vector database instance for corpus deletion
            **kwargs: Additional configuration options
            
        Raises:
            LexoraError: If vector_db is not provided or invalid
        """
        super().__init__(**kwargs)
        
        if not isinstance(vector_db, BaseVectorDB):
            raise create_tool_error(
                "vector_db must be an instance of BaseVectorDB",
                "delete_corpus",
                None,
                ErrorCode.TOOL_INVALID_PARAMETERS
            )
        
        self.vector_db = vector_db
        self.logger = get_logger(self.__class__.__name__)
        
        # Configuration
        self.require_explicit_confirmation = kwargs.get('require_explicit_confirmation', True)
    
    @property
    def name(self) -> str:
        """Tool name identifier."""
        return "delete_corpus"
    
    @property
    def description(self) -> str:
        """Tool description for users and LLMs."""
        return (
            "Delete an entire document corpus and all its contents from the vector database. "
            "This is a destructive operation that permanently removes all documents in the corpus. "
            "Includes safety measures and confirmation requirements to prevent accidental deletion. "
            "Use with caution as this operation cannot be undone."
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
                description="Name of the corpus to delete",
                required=True
            ),
            ToolParameter(
                name="confirm_deletion",
                type=ParameterType.BOOLEAN,
                description="Explicit confirmation that the corpus should be deleted",
                required=True
            ),
            ToolParameter(
                name="confirmation_phrase",
                type=ParameterType.STRING,
                description="Type the corpus name again to confirm deletion (safety measure)",
                required=False
            ),
            ToolParameter(
                name="force_delete_non_empty",
                type=ParameterType.BOOLEAN,
                description="Allow deletion of non-empty corpora (use with extreme caution)",
                required=False,
                default=False
            ),
            ToolParameter(
                name="dry_run",
                type=ParameterType.BOOLEAN,
                description="If true, validate deletion but don't actually delete the corpus",
                required=False,
                default=False
            ),
            ToolParameter(
                name="return_corpus_info",
                type=ParameterType.BOOLEAN,
                description="Whether to return information about the deleted corpus",
                required=False,
                default=True
            ),
            ToolParameter(
                name="backup_before_delete",
                type=ParameterType.BOOLEAN,
                description="Whether to create a backup before deletion (if supported)",
                required=False,
                default=False
            )
        ]
    
    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute corpus deletion.
        
        Args:
            **kwargs: Validated parameters for corpus deletion
            
        Returns:
            Dictionary containing deletion results and corpus information
            
        Raises:
            LexoraError: If corpus deletion fails
        """
        corpus_name = kwargs["corpus_name"]
        confirm_deletion = kwargs["confirm_deletion"]
        confirmation_phrase = kwargs.get("confirmation_phrase")
        force_delete_non_empty = kwargs.get("force_delete_non_empty", False)
        dry_run = kwargs.get("dry_run", False)
        return_corpus_info = kwargs.get("return_corpus_info", True)
        backup_before_delete = kwargs.get("backup_before_delete", False)
        
        try:
            # Validate corpus name
            if not corpus_name or not corpus_name.strip():
                raise create_tool_error(
                    "Corpus name cannot be empty",
                    self.name,
                    {"corpus_name": corpus_name},
                    ErrorCode.TOOL_INVALID_PARAMETERS
                )
            
            corpus_name = corpus_name.strip()
            
            # Ensure vector database is connected
            if not self.vector_db.is_connected():
                await self.vector_db.connect()
            
            # Verify corpus exists and get information
            existing_corpora = await self.vector_db.list_corpora()
            if corpus_name not in existing_corpora:
                raise create_tool_error(
                    f"Corpus '{corpus_name}' does not exist. Available corpora: {existing_corpora}",
                    self.name,
                    {"corpus_name": corpus_name, "existing_corpora": existing_corpora},
                    ErrorCode.TOOL_VALIDATION_FAILED
                )
            
            # Get corpus information before deletion
            try:
                corpus_info = await self.vector_db.get_corpus_info(corpus_name)
            except Exception as e:
                self.logger.warning(f"Could not retrieve corpus info before deletion: {e}")
                corpus_info = None
            
            # Safety checks and confirmations
            if not confirm_deletion and not dry_run:
                raise create_tool_error(
                    "Deletion not confirmed. Set confirm_deletion=true to proceed with corpus deletion",
                    self.name,
                    {"corpus_name": corpus_name},
                    ErrorCode.TOOL_VALIDATION_FAILED
                )
            
            # Check confirmation phrase if required
            if self.require_explicit_confirmation and confirmation_phrase != corpus_name:
            # Check confirmation phrase if required and provided
            if self.require_explicit_confirmation and confirmation_phrase is not None and confirmation_phrase != corpus_name:
                raise create_tool_error(
                    f"Confirmation phrase must exactly match the corpus name '{corpus_name}' for safety",
                    self.name,
                    {"corpus_name": corpus_name, "provided_phrase": confirmation_phrase},
                    ErrorCode.TOOL_VALIDATION_FAILED
                )            is_empty = corpus_info is None or corpus_info.document_count == 0
            if not is_empty and not force_delete_non_empty and not dry_run:
                raise create_tool_error(
                    f"Corpus '{corpus_name}' contains {corpus_info.document_count} documents. "
                    f"Set force_delete_non_empty=true to delete non-empty corpus",
                    self.name,
                    {"corpus_name": corpus_name, "document_count": corpus_info.document_count},
                    ErrorCode.TOOL_VALIDATION_FAILED
                )
            
            self.logger.info(
                f"{'Dry run' if dry_run else 'Executing'} corpus deletion: '{corpus_name}' "
                f"({corpus_info.document_count if corpus_info else 0} documents)"
            )
            
            # Prepare corpus information for response
            corpus_details = None
            if return_corpus_info and corpus_info:
                corpus_details = {
                    "name": corpus_info.name,
                    "document_count": corpus_info.document_count,
                    "created_at": corpus_info.created_at.isoformat(),
                    "metadata": corpus_info.metadata,
                    "age_days": (datetime.utcnow() - corpus_info.created_at).days
                }
            
            # Handle backup if requested
            backup_info = None
            if backup_before_delete and not dry_run:
                try:
                    backup_info = await self._create_backup(corpus_name, corpus_info)
                except Exception as e:
                    self.logger.warning(f"Backup creation failed: {e}")
                    backup_info = {"status": "failed", "error": str(e)}
            
            # Perform deletion (unless dry run)
            deletion_success = False
            deletion_error = None
            
            if not dry_run:
                try:
                    deletion_success = await self.vector_db.delete_corpus(corpus_name)
                    if not deletion_success:
                        deletion_error = "Vector database returned False for deletion operation"
                except Exception as e:
                    deletion_error = str(e)
                    self.logger.error(f"Failed to delete corpus '{corpus_name}': {e}")
            
            # Verify deletion (if not dry run)
            verification_result = None
            if not dry_run and deletion_success:
                try:
                    # Check that corpus no longer exists
                    updated_corpora = await self.vector_db.list_corpora()
                    if corpus_name in updated_corpora:
                        verification_result = "failed"
                        deletion_error = "Corpus still exists after deletion operation"
                    else:
                        verification_result = "success"
                except Exception as e:
                    verification_result = "unknown"
                    self.logger.warning(f"Could not verify corpus deletion: {e}")
            
            # Prepare response
            result = {
                "corpus_name": corpus_name,
                "dry_run": dry_run,
                "deletion_requested": True,
                "deletion_success": deletion_success if not dry_run else None,
                "verification_result": verification_result,
                "was_empty": is_empty
            }
            
            # Add corpus information if available
            if corpus_details:
                result["deleted_corpus_info"] = corpus_details
            
            # Add backup information if applicable
            if backup_info:
                result["backup_info"] = backup_info
            
            # Add error information if applicable
            if deletion_error:
                result["deletion_error"] = deletion_error
            
            # Add execution metadata
            result["execution_info"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "tool_version": self.version,
                "vector_db_provider": self.vector_db.get_provider_name(),
                "safety_checks": {
                    "confirmation_required": confirm_deletion,
                    "phrase_confirmation": confirmation_phrase == corpus_name if confirmation_phrase else False,
                    "force_non_empty": force_delete_non_empty,
                    "backup_requested": backup_before_delete
                }
            }
            
            # Generate appropriate message
            if dry_run:
                result["message"] = f"Dry run completed: corpus '{corpus_name}' would be deleted"
                if corpus_info:
                    result["message"] += f" (contains {corpus_info.document_count} documents)"
            elif deletion_success and verification_result == "success":
                result["message"] = f"Successfully deleted corpus '{corpus_name}'"
                if corpus_info:
                    result["message"] += f" and its {corpus_info.document_count} documents"
            elif deletion_success and verification_result != "success":
                result["message"] = f"Deletion operation completed but verification {verification_result or 'failed'}"
            else:
                result["message"] = f"Failed to delete corpus '{corpus_name}': {deletion_error or 'Unknown error'}"
            
            # If deletion failed, this is an error condition
            if not dry_run and not deletion_success:
                raise create_tool_error(
                    f"Corpus deletion failed: {deletion_error or 'Unknown error'}",
                    self.name,
                    {"corpus_name": corpus_name, "deletion_error": deletion_error},
                    ErrorCode.TOOL_EXECUTION_FAILED
                )
            
            return result
            
        except LexoraError:
            # Re-raise LexoraErrors as-is
            raise
            
        except Exception as e:
            # Wrap other exceptions
            raise create_tool_error(
                f"Unexpected error deleting corpus '{corpus_name}': {str(e)}",
                self.name,
                {"corpus_name": corpus_name, "error_type": type(e).__name__},
                ErrorCode.TOOL_EXECUTION_FAILED,
                e
            )
    
    async def _create_backup(self, corpus_name: str, corpus_info: Optional[CorpusInfo]) -> Dict[str, Any]:
        """
        Create a backup of the corpus before deletion.
        
        This is a placeholder implementation. In a real system, you would
        implement actual backup functionality.
        
        Args:
            corpus_name: Name of the corpus to backup
            corpus_info: Information about the corpus
            
        Returns:
            Dictionary containing backup information
        """
        # Placeholder implementation
        backup_id = f"backup_{corpus_name}_{int(datetime.utcnow().timestamp())}"
        
        return {
            "status": "not_implemented",
            "backup_id": backup_id,
            "note": "Backup functionality not implemented in this version",
            "corpus_name": corpus_name,
            "document_count": corpus_info.document_count if corpus_info else 0,
            "timestamp": datetime.utcnow().isoformat()
        }


# Convenience function for creating the tool
def create_delete_corpus_tool(vector_db: BaseVectorDB, **kwargs) -> DeleteCorpusTool:
    """
    Create a DeleteCorpusTool instance.
    
    Args:
        vector_db: Vector database instance
        **kwargs: Additional configuration options
        
    Returns:
        Configured DeleteCorpusTool instance
    """
    return DeleteCorpusTool(vector_db=vector_db, **kwargs)