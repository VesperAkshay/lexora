"""
Update Document Tool for the Lexora Agentic RAG SDK.

This tool allows users to modify existing documents in corpora with automatic
re-embedding and validation to maintain data consistency.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from .base_tool import BaseTool, ToolParameter, ParameterType
from ..models.core import Document
from ..vector_db.base_vector_db import BaseVectorDB
from ..utils.embeddings import EmbeddingManager
from ..exceptions import LexoraError, ErrorCode, create_tool_error
from ..utils.logging import get_logger


class UpdateDocumentTool(BaseTool):
    """
    Tool for updating existing documents in corpora.
    
    This tool provides safe document updates with automatic re-embedding,
    validation, and rollback capabilities. It ensures data consistency
    by validating document existence before updates and maintaining
    embedding synchronization.
    """
    
    def __init__(
        self,
        vector_db: BaseVectorDB,
        embedding_manager: EmbeddingManager,
        **kwargs
    ):
        """
        Initialize the update document tool.
        
        Args:
            vector_db: Vector database instance for document updates
            embedding_manager: Embedding manager for re-embedding updated content
            **kwargs: Additional configuration options
            
        Raises:
            LexoraError: If required dependencies are not provided or invalid
        """
        super().__init__(**kwargs)
        
        if not isinstance(vector_db, BaseVectorDB):
            raise create_tool_error(
                "vector_db must be an instance of BaseVectorDB",
                "update_document",
                None,
                ErrorCode.TOOL_INVALID_PARAMETERS
            )
        
        if not isinstance(embedding_manager, EmbeddingManager):
            raise create_tool_error(
                "embedding_manager must be an instance of EmbeddingManager",
                "update_document",
                None,
                ErrorCode.TOOL_INVALID_PARAMETERS
            )
        
        self.vector_db = vector_db
        self.embedding_manager = embedding_manager
        self.logger = get_logger(self.__class__.__name__)
        
        # Configuration
        self.auto_regenerate_embeddings = kwargs.get('auto_regenerate_embeddings', True)
        self.preserve_original_metadata = kwargs.get('preserve_original_metadata', True)
        self.validate_before_update = kwargs.get('validate_before_update', True)
    
    @property
    def name(self) -> str:
        """Tool name identifier."""
        return "update_document"
    
    @property
    def description(self) -> str:
        """Tool description for users and LLMs."""
        return (
            "Update existing documents in a corpus with automatic re-embedding. "
            "Supports updating content, metadata, or both while maintaining data "
            "consistency. Automatically regenerates embeddings when content changes "
            "and provides validation to ensure document integrity."
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
                description="Name of the corpus containing the document to update",
                required=True
            ),
            ToolParameter(
                name="document_id",
                type=ParameterType.STRING,
                description="ID of the document to update",
                required=True
            ),
            ToolParameter(
                name="new_content",
                type=ParameterType.STRING,
                description="New content for the document (if updating content)",
                required=False
            ),
            ToolParameter(
                name="new_metadata",
                type=ParameterType.OBJECT,
                description="New metadata for the document (if updating metadata)",
                required=False,
                default={}
            ),
            ToolParameter(
                name="merge_metadata",
                type=ParameterType.BOOLEAN,
                description="Whether to merge new metadata with existing metadata (true) or replace it (false)",
                required=False,
                default=True
            ),
            ToolParameter(
                name="regenerate_embedding",
                type=ParameterType.BOOLEAN,
                description="Whether to regenerate embeddings after content update",
                required=False,
                default=True
            ),
            ToolParameter(
                name="validate_update",
                type=ParameterType.BOOLEAN,
                description="Whether to validate the document exists before updating",
                required=False,
                default=True
            ),
            ToolParameter(
                name="dry_run",
                type=ParameterType.BOOLEAN,
                description="If true, validate update but don't actually modify the document",
                required=False,
                default=False
            ),
            ToolParameter(
                name="return_updated_document",
                type=ParameterType.BOOLEAN,
                description="Whether to return the updated document in the response",
                required=False,
                default=True
            ),
            ToolParameter(
                name="backup_original",
                type=ParameterType.BOOLEAN,
                description="Whether to create a backup of the original document",
                required=False,
                default=False
            )
        ]
    
    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute document update.
        
        Args:
            **kwargs: Validated parameters for document update
            
        Returns:
            Dictionary containing update results and document information
            
        Raises:
            LexoraError: If document update fails
        """
        corpus_name = kwargs["corpus_name"]
        document_id = kwargs["document_id"]
        new_content = kwargs.get("new_content")
        new_metadata = kwargs.get("new_metadata", None)  # Changed from {} to None to distinguish "not provided" vs "empty dict"
        merge_metadata = kwargs.get("merge_metadata", True)
        regenerate_embedding = kwargs.get("regenerate_embedding", True)
        validate_update = kwargs.get("validate_update", True)
        dry_run = kwargs.get("dry_run", False)
        return_updated_document = kwargs.get("return_updated_document", True)
        backup_original = kwargs.get("backup_original", False)
        
        try:
            # Validate input - must have either new content or new metadata
            if not new_content and not new_metadata:
                raise create_tool_error(
                    "Must provide either 'new_content' or 'new_metadata' to update the document",
                    self.name,
                    {"corpus_name": corpus_name, "document_id": document_id},
                    ErrorCode.TOOL_INVALID_PARAMETERS
                )
            
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
            
            # Get original document if validation is enabled
            original_document = None
            if validate_update or backup_original or return_updated_document:
                try:
                    if hasattr(self.vector_db, 'get_document'):
                        original_document = await self.vector_db.get_document(corpus_name, document_id)
                    else:
                        raise create_tool_error(
                            f"Vector database does not support get_document operation",
                            self.name,
                            {"corpus_name": corpus_name, "document_id": document_id},
                            ErrorCode.TOOL_EXECUTION_FAILED
                        )
                    
                    if not original_document and validate_update:
                        raise create_tool_error(
                            f"Document '{document_id}' not found in corpus '{corpus_name}'",
                            self.name,
                            {"corpus_name": corpus_name, "document_id": document_id},
                            ErrorCode.TOOL_VALIDATION_FAILED
                        )
                        
                except Exception as e:
                    if validate_update:
                        raise create_tool_error(
                            f"Failed to retrieve document '{document_id}' for validation: {str(e)}",
                            self.name,
                            {"corpus_name": corpus_name, "document_id": document_id, "error_type": type(e).__name__},
                            ErrorCode.TOOL_EXECUTION_FAILED,
                            e
                        )
                    else:
                        self.logger.warning(f"Could not retrieve original document for backup: {e}")
            
            self.logger.info(
                f"{'Dry run' if dry_run else 'Executing'} document update: '{document_id}' in corpus '{corpus_name}'"
            )
            
            # Create backup if requested
            backup_info = None
            if backup_original and original_document:
                # Safely copy embedding - handle numpy arrays, lists, and other types
                original_embedding_copy = None
                if original_document.embedding is not None:
                    if hasattr(original_document.embedding, 'copy'):
                        # Numpy array or object with copy method
                        original_embedding_copy = original_document.embedding.copy()
                    elif isinstance(original_document.embedding, (list, tuple)):
                        # List or tuple - create a new list
                        original_embedding_copy = list(original_document.embedding)
                    else:
                        # Other types - try to convert to list
                        try:
                            original_embedding_copy = list(original_document.embedding)
                        except (TypeError, ValueError):
                            # If conversion fails, keep reference (not ideal but safe)
                            original_embedding_copy = original_document.embedding
                
                backup_info = {
                    "document_id": original_document.id,
                    "original_content": original_document.content,
                    "original_metadata": original_document.metadata.copy(),
                    "original_embedding": original_embedding_copy,
                    "backup_timestamp": datetime.utcnow().isoformat()
                }
            
            # Prepare updated document
            if original_document:
                # Start with original document
                updated_content = new_content if new_content is not None else original_document.content
                
                # Handle metadata updates properly
                if merge_metadata and new_metadata is not None:
                    # Merge new metadata into a copy of original
                    updated_metadata = original_document.metadata.copy()
                    updated_metadata.update(new_metadata)
                elif new_metadata is not None:
                    # Replace with new metadata (even if empty dict)
                    updated_metadata = new_metadata.copy() if isinstance(new_metadata, dict) else new_metadata
                else:
                    # Keep original metadata
                    updated_metadata = original_document.metadata.copy()
                
                # Keep original embedding initially
                updated_embedding = original_document.embedding
            else:
                # No original document found, create new one
                updated_content = new_content or ""
                updated_metadata = new_metadata.copy() if new_metadata is not None else {}
                updated_embedding = None
            
            # Add update metadata
            update_metadata = {
                "updated_at": datetime.utcnow().isoformat(),
                "updated_by": "lexora_update_document_tool",
                "update_version": (
                    (updated_metadata.get("update_version", 0) + 1)
                    if isinstance(updated_metadata.get("update_version", 0), (int, float))
                    else 1
                ),                "content_updated": new_content is not None,
                "metadata_updated": bool(new_metadata)
            }
            updated_metadata.update(update_metadata)
            
            # Regenerate embedding if content changed and requested
            embedding_regenerated = False
            if new_content is not None and regenerate_embedding and not dry_run:
                try:
                    updated_embedding = await self.embedding_manager.generate_embedding(updated_content)
                    embedding_regenerated = True
                    updated_metadata["embedding_regenerated"] = True
                    updated_metadata["embedding_model"] = self.embedding_manager.get_model_name()
                except Exception as e:
                    self.logger.warning(f"Failed to regenerate embedding: {e}")
                    updated_metadata["embedding_regeneration_failed"] = str(e)
            
            # Create updated document
            updated_document = Document(
                id=document_id,
                content=updated_content,
                metadata=updated_metadata,
                embedding=updated_embedding
            )
            
            # Perform update (unless dry run)
            update_success = False
            update_error = None
            
            if not dry_run:
                try:
                    update_success = await self.vector_db.update_document(corpus_name, updated_document)
                    if not update_success:
                        update_error = "Vector database returned False for update operation"
                except Exception as e:
                    update_error = str(e)
                    self.logger.error(f"Failed to update document '{document_id}': {e}")
            
            # Prepare response
            result = {
                "corpus_name": corpus_name,
                "document_id": document_id,
                "dry_run": dry_run,
                "update_success": update_success if not dry_run else None,
                "changes_made": {
                    "content_updated": new_content is not None,
                    "metadata_updated": bool(new_metadata),
                    "embedding_regenerated": embedding_regenerated
                }
            }
            
            # Add original document info if available
            if original_document:
                result["original_document_info"] = {
                    "content_length": len(original_document.content),
                    "metadata_keys": list(original_document.metadata.keys()),
                    "had_embedding": original_document.embedding is not None
                }
            
            # Add updated document if requested
            if return_updated_document:
                result["updated_document"] = {
                    "id": updated_document.id,
                    "content": updated_document.content,
                    "content_length": len(updated_document.content),
                    "metadata": updated_document.metadata,
                    "has_embedding": updated_document.embedding is not None
                }
            
            # Add backup info if created
            if backup_info:
                result["backup_info"] = backup_info
            
            # Add error information if applicable
            if update_error:
                result["update_error"] = update_error
            
            # Add execution metadata
            result["execution_info"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "tool_version": self.version,
                "vector_db_provider": self.vector_db.get_provider_name(),
                "parameters_used": {
                    "merge_metadata": merge_metadata,
                    "regenerate_embedding": regenerate_embedding,
                    "validate_update": validate_update,
                    "backup_original": backup_original
                }
            }
            
            # Generate appropriate message
            if dry_run:
                changes = []
                if new_content is not None:
                    changes.append("content")
                if new_metadata:
                    changes.append("metadata")
                result["message"] = f"Dry run completed: document '{document_id}' would be updated ({', '.join(changes)})"
            elif update_success:
                changes = []
                if new_content is not None:
                    changes.append("content")
                if new_metadata:
                    changes.append("metadata")
                if embedding_regenerated:
                    changes.append("embedding")
                result["message"] = f"Successfully updated document '{document_id}' ({', '.join(changes)})"
            else:
                result["message"] = f"Failed to update document '{document_id}': {update_error or 'Unknown error'}"
            
            # If update failed, this is an error condition
            if not dry_run and not update_success:
                raise create_tool_error(
                    f"Document update failed: {update_error or 'Unknown error'}",
                    self.name,
                    {"corpus_name": corpus_name, "document_id": document_id, "update_error": update_error},
                    ErrorCode.TOOL_EXECUTION_FAILED
                )
            
            return result
            
        except LexoraError:
            # Re-raise LexoraErrors as-is
            raise
            
        except Exception as e:
            # Wrap other exceptions
            raise create_tool_error(
                f"Unexpected error updating document '{document_id}' in corpus '{corpus_name}': {str(e)}",
                self.name,
                {"corpus_name": corpus_name, "document_id": document_id, "error_type": type(e).__name__},
                ErrorCode.TOOL_EXECUTION_FAILED,
                e
            )


# Convenience function for creating the tool
def create_update_document_tool(
    vector_db: BaseVectorDB,
    embedding_manager: EmbeddingManager,
    **kwargs
) -> UpdateDocumentTool:
    """
    Create an UpdateDocumentTool instance.
    
    Args:
        vector_db: Vector database instance
        embedding_manager: Embedding manager instance
        **kwargs: Additional configuration options
        
    Returns:
        Configured UpdateDocumentTool instance
    """
    return UpdateDocumentTool(
        vector_db=vector_db,
        embedding_manager=embedding_manager,
        **kwargs
    )