"""
Create Corpus Tool for the Lexora Agentic RAG SDK.

This tool allows users to create new document corpora in the vector database
with optional metadata and configuration parameters.
"""

from typing import Any, Dict, Optional
from datetime import datetime

from .base_tool import BaseTool, ToolParameter, ParameterType
from ..vector_db.base_vector_db import BaseVectorDB
from ..exceptions import LexoraError, ErrorCode, create_tool_error, create_vector_db_error
from ..utils.logging import get_logger


class CreateCorpusTool(BaseTool):
    """
    Tool for creating new document corpora in the vector database.
    
    This tool provides a standardized interface for creating corpora with
    optional metadata and configuration parameters. It handles validation,
    error reporting, and provides structured responses.
    """
    
    def __init__(self, vector_db: BaseVectorDB, **kwargs):
        """
        Initialize the create corpus tool.
        
        Args:
            vector_db: Vector database instance to use for corpus creation
            **kwargs: Additional configuration options
            
        Raises:
            LexoraError: If vector_db is not provided or invalid
        """
        super().__init__(**kwargs)
        
        if not isinstance(vector_db, BaseVectorDB):
            raise create_tool_error(
                "vector_db must be an instance of BaseVectorDB",
                "create_corpus",
                ErrorCode.TOOL_INVALID_PARAMETERS
            )
        
        self.vector_db = vector_db
        self.logger = get_logger(self.__class__.__name__)
    
    @property
    def name(self) -> str:
        """Tool name identifier."""
        return "create_corpus"
    
    @property
    def description(self) -> str:
        """Tool description for users and LLMs."""
        return (
            "Create a new document corpus in the vector database. "
            "A corpus is a collection of documents that can be searched together. "
            "You can optionally provide metadata and configuration parameters."
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
                description="Name of the corpus to create. Must be unique and contain only alphanumeric characters, hyphens, and underscores.",
                required=True,
                pattern=r"^[a-zA-Z0-9_-]+$"
            ),
            ToolParameter(
                name="description",
                type=ParameterType.STRING,
                description="Optional description of the corpus and its intended use",
                required=False,
                default=""
            ),
            ToolParameter(
                name="metadata",
                type=ParameterType.OBJECT,
                description="Optional metadata dictionary with additional corpus information",
                required=False,
                default={}
            ),
            ToolParameter(
                name="embedding_dimension",
                type=ParameterType.INTEGER,
                description="Dimension of embeddings for this corpus (if supported by vector DB)",
                required=False,
                minimum=1,
                maximum=4096
            ),
            ToolParameter(
                name="similarity_metric",
                type=ParameterType.STRING,
                description="Similarity metric to use for this corpus",
                required=False,
                enum=["cosine", "euclidean", "dot_product"],
                default="cosine"
            ),
            ToolParameter(
                name="overwrite_existing",
                type=ParameterType.BOOLEAN,
                description="Whether to overwrite an existing corpus with the same name",
                required=False,
                default=False
            )
        ]
    
    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute corpus creation.
        
        Args:
            **kwargs: Validated parameters for corpus creation
            
        Returns:
            Dictionary containing creation results and corpus information
            
        Raises:
            LexoraError: If corpus creation fails
        """
        corpus_name = kwargs["corpus_name"]
        description = kwargs.get("description", "")
        metadata = kwargs.get("metadata", {})
        embedding_dimension = kwargs.get("embedding_dimension")
        similarity_metric = kwargs.get("similarity_metric", "cosine")
        overwrite_existing = kwargs.get("overwrite_existing", False)
        
        try:
            # Ensure vector database is connected
            if not self.vector_db.is_connected():
                await self.vector_db.connect()
            
            # Check if corpus already exists
            existing_corpora = await self.vector_db.list_corpora()
            corpus_exists = corpus_name in existing_corpora
            
            if corpus_exists and not overwrite_existing:
                raise create_tool_error(
                    f"Corpus '{corpus_name}' already exists. Use overwrite_existing=true to replace it.",
                    self.name,
                    {"corpus_name": corpus_name, "existing_corpora": existing_corpora},
                    ErrorCode.TOOL_VALIDATION_FAILED
                )
            
            # If overwriting, delete the existing corpus first            if corpus_exists and overwrite_existing:
                self.logger.info(f"Deleting existing corpus '{corpus_name}' for overwrite")
                try:
                    await self.vector_db.delete_corpus(corpus_name)
                except Exception as e:
                    raise create_tool_error(
                        f"Failed to delete existing corpus '{corpus_name}' during overwrite: {str(e)}",
                        self.name,
                        {"corpus_name": corpus_name, "error_type": type(e).__name__},
                        ErrorCode.TOOL_EXECUTION_FAILED,
                        e
                    )            
            # Prepare corpus creation parameters
            creation_params = {
                "description": description,
                "similarity_metric": similarity_metric,
                "created_at": datetime.utcnow().isoformat(),
                "created_by": "lexora_rag_agent",
                **metadata  # Include user-provided metadata
            }
            
            # Add embedding dimension if specified
            if embedding_dimension is not None:
                creation_params["embedding_dimension"] = embedding_dimension
            
            # Create the corpus
            self.logger.info(f"Creating corpus '{corpus_name}' with parameters: {creation_params}")
            success = await self.vector_db.create_corpus(corpus_name, **creation_params)
            
            if not success:
                raise create_tool_error(
                    f"Failed to create corpus '{corpus_name}' - vector database returned False",
                    self.name,
                    {"corpus_name": corpus_name, "creation_params": creation_params},
                    ErrorCode.TOOL_EXECUTION_FAILED
                )
            
            # Verify corpus was created by getting its info
            try:
                corpus_info = await self.vector_db.get_corpus_info(corpus_name)
                corpus_details = {
                    "name": corpus_info.name,
                    "document_count": corpus_info.document_count,
                    "created_at": corpus_info.created_at.isoformat(),
                    "metadata": corpus_info.metadata
                }
            except Exception as e:
                # If we can't get corpus info, still report success but with limited details
                self.logger.warning(f"Created corpus '{corpus_name}' but couldn't retrieve details: {e}")
                corpus_details = {
                    "name": corpus_name,
                    "document_count": 0,
                    "created_at": datetime.utcnow().isoformat(),
                    "metadata": creation_params
                }
            
            # Return success response
            return {
                "corpus_name": corpus_name,
                "created": True,
                "overwritten": corpus_exists and overwrite_existing,
                "corpus_info": corpus_details,
                "creation_parameters": {
                    "description": description,
                    "similarity_metric": similarity_metric,
                    "embedding_dimension": embedding_dimension,
                    "metadata": metadata
                },
                "message": f"Successfully created corpus '{corpus_name}'"
            }
            
        except LexoraError:
            # Re-raise LexoraErrors as-is
            raise
            
        except Exception as e:
            # Wrap other exceptions
            raise create_tool_error(
                f"Unexpected error creating corpus '{corpus_name}': {str(e)}",
                self.name,
                {"corpus_name": corpus_name, "error_type": type(e).__name__},
                ErrorCode.TOOL_EXECUTION_FAILED,
                e
            )


# Convenience function for creating the tool
def create_corpus_tool(vector_db: BaseVectorDB, **kwargs) -> CreateCorpusTool:
    """
    Create a CreateCorpusTool instance.
    
    Args:
        vector_db: Vector database instance
        **kwargs: Additional configuration options
        
    Returns:
        Configured CreateCorpusTool instance
    """
    return CreateCorpusTool(vector_db=vector_db, **kwargs)