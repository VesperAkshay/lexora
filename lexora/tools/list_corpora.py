"""
List Corpora Tool for the Lexora Agentic RAG SDK.

This tool lists all available document corpora in the vector database
with basic information and statistics for each corpus.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from .base_tool import BaseTool, ToolParameter, ParameterType
from ..models.core import CorpusInfo
from ..vector_db.base_vector_db import BaseVectorDB
from ..exceptions import LexoraError, ErrorCode, create_tool_error
from ..utils.logging import get_logger


class ListCorporaTool(BaseTool):
    """
    Tool for listing all available document corpora.
    
    This tool provides an overview of all corpora in the vector database,
    including basic statistics and metadata for each corpus. Useful for
    discovery and management of available knowledge bases.
    """
    
    def __init__(self, vector_db: BaseVectorDB, **kwargs):
        """
        Initialize the list corpora tool.
        
        Args:
            vector_db: Vector database instance to query for corpora
            **kwargs: Additional configuration options
            
        Raises:
            LexoraError: If vector_db is not provided or invalid
        """
        super().__init__(**kwargs)
        
        if not isinstance(vector_db, BaseVectorDB):
            raise create_tool_error(
                "vector_db must be an instance of BaseVectorDB",
                "list_corpora",
                None,
                ErrorCode.TOOL_INVALID_PARAMETERS
            )
        
        self.vector_db = vector_db
        self.logger = get_logger(self.__class__.__name__)
        
        # Configuration
        self.vector_db = vector_db
        self.logger = get_logger(self.__class__.__name__)    
    @property
    def name(self) -> str:
        """Tool name identifier."""
        return "list_corpora"
    
    @property
    def description(self) -> str:
        """Tool description for users and LLMs."""
        return (
            "List all available document corpora in the vector database. "
            "Returns basic information about each corpus including name, "
            "document count, creation date, and metadata. Useful for "
            "discovering available knowledge bases and their contents."
        )
    
    @property
    def version(self) -> str:
        """Tool version for compatibility tracking."""
        return "1.0.0"
    
    def _setup_parameters(self) -> None:
        """Set up tool parameters."""
        self._parameters = [
            ToolParameter(
                name="include_details",
                type=ParameterType.BOOLEAN,
                description="Whether to include detailed information for each corpus",
                required=False,
                default=True
            ),
            ToolParameter(
                name="include_empty",
                type=ParameterType.BOOLEAN,
                description="Whether to include corpora with no documents",
                required=False,
                default=True
            ),
            ToolParameter(
                name="sort_by",
                type=ParameterType.STRING,
                description="Field to sort corpora by",
                required=False,
                default="name",
                enum=["name", "created_at", "document_count", "last_modified"]
            ),
            ToolParameter(
                name="sort_order",
                type=ParameterType.STRING,
                description="Sort order for results",
                required=False,
                default="asc",
                enum=["asc", "desc"]
            ),
            ToolParameter(
                name="name_filter",
                type=ParameterType.STRING,
                description="Optional filter to match corpus names (case-insensitive substring match)",
                required=False
            ),
            ToolParameter(
                name="metadata_filter",
                type=ParameterType.OBJECT,
                description="Optional metadata filters to apply (key-value pairs)",
                required=False,
                default={}
            ),
            ToolParameter(
                name="limit",
                type=ParameterType.INTEGER,
                description="Maximum number of corpora to return",
                required=False,
                minimum=1,
                maximum=1000
            )
        ]
    
    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute corpus listing.
        
        Args:
            **kwargs: Validated parameters for corpus listing
            
        Returns:
            Dictionary containing list of corpora and summary statistics
            
        Raises:
            LexoraError: If listing fails
        """
        include_details = kwargs.get("include_details", True)
        include_empty = kwargs.get("include_empty", True)
        sort_by = kwargs.get("sort_by", "name")
        sort_order = kwargs.get("sort_order", "asc")
        name_filter = kwargs.get("name_filter")
        metadata_filter = kwargs.get("metadata_filter", {})
        limit = kwargs.get("limit")
        
        try:
            # Ensure vector database is connected
            if not self.vector_db.is_connected():
                await self.vector_db.connect()
            
            self.logger.info("Listing all available corpora")
            
            # Get list of corpus names
            corpus_names = await self.vector_db.list_corpora()
            
            if not corpus_names:
                return {
                    "corpora": [],
                    "total_count": 0,
                    "summary": {
                        "total_corpora": 0,
                        "total_documents": 0,
                        "empty_corpora": 0,
                        "non_empty_corpora": 0
                    },
                    "filters_applied": {
                        "include_empty": include_empty,
                        "name_filter": name_filter,
                        "metadata_filter": metadata_filter
                    },
                    "message": "No corpora found in the vector database"
                }
            
            # Collect detailed information for each corpus
            corpora_info = []
            total_documents = 0
            empty_corpora_count = 0
            failed_retrievals = []
            
            for corpus_name in corpus_names:
                try:
                    if include_details:
                        # Get detailed corpus information
                        corpus_info = await self.vector_db.get_corpus_info(corpus_name)
                        
                        corpus_data = {
                            "name": corpus_info.name,
                            "document_count": corpus_info.document_count,
                            "created_at": corpus_info.created_at.isoformat(),
                            "metadata": corpus_info.metadata
                        }
                        
                        # Add derived information
                        corpus_data["is_empty"] = corpus_info.document_count == 0
                        corpus_data["last_modified"] = corpus_info.metadata.get(
                            "last_modified", 
                            corpus_info.created_at.isoformat()
                        )
                        
                        total_documents += corpus_info.document_count
                        if corpus_info.document_count == 0:
                            empty_corpora_count += 1
                    
                    else:
                        # Basic information only
                        corpus_data = {
                            "name": corpus_name,
                            "document_count": None,
                            "created_at": None,
                            "metadata": {},
                            "is_empty": None,
                            "last_modified": None
                        }
                    
                    corpora_info.append(corpus_data)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to get info for corpus '{corpus_name}': {e}")
                    failed_retrievals.append({
                        "corpus_name": corpus_name,
                        "error": str(e)
                    })
                    
                    # Add basic entry for failed corpus
                    corpora_info.append({
                        "name": corpus_name,
                        "document_count": None,
                        "created_at": None,
                        "metadata": {},
                        "is_empty": None,
                        "last_modified": None,
                        "error": str(e)
                    })
            
            # Apply filters
            filtered_corpora = []
            
            for corpus in corpora_info:
                # Skip empty corpora if not requested
                if not include_empty and corpus.get("is_empty") is True):
                    continue
                
                # Apply name filter
                if name_filter:
                    if name_filter.lower() not in corpus["name"].lower():
                        continue
                
                # Apply metadata filters
                if metadata_filter:
                    corpus_metadata = corpus.get("metadata", {})
                    matches_filters = True
                    
                    for filter_key, filter_value in metadata_filter.items():
                        if filter_key not in corpus_metadata:
                            matches_filters = False
                            break
                        
                        corpus_value = corpus_metadata[filter_key]
                        if isinstance(filter_value, list):
                            if corpus_value not in filter_value:
                                matches_filters = False
                                break
                        else:
                            if corpus_value != filter_value:
                                matches_filters = False
                                break
                    
                    if not matches_filters:
                        continue
                
                filtered_corpora.append(corpus)
            
            # Sort results
            if sort_by and filtered_corpora:
                reverse_order = sort_order == "desc"
                
                if sort_by == "name":
                    filtered_corpora.sort(key=lambda x: x["name"], reverse=reverse_order)
                elif sort_by == "document_count":
                    # Handle None values for document count
                    filtered_corpora.sort(
                        key=lambda x: x["document_count"] if x["document_count"] is not None else -1,
                        reverse=reverse_order
                    )
                elif sort_by == "created_at":
                    # Handle None values for created_at
                    filtered_corpora.sort(
                        key=lambda x: x["created_at"] if x["created_at"] is not None else "",
                        reverse=reverse_order
                    )
                elif sort_by == "last_modified":
                    # Handle None values for last_modified
                    filtered_corpora.sort(
                        key=lambda x: x["last_modified"] if x["last_modified"] is not None else "",
                        reverse=reverse_order
                    )
            
            # Apply limit
            if limit and len(filtered_corpora) > limit:
                filtered_corpora = filtered_corpora[:limit]
            
            # Calculate summary statistics
            non_empty_corpora = sum(1 for c in corpora_info if c.get("document_count", 0) > 0)
            
            summary_stats = {
                "total_corpora": len(corpus_names),
                "total_documents": total_documents if include_details else None,
                "empty_corpora": empty_corpora_count if include_details else None,
                "non_empty_corpora": non_empty_corpora if include_details else None,
                "filtered_count": len(filtered_corpora),
                "failed_retrievals": len(failed_retrievals)
            }
            
            # Prepare response
            response = {
                "corpora": filtered_corpora,
                "total_count": len(filtered_corpora),
                "summary": summary_stats,
                "filters_applied": {
                    "include_details": include_details,
                    "include_empty": include_empty,
                    "sort_by": sort_by,
                    "sort_order": sort_order,
                    "name_filter": name_filter,
                    "metadata_filter": metadata_filter,
                    "limit": limit
                },
                "execution_info": {
                    "timestamp": datetime.utcnow().isoformat(),
# --- at the top of the file (add to your existing imports) ---
from datetime import datetime, timezone

# …later, in the execution_info block…
"execution_info": {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "tool_version": self.version,
    "vector_db_provider": self.vector_db.get_provider_name()
}                    "vector_db_provider": self.vector_db.get_provider_name()
                }
            }
            
            # Add failed retrievals if any
            if failed_retrievals:
                response["failed_retrievals"] = failed_retrievals
            
            # Add appropriate message
            if len(filtered_corpora) == 0:
                if len(corpus_names) > 0:
                    response["message"] = f"No corpora match the specified filters. {len(corpus_names)} total corpora available."
                else:
                    response["message"] = "No corpora found in the vector database"
            else:
                response["message"] = f"Found {len(filtered_corpora)} corpora"
                if len(filtered_corpora) != len(corpus_names):
                    response["message"] += f" (filtered from {len(corpus_names)} total)"
            
            return response
            
        except LexoraError:
            # Re-raise LexoraErrors as-is
            raise
            
        except Exception as e:
            # Wrap other exceptions
            raise create_tool_error(
                f"Unexpected error listing corpora: {str(e)}",
                self.name,
                {"error_type": type(e).__name__},
                ErrorCode.TOOL_EXECUTION_FAILED,
                e
            )


# Convenience function for creating the tool
def create_list_corpora_tool(vector_db: BaseVectorDB, **kwargs) -> ListCorporaTool:
    """
    Create a ListCorporaTool instance.
    
    Args:
        vector_db: Vector database instance
        **kwargs: Additional configuration options
        
    Returns:
        Configured ListCorporaTool instance
    """
    return ListCorporaTool(vector_db=vector_db, **kwargs)