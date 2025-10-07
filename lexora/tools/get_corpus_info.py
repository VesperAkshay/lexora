"""
Get Corpus Info Tool for the Lexora Agentic RAG SDK.

This tool retrieves detailed information and statistics about a specific
document corpus, including metadata, document counts, and usage statistics.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from .base_tool import BaseTool, ToolParameter, ParameterType
from ..models.core import CorpusInfo
from ..vector_db.base_vector_db import BaseVectorDB
from ..exceptions import LexoraError, ErrorCode, create_tool_error
from ..utils.logging import get_logger


class GetCorpusInfoTool(BaseTool):
    """
    Tool for retrieving detailed information about a specific corpus.
    
    This tool provides comprehensive information about a corpus including
    document count, creation date, metadata, and additional statistics
    that can be computed from the corpus contents.
    """
    
    def __init__(self, vector_db: BaseVectorDB, **kwargs):
        """
        Initialize the get corpus info tool.
        
        Args:
            vector_db: Vector database instance to query for corpus information
            **kwargs: Additional configuration options
            
        Raises:
            LexoraError: If vector_db is not provided or invalid
        """
        super().__init__(**kwargs)
        
        if not isinstance(vector_db, BaseVectorDB):
            raise create_tool_error(
                "vector_db must be an instance of BaseVectorDB",
                "get_corpus_info",
                None,
                ErrorCode.TOOL_INVALID_PARAMETERS
            )
        
        self.vector_db = vector_db
        self.logger = get_logger(self.__class__.__name__)
        
        # Configuration
        self.include_sample_documents = kwargs.get('include_sample_documents', False)
        self.sample_size = kwargs.get('sample_size', 3)
    
    @property
    def name(self) -> str:
        """Tool name identifier."""
        return "get_corpus_info"
    
    @property
    def description(self) -> str:
        """Tool description for users and LLMs."""
        return (
            "Retrieve detailed information and statistics about a specific document corpus. "
            "Returns comprehensive data including document count, creation date, metadata, "
            "storage statistics, and optionally sample documents. Useful for corpus "
            "analysis and management."
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
                description="Name of the corpus to get information about",
                required=True
            ),
            ToolParameter(
                name="include_statistics",
                type=ParameterType.BOOLEAN,
                description="Whether to include detailed statistics about the corpus",
                required=False,
                default=True
            ),
            ToolParameter(
                name="include_sample_documents",
                type=ParameterType.BOOLEAN,
                description="Whether to include sample documents from the corpus",
                required=False,
                default=False
            ),
            ToolParameter(
                name="sample_size",
                type=ParameterType.INTEGER,
                description="Number of sample documents to include (if enabled)",
                required=False,
                default=3,
                minimum=1,
                maximum=20
            ),
            ToolParameter(
                name="include_metadata_analysis",
                type=ParameterType.BOOLEAN,
                description="Whether to analyze and summarize document metadata",
                required=False,
                default=True
            ),
            ToolParameter(
                name="compute_content_stats",
                type=ParameterType.BOOLEAN,
                description="Whether to compute content-based statistics (may be slow for large corpora)",
                required=False,
                default=False
            )
        ]
    
    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute corpus information retrieval.
        
        Args:
            **kwargs: Validated parameters for corpus info retrieval
            
        Returns:
            Dictionary containing detailed corpus information and statistics
            
        Raises:
            LexoraError: If corpus info retrieval fails
        """
        corpus_name = kwargs["corpus_name"]
        include_statistics = kwargs.get("include_statistics", True)
        include_sample_documents = kwargs.get("include_sample_documents", False)
        sample_size = kwargs.get("sample_size", 3)
        include_metadata_analysis = kwargs.get("include_metadata_analysis", True)
        compute_content_stats = kwargs.get("compute_content_stats", False)
        
        try:
            # Ensure vector database is connected
            if not self.vector_db.is_connected():
                await self.vector_db.connect()
            
            self.logger.info(f"Retrieving information for corpus '{corpus_name}'")
            
            # Verify corpus exists and get basic info
            try:
                corpus_info = await self.vector_db.get_corpus_info(corpus_name)
            except Exception as e:
                # Check if corpus exists in the list
                existing_corpora = await self.vector_db.list_corpora()
                if corpus_name not in existing_corpora:
                    raise create_tool_error(
                        f"Corpus '{corpus_name}' does not exist. Available corpora: {existing_corpora}",
                        self.name,
                        {"corpus_name": corpus_name, "existing_corpora": existing_corpora},
                        ErrorCode.TOOL_VALIDATION_FAILED
                    )
                else:
                    raise create_tool_error(
                        f"Failed to retrieve corpus info for '{corpus_name}': {str(e)}",
                        self.name,
                        {"corpus_name": corpus_name, "error_type": type(e).__name__},
                        ErrorCode.TOOL_EXECUTION_FAILED,
                        e
                    )
            
            # Build basic corpus information
            result = {
                "corpus_name": corpus_info.name,
                "document_count": corpus_info.document_count,
                "created_at": corpus_info.created_at.isoformat(),
                "metadata": corpus_info.metadata,
                "is_empty": corpus_info.document_count == 0
            }
            
            # Add derived information from metadata
            if corpus_info.metadata:
                result["description"] = corpus_info.metadata.get("description", "")
                result["similarity_metric"] = corpus_info.metadata.get("similarity_metric", "unknown")
                result["embedding_model"] = corpus_info.metadata.get("embedding_model", "unknown")
                result["created_by"] = corpus_info.metadata.get("created_by", "unknown")
                
                # Calculate age
# --- file: lexora/tools/get_corpus_info.py
# (around line 9)
from datetime import datetime, timezone

# ... other imports ...

def _execute(self, corpus_id, ...):
    # ... earlier code ...
-    # At top of file, ensure we import timezone as well
-    from datetime import datetime, timezone

    created_at = datetime.strptime(corpus['created_at'], '%Y-%m-%dT%H:%M:%SZ')
    created_at = created_at.replace(tzinfo=timezone.utc)
    # ... rest of method ...
# Calculate age
if "created_at" in corpus_info.metadata:
    try:
        created_time = datetime.fromisoformat(corpus_info.metadata["created_at"])
        age_seconds = (datetime.now(timezone.utc) - created_time).total_seconds()
        result["age_days"] = round(age_seconds / 86400, 2)
    except (ValueError, TypeError):
        result["age_days"] = None
else:
    age_seconds = (datetime.now(timezone.utc) - corpus_info.created_at).total_seconds()
    result["age_days"] = round(age_seconds / 86400, 2)            
            # Include detailed statistics if requested
            if include_statistics and corpus_info.document_count > 0:
                stats = await self._compute_corpus_statistics(
                    corpus_name, 
                    corpus_info,
                    include_metadata_analysis,
                    compute_content_stats
                )
                result["statistics"] = stats
            
            # Include sample documents if requested
            if include_sample_documents and corpus_info.document_count > 0:
                samples = await self._get_sample_documents(corpus_name, sample_size)
                result["sample_documents"] = samples
            
            # Add execution metadata
            result["retrieval_info"] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool_version": self.version,
                "vector_db_provider": self.vector_db.get_provider_name(),
                "parameters_used": {
                    "include_statistics": include_statistics,
                    "include_sample_documents": include_sample_documents,
                    "sample_size": sample_size if include_sample_documents else None,
                    "include_metadata_analysis": include_metadata_analysis,
                    "compute_content_stats": compute_content_stats
                }
            }
            
            result["message"] = f"Retrieved detailed information for corpus '{corpus_name}'"
            
            return result
            
        except LexoraError:
            # Re-raise LexoraErrors as-is
            raise
            
        except Exception as e:
            # Wrap other exceptions
            raise create_tool_error(
                f"Unexpected error retrieving corpus info for '{corpus_name}': {str(e)}",
                self.name,
                {"corpus_name": corpus_name, "error_type": type(e).__name__},
                ErrorCode.TOOL_EXECUTION_FAILED,
                e
            )
    
    async def _compute_corpus_statistics(
        self, 
        corpus_name: str, 
        corpus_info: CorpusInfo,
        include_metadata_analysis: bool,
        compute_content_stats: bool
    ) -> Dict[str, Any]:
        """
        Compute detailed statistics about the corpus.
        
        Args:
            corpus_name: Name of the corpus
            corpus_info: Basic corpus information
            include_metadata_analysis: Whether to analyze metadata
            compute_content_stats: Whether to compute content statistics
            
        Returns:
            Dictionary containing corpus statistics
        """
        stats = {
            "document_count": corpus_info.document_count,
            "creation_date": corpus_info.created_at.isoformat(),
            "metadata_fields": list(corpus_info.metadata.keys()) if corpus_info.metadata else []
        }
        
        # If we have access to individual documents, compute more detailed stats
        if compute_content_stats and corpus_info.document_count > 0:
            try:
                # This is a simplified approach - in a real implementation,
                # you might want to sample documents or use database aggregation
                content_stats = await self._analyze_content_statistics(corpus_name)
                stats.update(content_stats)
            except Exception as e:
                self.logger.warning(f"Failed to compute content statistics: {e}")
                stats["content_analysis_error"] = str(e)
        
        # Analyze metadata if requested
        if include_metadata_analysis and corpus_info.document_count > 0:
            try:
                metadata_stats = await self._analyze_metadata_statistics(corpus_name)
                stats["metadata_analysis"] = metadata_stats
            except Exception as e:
                self.logger.warning(f"Failed to analyze metadata: {e}")
                stats["metadata_analysis_error"] = str(e)
        
        return stats
    
    async def _analyze_content_statistics(self, corpus_name: str) -> Dict[str, Any]:
        """
        Analyze content-based statistics for the corpus.
        
        This is a placeholder implementation. In a real system, you would
        implement more sophisticated content analysis.
        
        Args:
            corpus_name: Name of the corpus
            
        Returns:
            Dictionary containing content statistics
        """
        # Placeholder implementation
        return {
            "content_analysis_available": False,
            "note": "Content analysis requires sampling documents from the corpus"
        }
    
    async def _analyze_metadata_statistics(self, corpus_name: str) -> Dict[str, Any]:
        """
        Analyze metadata statistics for the corpus.
        
        This is a placeholder implementation. In a real system, you would
        aggregate metadata across all documents.
        
        Args:
            corpus_name: Name of the corpus
            
        Returns:
            Dictionary containing metadata statistics
        """
        # Placeholder implementation
        return {
            "metadata_analysis_available": False,
            "note": "Metadata analysis requires access to individual document metadata"
        }
    
    async def _get_sample_documents(self, corpus_name: str, sample_size: int) -> List[Dict[str, Any]]:
        """
        Get sample documents from the corpus.
        
        Args:
            corpus_name: Name of the corpus
            sample_size: Number of sample documents to retrieve
            
        Returns:
            List of sample document information
        """
        try:
            # Use a simple search to get some documents
            # In a real implementation, you might want random sampling
            sample_results = await self.vector_db.search(
                corpus_name=corpus_name,
                query="",  # Empty query to get any documents
                top_k=sample_size
            )
            
            samples = []
            for result in sample_results:
                doc = result.document
                sample = {
                    "document_id": doc.id,
                    "content_preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "content_length": len(doc.content),
                    "metadata": doc.metadata,
                    "has_embedding": doc.embedding is not None
                }
                samples.append(sample)
            
            return samples
            
        except Exception as e:
            self.logger.warning(f"Failed to retrieve sample documents: {e}")
            return [{
                "error": f"Failed to retrieve sample documents: {str(e)}"
            }]


# Convenience function for creating the tool
def create_get_corpus_info_tool(vector_db: BaseVectorDB, **kwargs) -> GetCorpusInfoTool:
    """
    Create a GetCorpusInfoTool instance.
    
    Args:
        vector_db: Vector database instance
        **kwargs: Additional configuration options
        
    Returns:
        Configured GetCorpusInfoTool instance
    """
    return GetCorpusInfoTool(vector_db=vector_db, **kwargs)