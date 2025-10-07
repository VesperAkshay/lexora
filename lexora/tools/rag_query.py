"""
RAG Query Tool for the Lexora Agentic RAG SDK.

This tool performs semantic search and context retrieval from document corpora,
providing the core RAG functionality for question answering and information retrieval.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from .base_tool import BaseTool, ToolParameter, ParameterType
from ..models.core import SearchResult
from ..vector_db.base_vector_db import BaseVectorDB
from ..utils.embeddings import EmbeddingManager
from ..exceptions import LexoraError, ErrorCode, create_tool_error
from ..utils.logging import get_logger


class RAGQueryTool(BaseTool):
    """
    Tool for performing semantic search and context retrieval in RAG systems.
    
    This tool searches for relevant documents in a corpus using semantic similarity,
    ranks results by relevance, and provides structured responses with metadata
    for downstream processing by reasoning engines.
    """
    
    def __init__(
        self,
        vector_db: BaseVectorDB,
        embedding_manager: EmbeddingManager,
        **kwargs
    ):
        """
        Initialize the RAG query tool.
        
        Args:
            vector_db: Vector database instance for document search
            embedding_manager: Embedding manager for query embedding generation
            **kwargs: Additional configuration options
            
        Raises:
            LexoraError: If required dependencies are not provided or invalid
        """
        if not isinstance(vector_db, BaseVectorDB):
            raise create_tool_error(
                "vector_db must be an instance of BaseVectorDB",
                "rag_query",
                None,
                ErrorCode.TOOL_INVALID_PARAMETERS
            )
        
        if not isinstance(embedding_manager, EmbeddingManager):
            raise create_tool_error(
                "embedding_manager must be an instance of EmbeddingManager",
                "rag_query",
                None,
                ErrorCode.TOOL_INVALID_PARAMETERS
            )
        
        # Configuration (set before calling super().__init__)
        self.default_top_k = kwargs.get('default_top_k', 10)
        self.max_top_k = kwargs.get('max_top_k', 100)
        self.min_similarity_score = kwargs.get('min_similarity_score', 0.0)
        
        self.vector_db = vector_db
        self.embedding_manager = embedding_manager
        
        super().__init__(**kwargs)
        
        self.logger = get_logger(self.__class__.__name__)
    
    @property
    def name(self) -> str:
        """Tool name identifier."""
        return "rag_query"
    
    @property
    def description(self) -> str:
        """Tool description for users and LLMs."""
        return (
            "Perform semantic search and context retrieval from document corpora. "
            "This tool finds the most relevant documents for a given query using "
            "vector similarity search and returns ranked results with metadata. "
            "Essential for RAG (Retrieval-Augmented Generation) workflows."
        )
    
    @property
    def version(self) -> str:
        """Tool version for compatibility tracking."""
        return "1.0.0"
    
    def _setup_parameters(self) -> None:
        """Set up tool parameters."""
        self._parameters = [
            ToolParameter(
                name="query",
                type=ParameterType.STRING,
                description="The search query or question to find relevant documents for",
                required=True
            ),
            ToolParameter(
                name="corpus_name",
                type=ParameterType.STRING,
                description="Name of the corpus to search in",
                required=True
            ),
            ToolParameter(
                name="top_k",
                type=ParameterType.INTEGER,
                description="Maximum number of results to return",
                required=False,
                default=self.default_top_k,
                minimum=1,
                maximum=self.max_top_k
            ),
            ToolParameter(
                name="min_score",
                type=ParameterType.NUMBER,
                description="Minimum similarity score threshold (0.0 to 1.0)",
                required=False,
                default=self.min_similarity_score,
                minimum=0.0,
                maximum=1.0
            ),
            ToolParameter(
                name="include_metadata",
                type=ParameterType.BOOLEAN,
                description="Whether to include document metadata in results",
                required=False,
                default=True
            ),
            ToolParameter(
                name="include_embeddings",
                type=ParameterType.BOOLEAN,
                description="Whether to include document embeddings in results",
                required=False,
                default=False
            ),
            ToolParameter(
                name="metadata_filters",
                type=ParameterType.OBJECT,
                description="Optional metadata filters to apply (key-value pairs)",
                required=False,
                default={}
            ),
            ToolParameter(
                name="rerank_results",
                type=ParameterType.BOOLEAN,
                description="Whether to apply additional reranking to results",
                required=False,
                default=False
            ),
            ToolParameter(
                name="context_window",
                type=ParameterType.INTEGER,
                description="Number of characters of context to include around matches",
                required=False,
                minimum=0,
                maximum=5000
            )
        ]
    
    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute RAG query search.
        
        Args:
            **kwargs: Validated parameters for RAG query
            
        Returns:
            Dictionary containing search results and metadata
            
        Raises:
            LexoraError: If query execution fails
        """
        query = kwargs["query"]
        corpus_name = kwargs["corpus_name"]
        top_k = kwargs.get("top_k", self.default_top_k)
        min_score = kwargs.get("min_score", self.min_similarity_score)
        include_metadata = kwargs.get("include_metadata", True)
        include_embeddings = kwargs.get("include_embeddings", False)
        metadata_filters = kwargs.get("metadata_filters", {})
        rerank_results = kwargs.get("rerank_results", False)
        context_window = kwargs.get("context_window")
        
        try:
            # Validate query
            if not query or not query.strip():
                raise create_tool_error(
                    "Query cannot be empty",
                    self.name,
                    {"query": query},
                    ErrorCode.TOOL_INVALID_PARAMETERS
                )
            
            query = query.strip()
            
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
            
            self.logger.info(
                f"Executing RAG query: '{query[:50]}...' in corpus '{corpus_name}' (top_k={top_k})"
            )
            
            # Perform vector search
            search_start_time = datetime.utcnow()
            
            # Use the vector database's search method with retry logic if available
            if hasattr(self.vector_db, 'search_with_retry'):
                search_results = await self.vector_db.search_with_retry(
                    corpus_name=corpus_name,
                    query=query,
                    top_k=top_k * 2 if rerank_results else top_k,  # Get more results if reranking
                    max_retries=2
                )
            else:
                search_results = await self.vector_db.search(
                    corpus_name=corpus_name,
                    query=query,
                    top_k=top_k * 2 if rerank_results else top_k
                )
            
            search_duration = (datetime.utcnow() - search_start_time).total_seconds()
            
            # Apply minimum score filtering
            if min_score > 0.0:
                search_results = [r for r in search_results if r.score >= min_score]
            
            # Apply metadata filters if provided
            if metadata_filters:
                filtered_results = []
                for result in search_results:
                    doc_metadata = result.document.metadata
                    matches_filters = True
                    
                    for filter_key, filter_value in metadata_filters.items():
                        if filter_key not in doc_metadata:
                            matches_filters = False
                            break
                        
                        doc_value = doc_metadata[filter_key]
                        if isinstance(filter_value, list):
                            # Check if document value is in the filter list
                            if doc_value not in filter_value:
                                matches_filters = False
                                break
                        else:
                            # Exact match
                            if doc_value != filter_value:
                                matches_filters = False
                                break
                    
                    if matches_filters:
                        filtered_results.append(result)
                
                search_results = filtered_results
            
            # Apply reranking if requested
            if rerank_results and len(search_results) > 1:
                search_results = await self._rerank_results(query, search_results)
            
            # Limit to requested top_k
            search_results = search_results[:top_k]
            
            # Process results for output
            processed_results = []
            total_context_length = 0
            
            for i, result in enumerate(search_results):
                doc = result.document
                
                # Prepare document data
                doc_data = {
                    "document_id": doc.id,
                    "content": doc.content,
                    "score": result.score,
                    "rank": i + 1,
                    "corpus_name": result.corpus_name
                }
                
                # Add metadata if requested
                if include_metadata:
                    doc_data["metadata"] = doc.metadata
                
                # Add embeddings if requested
                if include_embeddings and doc.embedding:
                    doc_data["embedding"] = doc.embedding
                
                # Add context window if specified
                if context_window and context_window > 0:
                    content_length = len(doc.content)
                    if content_length > context_window:
                        # Find the best context window around query terms
                        context = self._extract_context_window(
                            doc.content, query, context_window
                        )
                        doc_data["context"] = context
                        doc_data["context_info"] = {
                            "window_size": len(context),
                            "full_content_length": content_length,
                            "truncated": True
                        }
                    else:
                        doc_data["context"] = doc.content
                        doc_data["context_info"] = {
                            "window_size": content_length,
                            "full_content_length": content_length,
                            "truncated": False
                        }
                
                total_context_length += len(doc.content)
                processed_results.append(doc_data)
            
            # Calculate query statistics
            query_stats = {
                "query_length": len(query),
                "search_duration_seconds": search_duration,
                "total_results_found": len(search_results),
                "results_returned": len(processed_results),
                "min_score_applied": min_score,
                "metadata_filters_applied": len(metadata_filters) > 0,
                "reranking_applied": rerank_results,
                "total_context_length": total_context_length,
                "embedding_model": self.embedding_manager.get_model_name()
            }
            
            # Get corpus information for additional context
            try:
                corpus_info = await self.vector_db.get_corpus_info(corpus_name)
                corpus_stats = {
                    "corpus_name": corpus_info.name,
                    "total_documents": corpus_info.document_count,
                    "corpus_created_at": corpus_info.created_at.isoformat()
                }
            except Exception as e:
                self.logger.warning(f"Could not retrieve corpus info: {e}")
                corpus_stats = {
                    "corpus_name": corpus_name,
                    "total_documents": None,
                    "corpus_created_at": None
                }
            
            # Return structured response
            return {
                "query": query,
                "results": processed_results,
                "query_stats": query_stats,
                "corpus_stats": corpus_stats,
                "search_metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "tool_version": self.version,
                    "parameters_used": {
                        "top_k": top_k,
                        "min_score": min_score,
                        "include_metadata": include_metadata,
                        "include_embeddings": include_embeddings,
                        "metadata_filters": metadata_filters,
                        "rerank_results": rerank_results,
                        "context_window": context_window
                    }
                },
                "message": f"Found {len(processed_results)} relevant documents for query in corpus '{corpus_name}'"
            }
            
        except LexoraError:
            # Re-raise LexoraErrors as-is
            raise
            
        except Exception as e:
            # Wrap other exceptions
            raise create_tool_error(
                f"Unexpected error executing RAG query: {str(e)}",
                self.name,
                {"query": query, "corpus_name": corpus_name, "error_type": type(e).__name__},
                ErrorCode.TOOL_EXECUTION_FAILED,
                e
            )
    
    async def _rerank_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """
        Apply additional reranking to search results.
        
        This is a simple implementation that can be extended with more sophisticated
        reranking algorithms like cross-encoders or other ML models.
        
        Args:
            query: Original search query
            results: Initial search results to rerank
            
        Returns:
            Reranked search results
        """
        # Simple reranking based on query term overlap and content length
        query_terms = set(query.lower().split())
        
        def calculate_rerank_score(result: SearchResult) -> float:
            content = result.document.content.lower()
            
            # Base score from vector similarity
            base_score = result.score
            
            # Bonus for query term matches
            term_matches = sum(1 for term in query_terms if term in content)
            term_bonus = (term_matches / len(query_terms)) * 0.1 if query_terms else 0
            
            # Slight penalty for very short or very long documents
            content_length = len(result.document.content)
            if content_length < 50:
                length_penalty = 0.05
            elif content_length > 2000:
                length_penalty = 0.02
            else:
                length_penalty = 0
            
            # Bonus for documents with relevant metadata
            metadata_bonus = 0
            if "topic" in result.document.metadata:
                topic = result.document.metadata["topic"].lower()
                if any(term in topic for term in query_terms):
                    metadata_bonus = 0.05
            
            return base_score + term_bonus + metadata_bonus - length_penalty
        
        # Rerank results
        reranked_results = sorted(results, key=calculate_rerank_score, reverse=True)
        
        # Update scores to reflect reranking
        for i, result in enumerate(reranked_results):
            result.score = calculate_rerank_score(result)
        
        return reranked_results
    
    def _extract_context_window(self, content: str, query: str, window_size: int) -> str:
        """
        Extract a context window around query terms in the content.
        
        Args:
            content: Full document content
            query: Search query
            window_size: Maximum size of context window
            
        Returns:
            Context window string
        """
        if len(content) <= window_size:
            return content
        
        # Find the best position for the context window
        query_terms = query.lower().split()
        content_lower = content.lower()
        
        # Find positions of query terms
        term_positions = []
        for term in query_terms:
            pos = content_lower.find(term)
            if pos != -1:
                term_positions.append(pos)
        
        if not term_positions:
            # No query terms found, return beginning of content
            return content[:window_size] + "..." if len(content) > window_size else content
        
        # Find the center position that captures most query terms
        center_pos = sum(term_positions) // len(term_positions)
        
        # Calculate window boundaries
        half_window = window_size // 2
        start_pos = max(0, center_pos - half_window)
        end_pos = min(len(content), start_pos + window_size)
        
        # Adjust start position if we're at the end
        if end_pos == len(content):
            start_pos = max(0, end_pos - window_size)
        
        # Extract context
        context = content[start_pos:end_pos]
        
        # Add ellipsis if truncated
        if start_pos > 0:
            context = "..." + context
        if end_pos < len(content):
            context = context + "..."
        
        return context


# Convenience function for creating the tool
def create_rag_query_tool(
    vector_db: BaseVectorDB,
    embedding_manager: EmbeddingManager,
    **kwargs
) -> RAGQueryTool:
    """
    Create a RAGQueryTool instance.
    
    Args:
        vector_db: Vector database instance
        embedding_manager: Embedding manager instance
        **kwargs: Additional configuration options
        
    Returns:
        Configured RAGQueryTool instance
    """
    return RAGQueryTool(
        vector_db=vector_db,
        embedding_manager=embedding_manager,
        **kwargs
    )