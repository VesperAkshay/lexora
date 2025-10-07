"""
Health Check Tool for the Lexora Agentic RAG SDK.

This tool provides comprehensive system health monitoring for all components
including vector databases, embedding services, and overall system status.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from .base_tool import BaseTool, ToolParameter, ParameterType
from ..vector_db.base_vector_db import BaseVectorDB
from ..utils.embeddings import EmbeddingManager
from ..exceptions import LexoraError, ErrorCode, create_tool_error
from ..utils.logging import get_logger


class HealthCheckTool(BaseTool):
    """
    Tool for comprehensive system health monitoring.
    
    This tool checks the health and availability of all system components
    including vector databases, embedding services, and provides detailed
    diagnostics and performance metrics.
    """
    
    def __init__(
        self,
        vector_db: Optional[BaseVectorDB] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        **kwargs
    ):
        """
        Initialize the health check tool.
        
        Args:
            vector_db: Optional vector database instance to check
            embedding_manager: Optional embedding manager instance to check
            **kwargs: Additional configuration options
        """
        super().__init__(**kwargs)
        
        self.vector_db = vector_db
        self.embedding_manager = embedding_manager
        self.logger = get_logger(self.__class__.__name__)
        
        # Configuration
        self.timeout_seconds = kwargs.get('timeout_seconds', 30.0)
        self.include_performance_tests = kwargs.get('include_performance_tests', False)
        self.test_corpus_name = kwargs.get('test_corpus_name', '_health_check_test')
    
    @property
    def name(self) -> str:
        """Tool name identifier."""
        return "health_check"
    
    @property
    def description(self) -> str:
        """Tool description for users and LLMs."""
        return (
            "Perform comprehensive health checks on all system components including "
            "vector databases, embedding services, and overall system status. "
            "Provides detailed diagnostics, performance metrics, and recommendations "
            "for system optimization and troubleshooting."
        )
    
    @property
    def version(self) -> str:
        """Tool version for compatibility tracking."""
        return "1.0.0"
    
    def _setup_parameters(self) -> None:
        """Set up tool parameters."""
        self._parameters = [
            ToolParameter(
                name="check_vector_db",
                type=ParameterType.BOOLEAN,
                description="Whether to check vector database health",
                required=False,
                default=True
            ),
            ToolParameter(
                name="check_embeddings",
                type=ParameterType.BOOLEAN,
                description="Whether to check embedding service health",
                required=False,
                default=True
            ),
            ToolParameter(
                name="include_performance_tests",
                type=ParameterType.BOOLEAN,
                description="Whether to include performance benchmarking tests",
                required=False,
                default=False
            ),
            ToolParameter(
                name="detailed_diagnostics",
                type=ParameterType.BOOLEAN,
                description="Whether to include detailed diagnostic information",
                required=False,
                default=True
            ),
            ToolParameter(
                name="timeout_seconds",
                type=ParameterType.NUMBER,
                description="Timeout for health check operations in seconds",
                required=False,
                default=30.0,
                minimum=1.0,
                maximum=300.0
            ),
            ToolParameter(
                name="test_operations",
                type=ParameterType.BOOLEAN,
                description="Whether to perform actual test operations (may create temporary data)",
                required=False,
                default=False
            )
        ]
    
    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute comprehensive health check.
        
        Args:
            **kwargs: Validated parameters for health check
            
        Returns:
            Dictionary containing health check results and diagnostics
            
        Raises:
            LexoraError: If health check execution fails
        """
        check_vector_db = kwargs.get("check_vector_db", True)
        check_embeddings = kwargs.get("check_embeddings", True)
        include_performance_tests = kwargs.get("include_performance_tests", False)
        detailed_diagnostics = kwargs.get("detailed_diagnostics", True)
        timeout_seconds = kwargs.get("timeout_seconds", 30.0)
        test_operations = kwargs.get("test_operations", False)
        
        start_time = datetime.utcnow()
        
        try:
            self.logger.info("Starting comprehensive system health check")
            
            # Initialize results structure
            health_results = {
                "overall_status": "unknown",
                "timestamp": start_time.isoformat(),
                "components": {},
                "summary": {
                    "total_components": 0,
                    "healthy_components": 0,
                    "unhealthy_components": 0,
                    "warning_components": 0
                }
            }
            
            # Check vector database health
            if check_vector_db and self.vector_db:
                try:
                    vector_db_health = await asyncio.wait_for(
                        self._check_vector_db_health(test_operations, detailed_diagnostics),
                        timeout=timeout_seconds
                    )
                    health_results["components"]["vector_database"] = vector_db_health
                except asyncio.TimeoutError:
                    health_results["components"]["vector_database"] = {
                        "status": "unhealthy",
                        "error": f"Health check timed out after {timeout_seconds}s",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                except Exception as e:
                    health_results["components"]["vector_database"] = {
                        "status": "unhealthy",
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    }
            elif check_vector_db:
                health_results["components"]["vector_database"] = {
                    "status": "not_configured",
                    "message": "Vector database not provided to health check tool",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Check embedding service health
            if check_embeddings and self.embedding_manager:
                try:
                    embeddings_health = await asyncio.wait_for(
                        self._check_embeddings_health(test_operations, detailed_diagnostics),
                        timeout=timeout_seconds
                    )
                    health_results["components"]["embedding_service"] = embeddings_health
                except asyncio.TimeoutError:
                    health_results["components"]["embedding_service"] = {
                        "status": "unhealthy",
                        "error": f"Health check timed out after {timeout_seconds}s",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                except Exception as e:
                    health_results["components"]["embedding_service"] = {
                        "status": "unhealthy",
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    }
            elif check_embeddings:
                health_results["components"]["embedding_service"] = {
                    "status": "not_configured",
                    "message": "Embedding manager not provided to health check tool",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Check system-level health
            system_health = await self._check_system_health(detailed_diagnostics)
            health_results["components"]["system"] = system_health
            
            # Perform performance tests if requested
            if include_performance_tests:
                try:
                    performance_results = await asyncio.wait_for(
                        self._run_performance_tests(),
                        timeout=timeout_seconds * 2  # Allow more time for performance tests
                    )
                    health_results["performance_tests"] = performance_results
                except asyncio.TimeoutError:
                    health_results["performance_tests"] = {
                        "status": "timeout",
                        "error": f"Performance tests timed out after {timeout_seconds * 2}s"
                    }
                except Exception as e:
                    health_results["performance_tests"] = {
                        "status": "failed",
                        "error": str(e)
                    }
            
            # Calculate summary statistics
            total_components = len(health_results["components"])
            healthy_count = 0
            unhealthy_count = 0
            warning_count = 0
            
            for component_name, component_health in health_results["components"].items():
                status = component_health.get("status", "unknown")
                if status == "healthy":
                    healthy_count += 1
                elif status in ["unhealthy", "failed", "error"]:
                    unhealthy_count += 1
                elif status in ["warning", "degraded", "not_configured"]:
                    warning_count += 1
            
            health_results["summary"] = {
                "total_components": total_components,
                "healthy_components": healthy_count,
                "unhealthy_components": unhealthy_count,
                "warning_components": warning_count
            }
            
            # Determine overall status
            if unhealthy_count > 0:
                overall_status = "unhealthy"
            elif warning_count > 0:
                overall_status = "warning"
            elif healthy_count > 0:
                overall_status = "healthy"
            else:
                overall_status = "unknown"
            
            health_results["overall_status"] = overall_status
            
            # Add execution metadata
            end_time = datetime.utcnow()
            total_duration = (end_time - start_time).total_seconds()
            
            health_results["execution_info"] = {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": total_duration,
                "tool_version": self.version,
                "parameters_used": {
                    "check_vector_db": check_vector_db,
                    "check_embeddings": check_embeddings,
                    "include_performance_tests": include_performance_tests,
                    "detailed_diagnostics": detailed_diagnostics,
                    "timeout_seconds": timeout_seconds,
                    "test_operations": test_operations
                }
            }
            
            # Generate recommendations
            recommendations = self._generate_recommendations(health_results)
            if recommendations:
                health_results["recommendations"] = recommendations
            
            # Generate appropriate message
            if overall_status == "healthy":
                health_results["message"] = f"All {healthy_count} components are healthy"
            elif overall_status == "warning":
                health_results["message"] = f"System operational with {warning_count} warnings ({healthy_count} healthy, {unhealthy_count} unhealthy)"
            else:
                health_results["message"] = f"System has {unhealthy_count} unhealthy components ({healthy_count} healthy, {warning_count} warnings)"
            
            return health_results
            
        except Exception as e:
            # Wrap other exceptions
            raise create_tool_error(
                f"Unexpected error during health check: {str(e)}",
                self.name,
                {"error_type": type(e).__name__},
                ErrorCode.TOOL_EXECUTION_FAILED,
                e
            )
    
    async def _check_vector_db_health(self, test_operations: bool, detailed: bool) -> Dict[str, Any]:
        """
        Check vector database health.
        
        Args:
            test_operations: Whether to perform test operations
            detailed: Whether to include detailed diagnostics
            
        Returns:
            Dictionary containing vector database health information
        """
        start_time = time.time()
        
        try:
            # Basic connectivity check
            if not self.vector_db.is_connected():
                await self.vector_db.connect()
            
            # Get basic info
            provider_name = self.vector_db.get_provider_name()
            
            # Test basic operations
            corpora = await self.vector_db.list_corpora()
            
            # Perform test operations if requested
            test_results = {}
            if test_operations:
                test_results = await self._perform_vector_db_tests()
            
            duration = time.time() - start_time
            
            result = {
                "status": "healthy",
                "provider": provider_name,
                "connected": self.vector_db.is_connected(),
                "response_time_seconds": duration,
                "corpora_count": len(corpora),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if detailed:
                result["corpora_list"] = corpora[:10]  # Limit list size
                if len(corpora) > 10:
                    result["additional_corpora_count"] = len(corpora) - 10
            
            if test_results:
                result["test_operations"] = test_results
            
            return result
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.vector_db.__class__.__name__ if self.vector_db else "unknown",
                "connected": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }    
    async def _check_embeddings_health(self, test_operations: bool, detailed: bool) -> Dict[str, Any]:
        """
        Check embedding service health.
        
        Args:
            test_operations: Whether to perform test operations
            detailed: Whether to include detailed diagnostics
            
        Returns:
            Dictionary containing embedding service health information
        """
        start_time = time.time()
        
        try:
            # Get basic info
            model_name = self.embedding_manager.get_model_name()
            dimension = self.embedding_manager.get_dimension()
            
            # Test embedding generation if requested
            test_results = {}
            if test_operations:
                test_embedding = await self.embedding_manager.generate_embedding("health check test")
                test_results = {
                    "test_embedding_generated": True,
                    "test_embedding_dimension": len(test_embedding) if test_embedding else 0,
                    "dimension_matches": len(test_embedding) == dimension if test_embedding else False
                }
            
            duration = time.time() - start_time
            
            result = {
                "status": "healthy",
                "model_name": model_name,
                "embedding_dimension": dimension,
                "response_time_seconds": duration,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if detailed:
                result["cache_size"] = self.embedding_manager.get_cache_size()
            
            if test_results:
                result["test_operations"] = test_results
            
            return result
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "model_name": getattr(getattr(self.embedding_manager, 'provider', None), 'model_name', 'unknown') if self.embedding_manager else 'unknown',
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _check_system_health(self, detailed: bool) -> Dict[str, Any]:
        """
        Check system-level health.
        
        Args:
            detailed: Whether to include detailed diagnostics
            
        Returns:
            Dictionary containing system health information
        """
        try:
            import psutil
            import sys
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            result = {
                "status": "healthy",
                "python_version": sys.version,
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "disk_usage_percent": disk.percent,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if detailed:
                result["memory_total_gb"] = round(memory.total / (1024**3), 2)
                result["memory_available_gb"] = round(memory.available / (1024**3), 2)
                result["disk_total_gb"] = round(disk.total / (1024**3), 2)
                result["disk_free_gb"] = round(disk.free / (1024**3), 2)
            
            # Check for warning conditions
            if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
                result["status"] = "warning"
                result["warnings"] = []
                if cpu_percent > 90:
                    result["warnings"].append(f"High CPU usage: {cpu_percent}%")
                if memory.percent > 90:
                    result["warnings"].append(f"High memory usage: {memory.percent}%")
                if disk.percent > 90:
                    result["warnings"].append(f"High disk usage: {disk.percent}%")
            
            return result
            
        except ImportError:
            return {
                "status": "warning",
                "message": "psutil not available for system monitoring",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "warning",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _perform_vector_db_tests(self) -> Dict[str, Any]:
        """
        Perform test operations on vector database.
        
        Returns:
            Dictionary containing test results
        """
        test_results = {}
        
        try:
            # Test corpus creation
            test_corpus_created = False
            try:
                await self.vector_db.create_corpus(self.test_corpus_name)
                test_corpus_created = True
                test_results["corpus_creation"] = "success"
            except Exception as e:
                test_results["corpus_creation"] = f"failed: {str(e)}"
            
            # Test document addition if corpus was created
            if test_corpus_created:
                try:
                    from models.core import Document
                    test_doc = Document(
                        id="health_check_doc",
                        content="This is a health check test document",
                        metadata={"test": True, "timestamp": datetime.utcnow().isoformat()}
                    )
                    await self.vector_db.add_documents(self.test_corpus_name, [test_doc])
                    test_results["document_addition"] = "success"
                except Exception as e:
                    test_results["document_addition"] = f"failed: {str(e)}"
                
                # Test search
                try:
                    search_results = await self.vector_db.search(
                        self.test_corpus_name, "health check", top_k=1
                    )
                    test_results["search_operation"] = f"success: {len(search_results)} results"
                except Exception as e:
                    test_results["search_operation"] = f"failed: {str(e)}"
                
                # Clean up test corpus
                try:
                    await self.vector_db.delete_corpus(self.test_corpus_name)
                    test_results["cleanup"] = "success"
                except Exception as e:
                    test_results["cleanup"] = f"failed: {str(e)}"
            
            return test_results
            
        except Exception as e:
            return {"test_operations_error": str(e)}
    
    async def _run_performance_tests(self) -> Dict[str, Any]:
        """
        Run performance benchmarking tests.
        
        Returns:
            Dictionary containing performance test results
        """
        performance_results = {}
        
        try:
            # Test embedding generation performance
            if self.embedding_manager:
                start_time = time.time()
                test_texts = [
                    "This is a performance test for embedding generation",
                    "Another test text for benchmarking purposes",
                    "Third test text to measure embedding performance"
                ]
                
                embeddings = await self.embedding_manager.generate_embeddings_batch(test_texts)
                duration = time.time() - start_time
                
                performance_results["embedding_generation"] = {
                    "texts_processed": len(test_texts),
                    "duration_seconds": duration,
                    "texts_per_second": len(test_texts) / duration if duration > 0 else 0,
                    "embeddings_generated": len(embeddings)
                }
            
            # Test vector database performance
            if self.vector_db:
                start_time = time.time()
                corpora = await self.vector_db.list_corpora()
                list_duration = time.time() - start_time
                
                performance_results["vector_db_operations"] = {
                    "list_corpora_duration": list_duration,
                    "corpora_count": len(corpora)
                }
            
            return performance_results
            
        except Exception as e:
            return {"performance_test_error": str(e)}
    
    def _generate_recommendations(self, health_results: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on health check results.
        
        Args:
            health_results: Health check results
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Check overall status
        if health_results["overall_status"] == "unhealthy":
            recommendations.append("System has unhealthy components that require immediate attention")
        
        # Check individual components
        components = health_results.get("components", {})
        
        # Vector database recommendations
        vector_db = components.get("vector_database", {})
        if vector_db.get("status") == "unhealthy":
            recommendations.append("Vector database is unhealthy - check connectivity and configuration")
        elif vector_db.get("status") == "not_configured":
            recommendations.append("Consider configuring vector database for full RAG functionality")
        
        # Embedding service recommendations
        embeddings = components.get("embedding_service", {})
        if embeddings.get("status") == "unhealthy":
            recommendations.append("Embedding service is unhealthy - check API keys and network connectivity")
        elif embeddings.get("status") == "not_configured":
            recommendations.append("Consider configuring embedding service for document processing")
        
        # System recommendations
        system = components.get("system", {})
        if system.get("status") == "warning":
            warnings = system.get("warnings", [])
            for warning in warnings:
                recommendations.append(f"System warning: {warning}")
        
        # Performance recommendations
        performance = health_results.get("performance_tests", {})
        if performance:
            embedding_perf = performance.get("embedding_generation", {})
            if embedding_perf.get("texts_per_second", 0) < 1:
                recommendations.append("Embedding generation performance is slow - consider optimizing or upgrading")
        
        return recommendations


# Convenience function for creating the tool
def create_health_check_tool(
    vector_db: Optional[BaseVectorDB] = None,
    embedding_manager: Optional[EmbeddingManager] = None,
    **kwargs
) -> HealthCheckTool:
    """
    Create a HealthCheckTool instance.
    
    Args:
        vector_db: Optional vector database instance
        embedding_manager: Optional embedding manager instance
        **kwargs: Additional configuration options
        
    Returns:
        Configured HealthCheckTool instance
    """
    return HealthCheckTool(
        vector_db=vector_db,
        embedding_manager=embedding_manager,
        **kwargs
    )