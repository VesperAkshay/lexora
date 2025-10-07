"""
Exception handling system for the Lexora Agentic RAG SDK.

This module provides a comprehensive exception hierarchy with structured error information,
error codes, and context for debugging and monitoring system health.
"""

from typing import Any, Dict, List, Optional
from enum import Enum


class ErrorCode(Enum):
    """
    Enumeration of error codes for structured error handling.
    
    Error codes are organized by category to help with debugging and monitoring.
    """
    
    # Configuration Errors (1000-1099)
    INVALID_CONFIG = "LEXORA_1001"
    MISSING_CONFIG = "LEXORA_1002"
    INVALID_MODEL_CONFIG = "LEXORA_1003"
    INVALID_VECTOR_DB_CONFIG = "LEXORA_1004"
    INVALID_AGENT_CONFIG = "LEXORA_1005"
    
    # LLM Errors (2000-2099)
    LLM_CONNECTION_FAILED = "LEXORA_2001"
    LLM_AUTHENTICATION_FAILED = "LEXORA_2002"
    LLM_RATE_LIMIT_EXCEEDED = "LEXORA_2003"
    LLM_INVALID_RESPONSE = "LEXORA_2004"
    LLM_TIMEOUT = "LEXORA_2005"
    LLM_MODEL_NOT_FOUND = "LEXORA_2006"
    LLM_QUOTA_EXCEEDED = "LEXORA_2007"
    
    # Vector Database Errors (3000-3099)
    VECTOR_DB_CONNECTION_FAILED = "LEXORA_3001"
    VECTOR_DB_AUTHENTICATION_FAILED = "LEXORA_3002"
    VECTOR_DB_INDEX_NOT_FOUND = "LEXORA_3003"
    VECTOR_DB_CORPUS_NOT_FOUND = "LEXORA_3004"
    VECTOR_DB_DOCUMENT_NOT_FOUND = "LEXORA_3005"
    VECTOR_DB_INVALID_DIMENSION = "LEXORA_3006"
    VECTOR_DB_STORAGE_FULL = "LEXORA_3007"
    VECTOR_DB_OPERATION_TIMEOUT = "LEXORA_3008"
    VECTOR_DB_CORPUS_ALREADY_EXISTS = "LEXORA_3009"
    
    # Tool Execution Errors (4000-4099)
    TOOL_NOT_FOUND = "LEXORA_4001"
    TOOL_INVALID_PARAMETERS = "LEXORA_4002"
    TOOL_EXECUTION_FAILED = "LEXORA_4003"
    TOOL_TIMEOUT = "LEXORA_4004"
    TOOL_DEPENDENCY_FAILED = "LEXORA_4005"
    TOOL_VALIDATION_FAILED = "LEXORA_4006"
    
    # Planning Errors (5000-5099)
    PLANNING_FAILED = "LEXORA_5001"
    INVALID_QUERY = "LEXORA_5002"
    NO_SUITABLE_TOOLS = "LEXORA_5003"
    CIRCULAR_DEPENDENCY = "LEXORA_5004"
    MAX_STEPS_EXCEEDED = "LEXORA_5005"
    
    # General System Errors (9000-9099)
    UNKNOWN_ERROR = "LEXORA_9001"
    INTERNAL_ERROR = "LEXORA_9002"
    RESOURCE_EXHAUSTED = "LEXORA_9003"
    OPERATION_CANCELLED = "LEXORA_9004"


class LexoraError(Exception):
    """
    Base exception for Lexora SDK.
    
    This is the root exception class that provides structured error information
    including error codes, context, and suggested solutions for debugging.
    
    Attributes:
        message: Human-readable error message
        error_code: Structured error code for programmatic handling
        context: Additional context information about the error
        suggestions: List of suggested solutions or next steps
        original_error: The original exception that caused this error (if any)
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[ErrorCode] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize a LexoraError.
        
        Args:
            message: Human-readable error message
            error_code: Structured error code for programmatic handling
            context: Additional context information about the error
            suggestions: List of suggested solutions or next steps
            original_error: The original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or ErrorCode.UNKNOWN_ERROR
        self.context = context or {}
        self.suggestions = suggestions or []
        self.original_error = original_error
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the error to a dictionary representation.
        
        Returns:
            Dictionary representation of the error with all structured information
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code.value,
            "context": self.context,
            "suggestions": self.suggestions,
            "original_error": str(self.original_error) if self.original_error else None
        }
    
    def __str__(self) -> str:
        """Return a formatted string representation of the error."""
        parts = [f"{self.error_code.value}: {self.message}"]
        
        if self.context:
            parts.append(f"Context: {self.context}")
        
        if self.suggestions:
            parts.append(f"Suggestions: {', '.join(self.suggestions)}")
        
        if self.original_error:
            parts.append(f"Original error: {self.original_error}")
        
        return " | ".join(parts)


class ConfigurationError(LexoraError):
    """
    Raised when configuration is invalid.
    
    This exception is raised when there are issues with SDK configuration,
    including invalid parameters, missing required settings, or incompatible options.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[ErrorCode] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
        original_error: Optional[Exception] = None
    ):
        """Initialize a ConfigurationError with default suggestions."""
        default_suggestions = [
            "Check your configuration parameters",
            "Verify all required settings are provided",
            "Consult the documentation for valid configuration options"
        ]
        
        super().__init__(
            message=message,
            error_code=error_code or ErrorCode.INVALID_CONFIG,
            context=context,
            suggestions=default_suggestions if suggestions is None else suggestions,
            original_error=original_error
        )

class LLMError(LexoraError):
    """
    Raised when LLM operations fail.
    
    This exception covers all LLM-related failures including connection issues,
    authentication problems, rate limiting, and invalid responses.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[ErrorCode] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
        original_error: Optional[Exception] = None
    ):
        """Initialize an LLMError with default suggestions."""
        default_suggestions = [
            "Check your LLM provider configuration",
            "Verify API keys and authentication",
            "Check rate limits and quotas",
            "Try again with exponential backoff"
        ]
        
        super().__init__(
            message=message,
            error_code=error_code or ErrorCode.LLM_CONNECTION_FAILED,
            context=context,
            suggestions=suggestions or default_suggestions,
            original_error=original_error
        )


class VectorDBError(LexoraError):
    """
    Raised when vector database operations fail.
    
    This exception covers all vector database-related failures including
    connection issues, missing indices/corpora, and storage problems.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[ErrorCode] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
        original_error: Optional[Exception] = None
    ):
        """Initialize a VectorDBError with default suggestions."""
        default_suggestions = [
            "Check your vector database configuration",
            "Verify the database connection and credentials",
            "Ensure the corpus/index exists",
            "Check available storage space"
        ]
        
        super().__init__(
            message=message,
            error_code=error_code or ErrorCode.VECTOR_DB_CONNECTION_FAILED,
            context=context,
            suggestions=suggestions or default_suggestions,
            original_error=original_error
        )


class ToolExecutionError(LexoraError):
    """
    Raised when tool execution fails.
    
    This exception covers all tool-related failures including missing tools,
    invalid parameters, execution timeouts, and dependency failures.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[ErrorCode] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
        original_error: Optional[Exception] = None
    ):
        """Initialize a ToolExecutionError with default suggestions."""
        default_suggestions = [
            "Check that the tool is properly registered",
            "Verify tool parameters are valid",
            "Check tool dependencies are satisfied",
            "Review tool execution logs for details"
        ]
        
        super().__init__(
            message=message,
            error_code=error_code or ErrorCode.TOOL_EXECUTION_FAILED,
            context=context,
            suggestions=suggestions or default_suggestions,
            original_error=original_error
        )


class PlanningError(LexoraError):
    """
    Raised when query planning fails.
    
    This exception covers all planning-related failures including invalid queries,
    missing suitable tools, circular dependencies, and complexity limits.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[ErrorCode] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
        original_error: Optional[Exception] = None
    ):
        """Initialize a PlanningError with default suggestions."""
        default_suggestions = [
            "Simplify your query or break it into smaller parts",
            "Check that required tools are available",
            "Verify query parameters are valid",
            "Review tool dependencies for circular references"
        ]
        
        super().__init__(
            message=message,
            error_code=error_code or ErrorCode.PLANNING_FAILED,
            context=context,
            suggestions=suggestions or default_suggestions,
            original_error=original_error
        )


# Convenience functions for creating common errors

def create_configuration_error(
    message: str,
    config_type: str,
    invalid_field: Optional[str] = None,
    original_error: Optional[Exception] = None
) -> ConfigurationError:
    """
    Create a ConfigurationError with structured context.
    
    Args:
        message: Error message
        config_type: Type of configuration that failed (e.g., "LLMConfig")
        invalid_field: Specific field that was invalid
        original_error: Original exception if any
        
    Returns:
        ConfigurationError with structured context
    """
    context = {"config_type": config_type}
    if invalid_field:
        context["invalid_field"] = invalid_field
    
    return ConfigurationError(
        message=message,
        context=context,
        original_error=original_error
    )


def create_llm_error(
    message: str,
    provider: str,
    model: Optional[str] = None,
    error_code: Optional[ErrorCode] = None,
    original_error: Optional[Exception] = None
) -> LLMError:
    """
    Create an LLMError with structured context.
    
    Args:
        message: Error message
        provider: LLM provider name
        model: Model name if applicable
        error_code: Specific error code
        original_error: Original exception if any
        
    Returns:
        LLMError with structured context
    """
    context = {"provider": provider}
    if model:
        context["model"] = model
    
    return LLMError(
        message=message,
        error_code=error_code,
        context=context,
        original_error=original_error
    )


def create_vector_db_error(
    message: str,
    provider: str,
    corpus_name: Optional[str] = None,
    error_code: Optional[ErrorCode] = None,
    original_error: Optional[Exception] = None
) -> VectorDBError:
    """
    Create a VectorDBError with structured context.
    
    Args:
        message: Error message
        provider: Vector database provider name
        corpus_name: Corpus name if applicable
        error_code: Specific error code
        original_error: Original exception if any
        
    Returns:
        VectorDBError with structured context
    """
    context = {"provider": provider}
    if corpus_name:
        context["corpus_name"] = corpus_name
    
    return VectorDBError(
        message=message,
        error_code=error_code,
        context=context,
        original_error=original_error
    )


def create_tool_error(
    message: str,
    tool_name: str,
    parameters: Optional[Dict[str, Any]] = None,
    error_code: Optional[ErrorCode] = None,
    original_error: Optional[Exception] = None
) -> ToolExecutionError:
    """
    Create a ToolExecutionError with structured context.
    
    Args:
        message: Error message
        tool_name: Name of the tool that failed
        parameters: Tool parameters if applicable
        error_code: Specific error code
        original_error: Original exception if any
        
    Returns:
        ToolExecutionError with structured context
    """
    context = {"tool_name": tool_name}
    if parameters:
        context["parameters"] = parameters
    
    return ToolExecutionError(
        message=message,
        error_code=error_code,
        context=context,
        original_error=original_error
    )


def create_planning_error(
    message: str,
    query: str,
    available_tools: Optional[List[str]] = None,
    error_code: Optional[ErrorCode] = None,
    original_error: Optional[Exception] = None
) -> PlanningError:
    """
    Create a PlanningError with structured context.
    
    Args:
        message: Error message
        query: The query that failed to plan
        available_tools: List of available tools
        error_code: Specific error code
        original_error: Original exception if any
        
    Returns:
        PlanningError with structured context
    """
    context = {"query": query}
    if available_tools:
        context["available_tools"] = available_tools
    
    return PlanningError(
        message=message,
        error_code=error_code,
        context=context,
        original_error=original_error
    )