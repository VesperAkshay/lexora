"""
Logging configuration and utilities for the Lexora Agentic RAG SDK.

This module provides structured logging setup with correlation IDs, configurable log levels,
and integration with the SDK's error handling system.
"""

import logging
import logging.config
import sys
import json
import uuid
import time
from typing import Any, Dict, Optional, Union
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path

from ..exceptions import LexoraError, ErrorCode


# Context variable for correlation ID tracking
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class CorrelationIdFilter(logging.Filter):
    """
    Logging filter that adds correlation ID to log records.
    
    This filter automatically adds the current correlation ID to all log records,
    enabling request tracing across the system.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add correlation ID to the log record.
        
        Args:
            record: Log record to modify
            
        Returns:
            True to allow the record to be processed
        """
        correlation_id = correlation_id_var.get()
        record.correlation_id = correlation_id or "no-correlation"
        return True


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured JSON logs.
    
    This formatter creates structured log entries with consistent fields
    for better parsing and analysis in log aggregation systems.
    """
    
    def __init__(self, include_extra: bool = True):
        """
        Initialize structured formatter.
        
        Args:
            include_extra: Whether to include extra fields from log records
        """
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as structured JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log string
        """
        # Base log structure
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, 'correlation_id', 'no-correlation'),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None
            }
        
        # Add extra fields if enabled
        if self.include_extra:
            # Get all extra attributes (those not in the standard LogRecord)
            standard_attrs = {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
                'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'getMessage', 'correlation_id'
            }
            
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in standard_attrs and not key.startswith('_'):
                    # Ensure the value is JSON serializable
                    try:
                        json.dumps(value)
                        extra_fields[key] = value
                    except (TypeError, ValueError):
                        extra_fields[key] = str(value)
            
            if extra_fields:
                log_entry["extra"] = extra_fields
        
        return json.dumps(log_entry, ensure_ascii=False)


class LexoraLogger:
    """
    Enhanced logger wrapper with SDK-specific functionality.
    
    This class provides additional methods for logging SDK-specific events
    and integrates with the correlation ID system.
    """
    
    def __init__(self, name: str):
        """
        Initialize Lexora logger.
        
        Args:
            name: Logger name (typically module name)
        """
        self.logger = logging.getLogger(name)
        self.name = name
    
    def _log_with_context(self, level: int, msg: str, *args, **kwargs) -> None:
        """
        Log message with additional context.
        
        Args:
            level: Log level
            msg: Log message
            *args: Message formatting arguments
            **kwargs: Additional context fields
        """
        # Extract standard logging kwargs
        exc_info = kwargs.pop('exc_info', None)
        stack_info = kwargs.pop('stack_info', None)
        stacklevel = kwargs.pop('stacklevel', 1)
        
        # Add remaining kwargs as extra fields
        extra = kwargs if kwargs else None
        
        self.logger.log(
            level, msg, *args,
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel + 1,
            extra=extra
        )
    
    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log debug message with context."""
        self._log_with_context(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs) -> None:
        """Log info message with context."""
        self._log_with_context(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log warning message with context."""
        self._log_with_context(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs) -> None:
        """Log error message with context."""
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs) -> None:
        """Log critical message with context."""
        self._log_with_context(logging.CRITICAL, msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs) -> None:
        """Log exception with traceback."""
        kwargs['exc_info'] = True
        self.error(msg, *args, **kwargs)
    
    def log_operation_start(self, operation: str, **context) -> None:
        """
        Log the start of an operation.
        
        Args:
            operation: Name of the operation
            **context: Additional context fields
        """
        self.info(
            "Operation started: %s",
            operation,
            operation=operation,
            operation_phase="start",
            **context
        )
    
    def log_operation_end(self, operation: str, duration: float, success: bool = True, **context) -> None:
        """
        Log the end of an operation.
        
        Args:
            operation: Name of the operation
            duration: Operation duration in seconds
            success: Whether the operation was successful
            **context: Additional context fields
        """
        level = logging.INFO if success else logging.ERROR
        self._log_with_context(
            level,
            "Operation %s: %s (%.3fs)",
            "completed" if success else "failed",
            operation,
            duration,
            operation=operation,
            operation_phase="end",
            duration_seconds=duration,
            success=success,
            **context
        )
    
    def log_tool_execution(self, tool_name: str, parameters: Dict[str, Any], result: Optional[Dict[str, Any]] = None, error: Optional[Exception] = None) -> None:
        """
        Log tool execution details.
        
        Args:
            tool_name: Name of the executed tool
            parameters: Tool parameters
            result: Tool execution result (if successful)
            error: Error that occurred (if failed)
        """
        if error:
            self.error(
                "Tool execution failed: %s",
                tool_name,
                tool_name=tool_name,
                parameters=parameters,
                error_type=type(error).__name__,
                error_message=str(error),
                exc_info=True
            )
        else:
            self.info(
                "Tool executed successfully: %s",
                tool_name,
                tool_name=tool_name,
                parameters=parameters,
                result_keys=list(result.keys()) if result else None
            )
    
    def log_llm_request(self, provider: str, model: str, prompt_length: int, response_length: Optional[int] = None, duration: Optional[float] = None, error: Optional[Exception] = None) -> None:
        """
        Log LLM API request details.
        
        Args:
            provider: LLM provider name
            model: Model name
            prompt_length: Length of the prompt
            response_length: Length of the response (if successful)
            duration: Request duration in seconds
            error: Error that occurred (if failed)
        """
        if error:
            self.error(
                "LLM request failed: %s/%s",
                provider,
                model,
                provider=provider,
                model=model,
                prompt_length=prompt_length,
                duration_seconds=duration,
                error_type=type(error).__name__,
                error_message=str(error)
            )
        else:
            self.info(
                "LLM request successful: %s/%s",
                provider,
                model,
                provider=provider,
                model=model,
                prompt_length=prompt_length,
                response_length=response_length,
                duration_seconds=duration
            )
    
    def log_vector_operation(self, operation: str, corpus_name: str, document_count: Optional[int] = None, duration: Optional[float] = None, error: Optional[Exception] = None) -> None:
        """
        Log vector database operation details.
        
        Args:
            operation: Type of operation (search, add, delete, etc.)
            corpus_name: Name of the corpus
            document_count: Number of documents involved
            duration: Operation duration in seconds
            error: Error that occurred (if failed)
        """
        if error:
            self.error(
                "Vector DB operation failed: %s on %s",
                operation,
                corpus_name,
                operation=operation,
                corpus_name=corpus_name,
                document_count=document_count,
                duration_seconds=duration,
                error_type=type(error).__name__,
                error_message=str(error)
            )
        else:
            self.info(
                "Vector DB operation successful: %s on %s",
                operation,
                corpus_name,
                operation=operation,
                corpus_name=corpus_name,
                document_count=document_count,
                duration_seconds=duration
            )


def setup_logging(
    level: Union[str, int] = logging.INFO,
    format_type: str = "structured",
    log_file: Optional[str] = None,
    correlation_ids: bool = True,
    include_extra: bool = True
) -> None:
    """
    Set up logging configuration for the Lexora SDK.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Log format type ("structured" for JSON, "simple" for text)
        log_file: Optional file path for log output
        correlation_ids: Whether to enable correlation ID tracking
        include_extra: Whether to include extra fields in structured logs
        
    Raises:
        LexoraError: If logging setup fails
    """
    try:
        # Convert string level to int if needed
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        # Create formatters
        if format_type == "structured":
            formatter = StructuredFormatter(include_extra=include_extra)
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(correlation_id)s - %(message)s'
            )
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        
        if correlation_ids:
            console_handler.addFilter(CorrelationIdFilter())
        
        root_logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            
            if correlation_ids:
                file_handler.addFilter(CorrelationIdFilter())
            
            root_logger.addHandler(file_handler)
        
        # Set specific logger levels for noisy libraries
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        
        # Log successful setup
        logger = get_logger(__name__)
        logger.info(
            "Logging configured successfully",
            log_level=logging.getLevelName(level),
            format_type=format_type,
            log_file=log_file,
            correlation_ids=correlation_ids
        )
        
    except Exception as e:
        raise LexoraError(
            f"Failed to setup logging: {str(e)}",
            ErrorCode.INTERNAL_ERROR,
            original_error=e
        )


def get_logger(name: str) -> LexoraLogger:
    """
    Get a Lexora logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        LexoraLogger instance
    """
    return LexoraLogger(name)


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """
    Set correlation ID for the current context.
    
    Args:
        correlation_id: Correlation ID to set (generates UUID if None)
        
    Returns:
        The correlation ID that was set
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    correlation_id_var.set(correlation_id)
    return correlation_id


def get_correlation_id() -> Optional[str]:
    """
    Get the current correlation ID.
    
    Returns:
        Current correlation ID or None if not set
    """
    return correlation_id_var.get()


def clear_correlation_id() -> None:
    """Clear the current correlation ID."""
    correlation_id_var.set(None)


class LoggingContext:
    """
    Context manager for correlation ID management.
    
    This context manager automatically sets and clears correlation IDs
    for request tracking.
    """
    
    def __init__(self, correlation_id: Optional[str] = None):
        """
        Initialize logging context.
        
        Args:
            correlation_id: Correlation ID to use (generates UUID if None)
        """
        self.correlation_id = correlation_id
        self.previous_id = None
    
    def __enter__(self) -> str:
        """Enter the context and set correlation ID."""
        self.previous_id = get_correlation_id()
        self.correlation_id = set_correlation_id(self.correlation_id)
        return self.correlation_id
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context and restore previous correlation ID."""
        correlation_id_var.set(self.previous_id)


class OperationTimer:
    """
    Context manager for timing operations with automatic logging.
    
    This context manager automatically logs operation start/end times
    and durations.
    """
    
    def __init__(self, logger: LexoraLogger, operation: str, **context):
        """
        Initialize operation timer.
        
        Args:
            logger: Logger to use for timing logs
            operation: Name of the operation being timed
            **context: Additional context to include in logs
        """
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
        self.success = True
    
    def __enter__(self) -> 'OperationTimer':
        """Start timing the operation."""
        self.start_time = time.time()
        self.logger.log_operation_start(self.operation, **self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End timing and log results."""
        duration = time.time() - self.start_time
        self.success = exc_type is None
        
        if exc_type:
            self.context.update({
                "error_type": exc_type.__name__,
                "error_message": str(exc_val)
            })
        
        self.logger.log_operation_end(
            self.operation,
            duration,
            success=self.success,
            **self.context
        )
    
    def mark_failure(self, error_message: str) -> None:
        """
        Mark the operation as failed.
        
        Args:
            error_message: Error message to include
        """
        self.success = False
        self.context["error_message"] = error_message


# Convenience function for quick logging setup
def configure_logging(
    level: str = "INFO",
    structured: bool = True,
    file_path: Optional[str] = None
) -> None:
    """
    Quick logging configuration with sensible defaults.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        structured: Whether to use structured JSON logging
        file_path: Optional file path for log output
    """
    setup_logging(
        level=level,
        format_type="structured" if structured else "simple",
        log_file=file_path,
        correlation_ids=True,
        include_extra=True
    )