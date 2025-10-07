"""
Base tool interface for the Lexora Agentic RAG SDK.

This module provides the abstract base class that all tools must implement,
ensuring consistent interfaces and behavior across different tool implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import json
import asyncio
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, validator
from ..exceptions import LexoraError, ErrorCode, create_tool_error
from ..utils.logging import get_logger
from ..utils.validation import validate_parameters, validate_schema


class ToolStatus(str, Enum):
    """Enumeration of possible tool execution statuses."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    TIMEOUT = "timeout"


class ParameterType(str, Enum):
    """Enumeration of supported parameter types."""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class ToolParameter(BaseModel):
    """
    Model for tool parameter definition.
    
    This class defines the structure and validation rules for tool parameters.
    """
    name: str = Field(..., description="Parameter name")
    type: ParameterType = Field(..., description="Parameter type")
    description: str = Field(..., description="Parameter description")
    required: bool = Field(default=False, description="Whether parameter is required")
    default: Optional[Any] = Field(default=None, description="Default value if not provided")
    enum: Optional[List[Any]] = Field(default=None, description="Allowed values for parameter")
    minimum: Optional[Union[int, float]] = Field(default=None, description="Minimum value for numeric types")
    maximum: Optional[Union[int, float]] = Field(default=None, description="Maximum value for numeric types")
    pattern: Optional[str] = Field(default=None, description="Regex pattern for string validation")
    
    @validator('name')
    def validate_name(cls, v):
        """Validate parameter name format."""
        if not v or not isinstance(v, str):
            raise ValueError("Parameter name must be a non-empty string")
        if not v.replace('_', '').isalnum():
            raise ValueError("Parameter name must contain only alphanumeric characters and underscores")
        return v
    
    @validator('default')
    def validate_default_type(cls, v, values):
        """Validate that default value matches parameter type."""
        if v is None:
            return v
        
        param_type = values.get('type')
        
        # Check boolean first since bool is a subclass of int in Python
        if param_type == ParameterType.BOOLEAN:
            if not isinstance(v, bool):
                raise ValueError("Default value must be boolean for boolean parameter")
        elif param_type == ParameterType.STRING:
            if not isinstance(v, str):
                raise ValueError("Default value must be string for string parameter")
        elif param_type == ParameterType.INTEGER:
            # Explicitly exclude booleans from integer check
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError("Default value must be integer for integer parameter")
        elif param_type == ParameterType.NUMBER:
            # Exclude booleans from number check as well
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError("Default value must be number for number parameter")
        elif param_type == ParameterType.ARRAY:
            if not isinstance(v, list):
                raise ValueError("Default value must be array for array parameter")
        elif param_type == ParameterType.OBJECT:
            if not isinstance(v, dict):
                raise ValueError("Default value must be object for object parameter")
        
        return v


class ToolResult(BaseModel):
    """
    Model for tool execution results.
    
    This class provides a standardized format for tool outputs.
    """
    status: ToolStatus = Field(..., description="Execution status")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Result data")
    message: Optional[str] = Field(default=None, description="Human-readable message")
    error: Optional[str] = Field(default=None, description="Error message if status is error")
    execution_time: Optional[float] = Field(default=None, description="Execution time in seconds")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    
    @validator('status')
    def validate_status(cls, v):
        """Validate status value."""
        if not isinstance(v, ToolStatus):
            if isinstance(v, str) and v in [s.value for s in ToolStatus]:
                return ToolStatus(v)
            raise ValueError(f"Status must be one of: {[s.value for s in ToolStatus]}")
        return v
    
    @validator('error')
    def validate_error_with_status(cls, v, values):
        """Validate that error is provided when status is error."""
        status = values.get('status')
        if status == ToolStatus.ERROR and not v:
            raise ValueError("Error message must be provided when status is error")
        return v


class BaseTool(ABC):
    """
    Abstract base class for all RAG tools.
    
    This class defines the interface that all tools must implement to ensure
    consistent behavior and integration with the RAG agent system.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the base tool.
        
        Args:
            **kwargs: Additional configuration options
        """
        self.logger = get_logger(self.__class__.__name__)
        self.config = kwargs
        self._parameters: List[ToolParameter] = []
        self._setup_parameters()
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Tool name identifier.
        
        Returns:
            Unique name for this tool
        """
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """
        Tool description for users and LLMs.
        
        Returns:
            Human-readable description of what this tool does
        """
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """
        Tool version for compatibility tracking.
        
        Returns:
            Version string (e.g., "1.0.0")
        """
        pass
    
    @abstractmethod
    def _setup_parameters(self) -> None:
        """
        Set up tool parameters.
        
        This method should populate self._parameters with ToolParameter instances
        that define the expected inputs for this tool.
        """
        pass
    
    @abstractmethod
    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the core tool logic.
        
        This method contains the actual implementation of the tool's functionality.
        It should not handle parameter validation or result formatting - those are
        handled by the run() method.
        
        Args:
            **kwargs: Parameters for tool execution
            
        Returns:
            Tool-specific results
        """
        pass
    
    async def run(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.
        
        This method handles parameter validation, execution, error handling,
        and result formatting.
        
        Args:
            **kwargs: Parameters for tool execution
            
        Returns:
            ToolResult containing execution status and results
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Validate parameters
            validated_params = self.validate_parameters(**kwargs)
            
            # Execute the tool
            self.logger.info(f"Executing tool {self.name} with parameters: {list(validated_params.keys())}")
            
            result_data = await self._execute(**validated_params)
            
            # Calculate execution time
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Create successful result
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result_data,
                message=f"Tool {self.name} executed successfully",
                execution_time=execution_time,
                metadata={
                    "tool_name": self.name,
                    "tool_version": self.version,
                    "timestamp": start_time.isoformat()
                }
            )
            
        except LexoraError as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.logger.error(f"Tool {self.name} execution failed: {str(e)}")
            
            return ToolResult(
                status=ToolStatus.ERROR,
                error=str(e),
                message=f"Tool {self.name} execution failed",
                execution_time=execution_time,
                metadata={
                    "tool_name": self.name,
                    "tool_version": self.version,
                    "timestamp": start_time.isoformat(),
                    "error_code": e.error_code.value if hasattr(e, 'error_code') else None
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.logger.error(f"Unexpected error in tool {self.name}: {str(e)}")
            
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Unexpected error: {str(e)}",
                message=f"Tool {self.name} encountered an unexpected error",
                execution_time=execution_time,
                metadata={
                    "tool_name": self.name,
                    "tool_version": self.version,
                    "timestamp": start_time.isoformat()
                }
            )    
    def validate_parameters(self, **kwargs) -> Dict[str, Any]:
        """
        Validate input parameters against tool schema.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            Dictionary of validated parameters with defaults applied
            
        Raises:
            LexoraError: If validation fails
        """
        try:
            validated = {}
            
            # Check for required parameters
            required_params = [p.name for p in self._parameters if p.required]
            missing_params = [p for p in required_params if p not in kwargs]
            
            if missing_params:
                raise create_tool_error(
                    f"Missing required parameters: {missing_params}",
                    self.name,
                    ErrorCode.TOOL_INVALID_PARAMETERS
                )
            
            # Validate each parameter
            for param in self._parameters:
                value = kwargs.get(param.name, param.default)
                
                if value is not None:
                    validated[param.name] = self._validate_parameter_value(param, value)
                elif param.required:
                    raise create_tool_error(
                        f"Required parameter '{param.name}' is missing",
                        self.name,
                        ErrorCode.TOOL_INVALID_PARAMETERS
                    )
            
            # Check for unexpected parameters
            expected_params = {p.name for p in self._parameters}
            unexpected_params = set(kwargs.keys()) - expected_params
            
            if unexpected_params:
                self.logger.warning(f"Unexpected parameters ignored: {unexpected_params}")
            
            return validated
            
        except Exception as e:
            if isinstance(e, LexoraError):
                raise
            raise create_tool_error(
                f"Parameter validation failed: {str(e)}",
                self.name,
                ErrorCode.TOOL_INVALID_PARAMETERS
            )
    
    def _validate_parameter_value(self, param: ToolParameter, value: Any) -> Any:
        """
        Validate a single parameter value.
        
        Args:
            param: Parameter definition
            value: Value to validate
            
        Returns:
            Validated value (potentially converted)
            
        Raises:
            LexoraError: If validation fails
        """
        # Type validation and conversion
        if param.type == ParameterType.STRING:
            if not isinstance(value, str):
                value = str(value)
        elif param.type == ParameterType.INTEGER:
            if not isinstance(value, int):
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    raise create_tool_error(
                        f"Parameter '{param.name}' must be an integer",
                        self.name,
                        ErrorCode.TOOL_INVALID_PARAMETERS
                    )
        elif param.type == ParameterType.NUMBER:
            if not isinstance(value, (int, float)):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    raise create_tool_error(
                        f"Parameter '{param.name}' must be a number",
                        self.name,
                        ErrorCode.TOOL_INVALID_PARAMETERS
                    )
        elif param.type == ParameterType.BOOLEAN:
            if not isinstance(value, bool):
                if isinstance(value, str):
                    lower_val = value.lower()
                    if lower_val in ('true', '1', 'yes', 'on'):
                        value = True
                    elif lower_val in ('false', '0', 'no', 'off'):
                        value = False
                    else:
                        raise create_tool_error(
                            f"Parameter '{param.name}' has invalid boolean value: {value}",
                            self.name,
                            ErrorCode.TOOL_INVALID_PARAMETERS
                        )
                else:
                    value = bool(value)
            
            elif param.type == ParameterType.ARRAY:
                if not isinstance(value, list):
                    raise create_tool_error(
                        f"Parameter '{param.name}' must be an array",
                        self.name,
                        ErrorCode.TOOL_INVALID_PARAMETERS
                    )
            elif param.type == ParameterType.OBJECT:
                if not isinstance(value, dict):
                    raise create_tool_error(
                        f"Parameter '{param.name}' must be an object",
                        self.name,
                        ErrorCode.TOOL_INVALID_PARAMETERS
                    )
            
            # Enum validation
            if param.enum and value not in param.enum:
                raise create_tool_error(
                    f"Parameter '{param.name}' must be one of: {param.enum}",
                    self.name,
                    ErrorCode.TOOL_INVALID_PARAMETERS
                )
            
            # Range validation for numeric types
            if param.type in (ParameterType.INTEGER, ParameterType.NUMBER):
                if param.minimum is not None and value < param.minimum:
                    raise create_tool_error(
                        f"Parameter '{param.name}' must be >= {param.minimum}",
                        self.name,
                        ErrorCode.TOOL_INVALID_PARAMETERS
                    )
                if param.maximum is not None and value > param.maximum:
                    raise create_tool_error(
                        f"Parameter '{param.name}' must be <= {param.maximum}",
                        self.name,
                        ErrorCode.TOOL_INVALID_PARAMETERS
                    )
        
        # Pattern validation for strings
        if param.type == ParameterType.STRING and param.pattern:
            import re
            if not re.match(param.pattern, value):
                raise create_tool_error(
                    f"Parameter '{param.name}' does not match required pattern",
                    self.name,
                    ErrorCode.TOOL_INVALID_PARAMETERS
                )
        
        return value
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Return JSON schema for tool parameters.
        
        Returns:
            JSON schema dictionary describing the tool's parameters
        """
        properties = {}
        required = []
        
        for param in self._parameters:
            prop = {
                "type": param.type.value,
                "description": param.description
            }
            
            if param.default is not None:
                prop["default"] = param.default
            
            if param.enum:
                prop["enum"] = param.enum
            
            if param.type in (ParameterType.INTEGER, ParameterType.NUMBER):
                if param.minimum is not None:
                    prop["minimum"] = param.minimum
                if param.maximum is not None:
                    prop["maximum"] = param.maximum
            
            if param.type == ParameterType.STRING and param.pattern:
                prop["pattern"] = param.pattern
            
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        schema = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False
        }
        
        if required:
            schema["required"] = required
        
        return schema
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get tool metadata for registration and discovery.
        
        Returns:
            Dictionary containing tool metadata
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "schema": self.get_schema(),
            "parameters": [param.dict() for param in self._parameters],
            "class": self.__class__.__name__,
            "module": self.__class__.__module__
        }
    
    def __str__(self) -> str:
        """Return string representation of the tool."""
def validate_tool_interface(tool_class: type) -> bool:
    """
    Validate that a class implements the BaseTool interface correctly.
    
    Args:
        tool_class: Class to validate
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        LexoraError: If validation fails with details
    """
    if not issubclass(tool_class, BaseTool):
        raise create_tool_error(
            f"Tool class {tool_class.__name__} must inherit from BaseTool",
            tool_class.__name__,
            ErrorCode.TOOL_VALIDATION_FAILED
        )
    
    # Check that all abstract methods are implemented
    abstract_methods = set()
    for base in tool_class.__mro__:
        if base is not BaseTool:
            continue
        for name, value in base.__dict__.items():
            if getattr(value, '__isabstractmethod__', False):
                abstract_methods.add(name)
    
    # Check if concrete class still has abstract methods
    unimplemented = []
    for name in abstract_methods:
        if getattr(getattr(tool_class, name, None), '__isabstractmethod__', False):
            unimplemented.append(name)
    
    if unimplemented:
        raise create_tool_error(
            f"Tool class {tool_class.__name__} has unimplemented abstract methods: {unimplemented}",
            tool_class.__name__,
            ErrorCode.TOOL_VALIDATION_FAILED
        )
    
    return True    
    if missing_methods:
        raise create_tool_error(
            f"Tool class {tool_class.__name__} missing required methods: {missing_methods}",
            tool_class.__name__,
            ErrorCode.TOOL_VALIDATION_FAILED
        )
    
    return True


def create_tool_parameter(
    name: str,
    param_type: ParameterType,
    description: str,
    required: bool = False,
    default: Any = None,
    **kwargs
) -> ToolParameter:
    """
    Convenience function to create a ToolParameter.
    
    Args:
        name: Parameter name
        param_type: Parameter type
        description: Parameter description
        required: Whether parameter is required
        default: Default value
        **kwargs: Additional parameter options
        
    Returns:
        ToolParameter instance
    """
    return ToolParameter(
        name=name,
        type=param_type,
        description=description,
        required=required,
        default=default,
        **kwargs
    )