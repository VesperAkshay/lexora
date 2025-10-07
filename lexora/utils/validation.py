"""
Validation utilities for the Lexora Agentic RAG SDK.

This module provides parameter validation helpers and schema validation
for tool parameters, ensuring data integrity throughout the system.
"""

import re
from typing import Any, Dict, List, Optional, Union, Type, get_origin, get_args
from enum import Enum
import inspect

try:
    from pydantic import BaseModel, ValidationError, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

from ..exceptions import LexoraError, ErrorCode, create_tool_error


class ValidationSeverity(Enum):
    """
    Severity levels for validation issues.
    """
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationResult:
    """
    Result of a validation operation.
    
    Contains information about validation success/failure and any issues found.
    """
    
    def __init__(self, is_valid: bool = True):
        """
        Initialize validation result.
        
        Args:
            is_valid: Whether validation passed
        """
        self.is_valid = is_valid
        self.issues: List[Dict[str, Any]] = []
        self.warnings: List[str] = []
    
    def add_issue(
        self,
        field: str,
        message: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        value: Any = None
    ) -> None:
        """
        Add a validation issue.
        
        Args:
            field: Field name that failed validation
            message: Error message
            severity: Severity level
            value: The invalid value
        """
        self.issues.append({
            "field": field,
            "message": message,
            "severity": severity.value,
            "value": value
        })
        
        if severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL):
            self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """
        Add a validation warning.
        
        Args:
            message: Warning message
        """
        self.warnings.append(message)
    
    def get_errors(self) -> List[Dict[str, Any]]:
        """Get all error-level issues."""
        return [
            issue for issue in self.issues
            if issue["severity"] in ("error", "critical")
        ]
    
    def get_warnings(self) -> List[Dict[str, Any]]:
        """Get all warning-level issues."""
        return [
            issue for issue in self.issues
            if issue["severity"] == "warning"
        ] + [{"message": w, "severity": "warning"} for w in self.warnings]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary."""
        return {
            "is_valid": self.is_valid,
            "issues": self.issues,
            "warnings": self.warnings,
            "error_count": len(self.get_errors()),
            "warning_count": len(self.get_warnings())
        }
    
    def __bool__(self) -> bool:
        """Return validation status."""
        return self.is_valid


class ParameterValidator:
    """
    Utility class for validating function/method parameters.
    
    Provides various validation methods for common parameter types and patterns.
    """
    
    @staticmethod
    def validate_required_params(params: Dict[str, Any], required: List[str]) -> ValidationResult:
        """
        Validate that all required parameters are present.
        
        Args:
            params: Parameter dictionary to validate
            required: List of required parameter names
            
        Returns:
            ValidationResult indicating success/failure
        """
        result = ValidationResult()
        
        for param_name in required:
            if param_name not in params:
                result.add_issue(
                    param_name,
                    f"Required parameter '{param_name}' is missing",
                    ValidationSeverity.ERROR
                )
            elif params[param_name] is None:
                result.add_issue(
                    param_name,
                    f"Required parameter '{param_name}' cannot be None",
                    ValidationSeverity.ERROR
                )
        
        return result
    
    @staticmethod
    def validate_string_param(
        value: Any,
        param_name: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        allowed_values: Optional[List[str]] = None,
        required: bool = True
    ) -> ValidationResult:
        """
        Validate a string parameter.
        
        Args:
            value: Value to validate
            param_name: Parameter name for error messages
            min_length: Minimum string length
            max_length: Maximum string length
            pattern: Regex pattern to match
            allowed_values: List of allowed string values
            required: Whether the parameter is required
            
        Returns:
            ValidationResult
        """
        result = ValidationResult()
        
        # Check if value is provided
        if value is None:
            if required:
                result.add_issue(
                    param_name,
                    f"Parameter '{param_name}' is required",
                    ValidationSeverity.ERROR
                )
            return result
        
        # Check type
        if not isinstance(value, str):
            result.add_issue(
                param_name,
                f"Parameter '{param_name}' must be a string, got {type(value).__name__}",
                ValidationSeverity.ERROR,
                value
            )
            return result
        
        # Check length constraints
        if min_length is not None and len(value) < min_length:
            result.add_issue(
                param_name,
                f"Parameter '{param_name}' must be at least {min_length} characters long",
                ValidationSeverity.ERROR,
                value
            )
        
        if max_length is not None and len(value) > max_length:
            result.add_issue(
                param_name,
                f"Parameter '{param_name}' must be at most {max_length} characters long",
                ValidationSeverity.ERROR,
                value
            )
        
        # Check pattern
        if pattern is not None:
            try:
                if not re.match(pattern, value):
                    result.add_issue(
                        param_name,
                        f"Parameter '{param_name}' does not match required pattern: {pattern}",
                        ValidationSeverity.ERROR,
                        value
                    )
            except re.error as e:
                result.add_issue(
                    param_name,
                    f"Invalid regex pattern for '{param_name}': {str(e)}",
                    ValidationSeverity.CRITICAL
                )
        
        # Check allowed values
        if allowed_values is not None and value not in allowed_values:
            result.add_issue(
                param_name,
                f"Parameter '{param_name}' must be one of {allowed_values}, got '{value}'",
                ValidationSeverity.ERROR,
                value
            )
        
        return result
    
    @staticmethod
    def validate_numeric_param(
        value: Any,
        param_name: str,
        param_type: Type = int,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        required: bool = True
    ) -> ValidationResult:
        """
        Validate a numeric parameter.
        
        Args:
            value: Value to validate
            param_name: Parameter name for error messages
            param_type: Expected numeric type (int or float)
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            required: Whether the parameter is required
            
        Returns:
            ValidationResult
        """
        result = ValidationResult()
        
        # Check if value is provided
        if value is None:
            if required:
                result.add_issue(
                    param_name,
                    f"Parameter '{param_name}' is required",
                    ValidationSeverity.ERROR
                )
            return result
        
        # Check type
        if not isinstance(value, (int, float)):
            result.add_issue(
                param_name,
                f"Parameter '{param_name}' must be numeric, got {type(value).__name__}",
                ValidationSeverity.ERROR,
                value
            )
            return result
        
        # Check specific type if required
        if param_type == int and (not isinstance(value, int) or isinstance(value, bool)):
            result.add_issue(
                param_name,
                f"Parameter '{param_name}' must be an integer, got {type(value).__name__}",
                ValidationSeverity.ERROR,
                value
            )        
        # Check range constraints
        if min_value is not None and value < min_value:
            result.add_issue(
                param_name,
                f"Parameter '{param_name}' must be at least {min_value}",
                ValidationSeverity.ERROR,
                value
            )
        
        if max_value is not None and value > max_value:
            result.add_issue(
                param_name,
                f"Parameter '{param_name}' must be at most {max_value}",
                ValidationSeverity.ERROR,
                value
            )
        
        return result
    
    @staticmethod
    def validate_list_param(
        value: Any,
        param_name: str,
        item_type: Optional[Type] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        required: bool = True
    ) -> ValidationResult:
        """
        Validate a list parameter.
        
        Args:
            value: Value to validate
            param_name: Parameter name for error messages
            item_type: Expected type of list items
            min_length: Minimum list length
            max_length: Maximum list length
            required: Whether the parameter is required
            
        Returns:
            ValidationResult
        """
        result = ValidationResult()
        
        # Check if value is provided
        if value is None:
            if required:
                result.add_issue(
                    param_name,
                    f"Parameter '{param_name}' is required",
                    ValidationSeverity.ERROR
                )
            return result
        
        # Check type
        if not isinstance(value, list):
            result.add_issue(
                param_name,
                f"Parameter '{param_name}' must be a list, got {type(value).__name__}",
                ValidationSeverity.ERROR,
                value
            )
            return result
        
        # Check length constraints
        if min_length is not None and len(value) < min_length:
            result.add_issue(
                param_name,
                f"Parameter '{param_name}' must have at least {min_length} items",
                ValidationSeverity.ERROR,
                value
            )
        
        if max_length is not None and len(value) > max_length:
            result.add_issue(
                param_name,
                f"Parameter '{param_name}' must have at most {max_length} items",
                ValidationSeverity.ERROR,
                value
            )
        
        # Check item types
        if item_type is not None:
            for i, item in enumerate(value):
                if not isinstance(item, item_type):
                    result.add_issue(
                        f"{param_name}[{i}]",
                        f"List item at index {i} must be of type {item_type.__name__}, got {type(item).__name__}",
                        ValidationSeverity.ERROR,
                        item
                    )
        
        return result
    
    @staticmethod
    def validate_dict_param(
        value: Any,
        param_name: str,
        required_keys: Optional[List[str]] = None,
        optional_keys: Optional[List[str]] = None,
        key_type: Optional[Type] = None,
        value_type: Optional[Type] = None,
        required: bool = True
    ) -> ValidationResult:
        """
        Validate a dictionary parameter.
        
        Args:
            value: Value to validate
            param_name: Parameter name for error messages
            required_keys: List of required dictionary keys
            optional_keys: List of optional dictionary keys
            key_type: Expected type of dictionary keys
            value_type: Expected type of dictionary values
            required: Whether the parameter is required
            
        Returns:
            ValidationResult
        """
        result = ValidationResult()
        
        # Check if value is provided
        if value is None:
            if required:
                result.add_issue(
                    param_name,
                    f"Parameter '{param_name}' is required",
                    ValidationSeverity.ERROR
                )
            return result
        
        # Check type
        if not isinstance(value, dict):
            result.add_issue(
                param_name,
                f"Parameter '{param_name}' must be a dictionary, got {type(value).__name__}",
                ValidationSeverity.ERROR,
                value
            )
            return result
        
        # Check required keys
        if required_keys:
            for key in required_keys:
                if key not in value:
                    result.add_issue(
                        f"{param_name}.{key}",
                        f"Required key '{key}' is missing from '{param_name}'",
                        ValidationSeverity.ERROR
                    )
        
        # Check for unexpected keys
        if required_keys is not None or optional_keys is not None:
            allowed_keys = set(required_keys or []) | set(optional_keys or [])
            for key in value.keys():
                if key not in allowed_keys:
                    result.add_warning(
                        f"Unexpected key '{key}' in parameter '{param_name}'"
                    )
        
        # Check key and value types
        for key, val in value.items():
            if key_type is not None and not isinstance(key, key_type):
                result.add_issue(
                    f"{param_name}.{key}",
                    f"Dictionary key '{key}' must be of type {key_type.__name__}, got {type(key).__name__}",
                    ValidationSeverity.ERROR,
                    key
                )
            
            if value_type is not None and not isinstance(val, value_type):
                result.add_issue(
                    f"{param_name}.{key}",
                    f"Dictionary value for key '{key}' must be of type {value_type.__name__}, got {type(val).__name__}",
                    ValidationSeverity.ERROR,
                    val
                )
        
        return result


class SchemaValidator:
    """
    Schema-based validation using Pydantic models or custom schemas.
    
    Provides validation against predefined schemas for complex data structures.
    """
    
    @staticmethod
    def validate_with_pydantic(data: Dict[str, Any], model_class: Type[BaseModel]) -> ValidationResult:
        """
        Validate data against a Pydantic model.
        
        Args:
            data: Data to validate
            model_class: Pydantic model class to validate against
            
        Returns:
            ValidationResult
            
        Raises:
            LexoraError: If Pydantic is not available
        """
        if not PYDANTIC_AVAILABLE:
            raise LexoraError(
                "Pydantic is not available. Install with: pip install pydantic",
                ErrorCode.TOOL_EXECUTION_FAILED
            )
        
        result = ValidationResult()
        
        try:
            # Attempt to create model instance
            model_class(**data)
            return result  # Validation successful
        
        except ValidationError as e:
            # Convert Pydantic errors to our format
            for error in e.errors():
                field_path = ".".join(str(loc) for loc in error["loc"])
                result.add_issue(
                    field_path,
                    error["msg"],
                    ValidationSeverity.ERROR,
                    error.get("input")
                )
        
        except Exception as e:
            result.add_issue(
                "schema",
                f"Unexpected validation error: {str(e)}",
                ValidationSeverity.CRITICAL
            )
        
        return result
    
    @staticmethod
    def validate_tool_output(output: Any, tool_name: str) -> ValidationResult:
        """
        Validate tool output format according to SDK requirements.
        
        Tool outputs must be dictionaries with specific structure for
        integration with the reasoning engine.
        
        Args:
            output: Tool output to validate
            tool_name: Name of the tool for error messages
            
        Returns:
            ValidationResult
        """
        result = ValidationResult()
        
        # Check that output is a dictionary
        if not isinstance(output, dict):
            result.add_issue(
                "output",
                f"Tool '{tool_name}' must return a dictionary, got {type(output).__name__}",
                ValidationSeverity.ERROR,
                output
            )
            return result
        
        # Check for required fields
        required_fields = ["status", "data"]
        for field in required_fields:
            if field not in output:
                result.add_issue(
                    field,
                    f"Tool '{tool_name}' output missing required field '{field}'",
                    ValidationSeverity.ERROR
                )
        
        # Validate status field
        if "status" in output:
            status = output["status"]
            if not isinstance(status, str):
                result.add_issue(
                    "status",
                    f"Tool '{tool_name}' status must be a string, got {type(status).__name__}",
                    ValidationSeverity.ERROR,
                    status
                )
            elif status not in ["success", "error", "partial"]:
                result.add_warning(
                    f"Tool '{tool_name}' status '{status}' is not a standard value (success, error, partial)"
                )
        
        # Validate data field
        if "data" in output:
            data = output["data"]
            if not isinstance(data, dict):
                result.add_issue(
                    "data",
                    f"Tool '{tool_name}' data must be a dictionary, got {type(data).__name__}",
                    ValidationSeverity.ERROR,
                    data
                )
        
        # Check for optional but recommended fields
        recommended_fields = ["message", "execution_time"]
        for field in recommended_fields:
            if field not in output:
                result.add_warning(
                    f"Tool '{tool_name}' output missing recommended field '{field}'"
                )
        
        return result
    
    @staticmethod
    def create_tool_schema(
        required_params: Optional[List[str]] = None,
        optional_params: Optional[List[str]] = None,
        param_types: Optional[Dict[str, Type]] = None,
        param_constraints: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Create a validation schema for tool parameters.
        
        Args:
            required_params: List of required parameter names
            optional_params: List of optional parameter names
            param_types: Dictionary mapping parameter names to types
            param_constraints: Dictionary mapping parameter names to constraint dictionaries
            
        Returns:
            Schema dictionary for validation
        """
        schema = {
            "type": "object",
            "properties": {},
            "required": required_params or [],
            "additionalProperties": False
        }
        
        all_params = (required_params or []) + (optional_params or [])
        
        for param in all_params:
            param_schema = {"type": "string"}  # Default type
            
            # Set type based on param_types
            if param_types and param in param_types:
                param_type = param_types[param]
                if param_type == str:
                    param_schema["type"] = "string"
                elif param_type == int:
                    param_schema["type"] = "integer"
                elif param_type == float:
                    param_schema["type"] = "number"
                elif param_type == bool:
                    param_schema["type"] = "boolean"
                elif param_type == list:
                    param_schema["type"] = "array"
                elif param_type == dict:
                    param_schema["type"] = "object"
            
            # Add constraints
            if param_constraints and param in param_constraints:
                constraints = param_constraints[param]
                param_schema.update(constraints)
            
            schema["properties"][param] = param_schema
        
        return schema


# Convenience functions for common validation patterns

def validate_parameters(
    params: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
    pydantic_model: Optional[Type[BaseModel]] = None,
    **validation_rules
) -> ValidationResult:
    """
    Validate parameters using various validation methods.
    
    Args:
        params: Parameters to validate
        schema: JSON schema for validation
        pydantic_model: Pydantic model for validation
        **validation_rules: Additional validation rules
        
    Returns:
        ValidationResult
        
    Raises:
        LexoraError: If validation configuration is invalid
    """
    result = ValidationResult()
    
    # Validate with Pydantic model if provided
    if pydantic_model:
        pydantic_result = SchemaValidator.validate_with_pydantic(params, pydantic_model)
        result.issues.extend(pydantic_result.issues)
        result.warnings.extend(pydantic_result.warnings)
        if not pydantic_result.is_valid:
            result.is_valid = False
    
    # Validate required parameters
    if "required" in validation_rules:
        required_result = ParameterValidator.validate_required_params(
            params, validation_rules["required"]
        )
        result.issues.extend(required_result.issues)
        if not required_result.is_valid:
            result.is_valid = False
    
    return result


def validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> ValidationResult:
    """
    Validate data against a JSON schema.
    
    Args:
        data: Data to validate
        schema: JSON schema to validate against
        
    Returns:
        ValidationResult
        
    Note:
        This is a basic implementation. For full JSON Schema support,
        consider using the jsonschema library.
    """
    result = ValidationResult()
    
    # Basic schema validation (simplified)
    if schema.get("type") == "object":
        if not isinstance(data, dict):
            result.add_issue(
                "root",
                f"Expected object, got {type(data).__name__}",
                ValidationSeverity.ERROR,
                data
            )
            return result
        
        # Check required properties
        required = schema.get("required", [])
        for prop in required:
            if prop not in data:
                result.add_issue(
                    prop,
                    f"Required property '{prop}' is missing",
                    ValidationSeverity.ERROR
                )
        
        # Validate properties
        properties = schema.get("properties", {})
        for prop, value in data.items():
            if prop in properties:
                prop_schema = properties[prop]
def validate_schema(
    data: Dict[str, Any],
    schema: Dict[str, Any],
    _depth: int = 0,
    _max_depth: int = 100
) -> ValidationResult:
    """
    Validate data against a JSON schema.

    Args:
        data: Data to validate
        schema: JSON schema to validate against

    Returns:
        ValidationResult

    Note:
        This is a basic implementation. For full JSON Schema support,
        consider using the jsonschema library.
    """
    result = ValidationResult()
    
    if _depth > _max_depth:
        result.add_issue(
            "root",
            f"Schema validation depth limit exceeded ({_max_depth})",
            ValidationSeverity.CRITICAL
        )
        return result

    # Basic schema validation (simplified)
    if schema.get("type") == "object":
        if not isinstance(data, dict):
            result.add_issue(
                "root",
                f"Expected object, got {type(data).__name__}",
                ValidationSeverity.ERROR,
                data
            )
            return result

        # Check required properties
        required = schema.get("required", [])
        for prop in required:
            if prop not in data:
                result.add_issue(
                    prop,
                    f"Required property '{prop}' is missing",
                    ValidationSeverity.ERROR
                )

        # Validate properties
        properties = schema.get("properties", {})
        for prop, value in data.items():
            if prop in properties:
                prop_schema = properties[prop]
                prop_result = validate_schema(
                    value,
                    prop_schema,
                    _depth + 1,
                    _max_depth
                )

                # Prefix field names with current property
                for issue in prop_result.issues:
                    issue["field"] = (
                        f"{prop}.{issue['field']}"
                        if issue["field"] != "root"
                        else prop
                    )

                result.issues.extend(prop_result.issues)
                if not prop_result.is_valid:
                    result.is_valid = False

    return result    return result


def validate_tool_parameters(tool_name: str, parameters: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """
    Validate tool parameters and raise exception if invalid.
    
    Args:
        tool_name: Name of the tool
        parameters: Parameters to validate
        schema: Validation schema
        
    Raises:
        LexoraError: If validation fails
    """
    result = validate_schema(parameters, schema)
    
    if not result.is_valid:
        error_messages = [issue["message"] for issue in result.get_errors()]
        raise create_tool_error(
            f"Invalid parameters for tool '{tool_name}': {'; '.join(error_messages)}",
            tool_name,
            parameters,
            ErrorCode.TOOL_INVALID_PARAMETERS
        )


def validate_tool_output_format(tool_name: str, output: Any) -> None:
    """
    Validate tool output format and raise exception if invalid.
    
    Args:
        tool_name: Name of the tool
        output: Tool output to validate
        
    Raises:
        LexoraError: If validation fails
    """
    result = SchemaValidator.validate_tool_output(output, tool_name)
    
    if not result.is_valid:
        error_messages = [issue["message"] for issue in result.get_errors()]
        raise create_tool_error(
            f"Invalid output format for tool '{tool_name}': {'; '.join(error_messages)}",
            tool_name,
            {"output": output},
            ErrorCode.TOOL_EXECUTION_FAILED
        )