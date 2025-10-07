"""
Response and execution models for the Lexora Agentic RAG SDK.

This module contains models for representing query intents, tool calls, execution plans,
tool results, and agent responses throughout the RAG system workflow.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator
from .core import SearchResult


class QueryIntent(BaseModel):
    """
    Parsed intent from user query.
    
    This model represents the analyzed intent of a user query, including
    the action to be performed, extracted entities, and confidence score.
    
    Attributes:
        action: The primary action identified in the query
        entities: Extracted entities and their values from the query
        confidence: Confidence score for the intent classification (0.0-1.0)
    """
    
    action: str = Field(
        ...,
        description="The primary action identified in the query"
    )
    entities: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted entities and their values from the query"
    )
    confidence: float = Field(
        ...,
        description="Confidence score for the intent classification (0.0-1.0)"
    )
    
    @validator('action')
    def validate_action(cls, v):
        """Validate that action is not empty."""
        if not v or not v.strip():
            raise ValueError("Action cannot be empty")
        return v.strip()
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Validate confidence is between 0.0 and 1.0."""
        if not isinstance(v, (int, float)):
            raise ValueError("Confidence must be a number")
        if v < 0.0 or v > 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return float(v)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the query intent to a dictionary representation.
        
        Returns:
            Dictionary representation of the query intent
        """
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryIntent':
        """
        Create a QueryIntent instance from a dictionary.
        
        Args:
            data: Dictionary containing query intent data
            
        Returns:
            QueryIntent instance
        """
        return cls(**data)


class ToolCall(BaseModel):
    """
    Represents a planned tool execution.
    
    This model defines a tool that should be executed as part of an execution plan,
    including its parameters and dependencies on other tool calls.
    
    Attributes:
        tool_name: Name of the tool to execute
        parameters: Parameters to pass to the tool
        depends_on: List of tool call IDs this call depends on
    """
    
    tool_name: str = Field(
        ...,
        description="Name of the tool to execute"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters to pass to the tool"
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="List of tool call IDs this call depends on"
    )
    
    @validator('tool_name')
    def validate_tool_name(cls, v):
        """Validate that tool name is not empty."""
        if not v or not v.strip():
            raise ValueError("Tool name cannot be empty")
        return v.strip()
    
    @validator('depends_on')
    def validate_depends_on(cls, v):
        """Validate dependencies list."""
        if not isinstance(v, list):
            raise ValueError("Dependencies must be a list")
        # Validate each dependency is a non-empty string
        for dep in v:
            if not isinstance(dep, str) or not dep.strip():
                raise ValueError("Each dependency must be a non-empty string")
        return [dep.strip() for dep in v]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the tool call to a dictionary representation.
        
        Returns:
            Dictionary representation of the tool call
        """
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolCall':
        """
        Create a ToolCall instance from a dictionary.
        
        Args:
            data: Dictionary containing tool call data
            
        Returns:
            ToolCall instance
        """
        return cls(**data)


class ExecutionPlan(BaseModel):
    """
    Complete execution plan for a query.
    
    This model represents the complete plan for executing a user query,
    including the sequence of tool calls and estimated complexity.
    
    Attributes:
        query: The original user query
        tool_calls: List of tool calls to execute in order
        estimated_steps: Estimated number of execution steps
    """
    
    query: str = Field(
        ...,
        description="The original user query"
    )
    tool_calls: List[ToolCall] = Field(
        default_factory=list,
        description="List of tool calls to execute in order"
    )
    estimated_steps: int = Field(
        ...,
        description="Estimated number of execution steps"
    )
    
    @validator('query')
    def validate_query(cls, v):
        """Validate that query is not empty."""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()
    
    @validator('estimated_steps')
    def validate_estimated_steps(cls, v):
        """Validate estimated steps is non-negative."""
        if not isinstance(v, int):
            raise ValueError("Estimated steps must be an integer")
        if v < 0:
            raise ValueError("Estimated steps cannot be negative")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the execution plan to a dictionary representation.
        
        Returns:
            Dictionary representation of the execution plan
        """
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionPlan':
        """
        Create an ExecutionPlan instance from a dictionary.
        
        Args:
            data: Dictionary containing execution plan data
            
        Returns:
            ExecutionPlan instance
        """
        # Handle nested ToolCall objects
        if 'tool_calls' in data:
            data = data.copy()
            data['tool_calls'] = [
                ToolCall.from_dict(tc) if isinstance(tc, dict) else tc
                for tc in data['tool_calls']
            ]
        return cls(**data)


class ToolResult(BaseModel):
    """
    Result from tool execution.
    
    This model represents the result of executing a specific tool,
    including success/failure status, returned data, and execution metrics.
    
    Attributes:
        tool_name: Name of the tool that was executed
        status: Execution status ("success", "error", "timeout", etc.)
        data: Data returned by the tool execution
        error: Error message if execution failed
        execution_time: Time taken to execute the tool in seconds
    """
    
    tool_name: str = Field(
        ...,
        description="Name of the tool that was executed"
    )
    status: str = Field(
        ...,
        description="Execution status ('success', 'error', 'timeout', etc.)"
    )
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Data returned by the tool execution"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if execution failed"
    )
    execution_time: float = Field(
        ...,
        description="Time taken to execute the tool in seconds"
    )
    
    @validator('tool_name')
    def validate_tool_name(cls, v):
        """Validate that tool name is not empty."""
        if not v or not v.strip():
            raise ValueError("Tool name cannot be empty")
        return v.strip()
    
    @validator('status')
    def validate_status(cls, v):
        """Validate status is not empty."""
        if not v or not v.strip():
            raise ValueError("Status cannot be empty")
        return v.strip().lower()
    
    @validator('execution_time')
    def validate_execution_time(cls, v):
        """Validate execution time is non-negative."""
        if not isinstance(v, (int, float)):
            raise ValueError("Execution time must be a number")
        if v < 0:
            raise ValueError("Execution time cannot be negative")
        return float(v)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the tool result to a dictionary representation.
        
        Returns:
            Dictionary representation of the tool result
        """
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolResult':
        """
        Create a ToolResult instance from a dictionary.
        
        Args:
            data: Dictionary containing tool result data
            
        Returns:
            ToolResult instance
        """
        return cls(**data)


class AgentResponse(BaseModel):
    """
    Final response from RAG agent.
    
    This model represents the complete response from the RAG agent after
    processing a user query, including the answer, sources, and execution details.
    
    Attributes:
        answer: The final answer to the user's query
        sources: List of search results that contributed to the answer
        tool_calls_made: List of tool results from the execution
        confidence: Confidence score for the generated answer (0.0-1.0)
        execution_time: Total time taken to process the query in seconds
    """
    
    answer: str = Field(
        ...,
        description="The final answer to the user's query"
    )
    sources: List[SearchResult] = Field(
        default_factory=list,
        description="List of search results that contributed to the answer"
    )
    tool_calls_made: List[ToolResult] = Field(
        default_factory=list,
        description="List of tool results from the execution"
    )
    confidence: float = Field(
        ...,
        description="Confidence score for the generated answer (0.0-1.0)"
    )
    execution_time: float = Field(
        ...,
        description="Total time taken to process the query in seconds"
    )
    
    @validator('answer')
    def validate_answer(cls, v):
        """Validate that answer is not empty."""
        if not v or not v.strip():
            raise ValueError("Answer cannot be empty")
        return v.strip()
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Validate confidence is between 0.0 and 1.0."""
        if not isinstance(v, (int, float)):
            raise ValueError("Confidence must be a number")
        if v < 0.0 or v > 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return float(v)
    
    @validator('execution_time')
    def validate_execution_time(cls, v):
        """Validate execution time is non-negative."""
        if not isinstance(v, (int, float)):
            raise ValueError("Execution time must be a number")
        if v < 0:
            raise ValueError("Execution time cannot be negative")
        return float(v)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the agent response to a dictionary representation.
        
        Returns:
            Dictionary representation of the agent response
        """
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentResponse':
        """
        Create an AgentResponse instance from a dictionary.
        
        Args:
            data: Dictionary containing agent response data
            
        Returns:
            AgentResponse instance
        """
        # Handle nested SearchResult and ToolResult objects
        if 'sources' in data:
            data = data.copy()
            data['sources'] = [
                SearchResult.from_dict(sr) if isinstance(sr, dict) else sr
                for sr in data['sources']
            ]
        if 'tool_calls_made' in data:
            if 'sources' not in data:
                data = data.copy()
            data['tool_calls_made'] = [
                ToolResult.from_dict(tr) if isinstance(tr, dict) else tr
                for tr in data['tool_calls_made']
            ]
        return cls(**data)


class ExecutionResult(BaseModel):
    """
    Result of executing an execution plan.
    
    This model represents the overall result of executing a complete execution plan,
    including all tool results and final status information.
    
    Attributes:
        plan: The execution plan that was executed
        tool_results: Results from all tool executions
        status: Overall execution status ("completed", "failed", "partial")
        total_execution_time: Total time taken for the entire execution
        error: Error message if execution failed
    """
    
    plan: ExecutionPlan = Field(
        ...,
        description="The execution plan that was executed"
    )
    tool_results: List[ToolResult] = Field(
        default_factory=list,
        description="Results from all tool executions"
    )
    status: str = Field(
        ...,
        description="Overall execution status ('completed', 'failed', 'partial')"
    )
    total_execution_time: float = Field(
        ...,
        description="Total time taken for the entire execution"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if execution failed"
    )
    
    @validator('status')
    def validate_status(cls, v):
        """Validate status is valid."""
        if not v or not v.strip():
            raise ValueError("Status cannot be empty")
        
        valid_statuses = {"completed", "failed", "partial"}
        status_lower = v.strip().lower()
        
        if status_lower not in valid_statuses:
            raise ValueError(
                f"Status must be one of {valid_statuses}, got '{v}'"
            )
        
        return status_lower
    
    @validator('total_execution_time')
    def validate_total_execution_time(cls, v):
        """Validate total execution time is non-negative."""
        if not isinstance(v, (int, float)):
            raise ValueError("Total execution time must be a number")
        if v < 0:
            raise ValueError("Total execution time cannot be negative")
        return float(v)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the execution result to a dictionary representation.
        
        Returns:
            Dictionary representation of the execution result
        """
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionResult':
        """
        Create an ExecutionResult instance from a dictionary.
        
        Args:
            data: Dictionary containing execution result data
            
        Returns:
            ExecutionResult instance
        """
        # Handle nested ExecutionPlan and ToolResult objects
        if 'plan' in data:
            data = data.copy()
            if isinstance(data['plan'], dict):
                data['plan'] = ExecutionPlan.from_dict(data['plan'])
        if 'tool_results' in data:
            if 'plan' not in data:
                data = data.copy()
            data['tool_results'] = [
                ToolResult.from_dict(tr) if isinstance(tr, dict) else tr
                for tr in data['tool_results']
            ]
        return cls(**data)