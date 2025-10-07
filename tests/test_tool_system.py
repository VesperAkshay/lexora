"""
Unit tests for the tool system.

This module tests the base tool interface and tool registry system.
"""

import asyncio
import pytest
from typing import Dict, Any

from tools.base_tool import (
    BaseTool, ToolResult, ToolParameter, ParameterType, ToolStatus,
    validate_tool_interface, create_tool_parameter
)
from tools import ToolRegistry, get_registry, register_tool, get_tool, list_tools
from exceptions import LexoraError


class MockTool(BaseTool):
    """Mock tool for testing."""
    
    @property
    def name(self) -> str:
        return "mock_tool"
    
    @property
    def description(self) -> str:
        return "A mock tool for testing purposes"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def _setup_parameters(self) -> None:
        self._parameters = [
            create_tool_parameter(
                "text",
                ParameterType.STRING,
                "Text to process",
                required=True
            ),
            create_tool_parameter(
                "count",
                ParameterType.INTEGER,
                "Number of times to repeat",
                default=1,
                minimum=1,
                maximum=10
            ),
            create_tool_parameter(
                "uppercase",
                ParameterType.BOOLEAN,
                "Whether to convert to uppercase",
                default=False
            )
        ]
    
    async def _execute(self, **kwargs) -> Dict[str, Any]:
        text = kwargs["text"]
        count = kwargs.get("count", 1)
        uppercase = kwargs.get("uppercase", False)
        
        if uppercase:
            text = text.upper()
        
        result = text * count
        
        return {
            "result": result,
            "original_text": kwargs["text"],
            "processed_count": count,
            "was_uppercased": uppercase
        }


class FailingTool(BaseTool):
    """Tool that always fails for testing error handling."""
    
    @property
    def name(self) -> str:
        return "failing_tool"
    
    @property
    def description(self) -> str:
        return "A tool that always fails"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def _setup_parameters(self) -> None:
        self._parameters = []
    
    async def _execute(self, **kwargs) -> Dict[str, Any]:
        raise LexoraError("This tool always fails")


class TestBaseTool:
    """Test the BaseTool abstract class and functionality."""
    
    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool instance."""
        return MockTool()
    
    @pytest.fixture
    def failing_tool(self):
        """Create a failing tool instance."""
        return FailingTool()
    
    def test_tool_properties(self, mock_tool):
        """Test tool basic properties."""
        assert mock_tool.name == "mock_tool"
        assert mock_tool.description == "A mock tool for testing purposes"
        assert mock_tool.version == "1.0.0"
    
    def test_tool_schema_generation(self, mock_tool):
        """Test tool schema generation."""
        schema = mock_tool.get_schema()
        
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "text" in schema["properties"]
        assert "count" in schema["properties"]
        assert "uppercase" in schema["properties"]
        
        # Check required fields
        assert "required" in schema
        assert "text" in schema["required"]
        assert "count" not in schema["required"]
        
        # Check defaults
        assert schema["properties"]["count"]["default"] == 1
        assert schema["properties"]["uppercase"]["default"] is False
    
    def test_tool_metadata(self, mock_tool):
        """Test tool metadata generation."""
        metadata = mock_tool.get_metadata()
        
        assert metadata["name"] == "mock_tool"
        assert metadata["description"] == "A mock tool for testing purposes"
        assert metadata["version"] == "1.0.0"
        assert "schema" in metadata
        assert "parameters" in metadata
        assert len(metadata["parameters"]) == 3
    
    def test_parameter_validation_success(self, mock_tool):
        """Test successful parameter validation."""
        # Valid parameters
        params = mock_tool.validate_parameters(text="hello", count=3, uppercase=True)
        assert params["text"] == "hello"
        assert params["count"] == 3
        assert params["uppercase"] is True
        
        # With defaults
        params = mock_tool.validate_parameters(text="hello")
        assert params["text"] == "hello"
        assert params["count"] == 1
        assert params["uppercase"] is False
    
    def test_parameter_validation_type_conversion(self, mock_tool):
        """Test parameter type conversion."""
        # String to int conversion
        params = mock_tool.validate_parameters(text="hello", count="3")
        assert params["count"] == 3
        assert isinstance(params["count"], int)
        
        # String to bool conversion
        params = mock_tool.validate_parameters(text="hello", uppercase="true")
        assert params["uppercase"] is True
        
        params = mock_tool.validate_parameters(text="hello", uppercase="false")
        assert params["uppercase"] is False
    
    def test_parameter_validation_errors(self, mock_tool):
        """Test parameter validation errors."""
        # Missing required parameter
        with pytest.raises(LexoraError):
            mock_tool.validate_parameters(count=3)
        
        # Invalid range
        with pytest.raises(LexoraError):
            mock_tool.validate_parameters(text="hello", count=15)  # max is 10
        
        with pytest.raises(LexoraError):
            mock_tool.validate_parameters(text="hello", count=0)  # min is 1
    
    @pytest.mark.asyncio
    async def test_tool_execution_success(self, mock_tool):
        """Test successful tool execution."""
        result = await mock_tool.run(text="hello", count=2, uppercase=True)
        
        assert isinstance(result, ToolResult)
        assert result.status == ToolStatus.SUCCESS
        assert result.data["result"] == "HELLOHELLO"
        assert result.data["original_text"] == "hello"
        assert result.data["processed_count"] == 2
        assert result.data["was_uppercased"] is True
        assert result.execution_time is not None
        assert result.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_tool_execution_with_defaults(self, mock_tool):
        """Test tool execution with default parameters."""
        result = await mock_tool.run(text="world")
        
        assert result.status == ToolStatus.SUCCESS
        assert result.data["result"] == "world"
        assert result.data["processed_count"] == 1
        assert result.data["was_uppercased"] is False
    
    @pytest.mark.asyncio
    async def test_tool_execution_error(self, failing_tool):
        """Test tool execution error handling."""
        result = await failing_tool.run()
        
        assert isinstance(result, ToolResult)
        assert result.status == ToolStatus.ERROR
        assert "This tool always fails" in result.error
        assert result.execution_time is not None
    
    @pytest.mark.asyncio
    async def test_tool_parameter_validation_error(self, mock_tool):
        """Test tool execution with parameter validation error."""
        result = await mock_tool.run(count=15)  # Missing required param and invalid range
        
        assert result.status == ToolStatus.ERROR
        assert "Missing required parameters" in result.error


class TestToolRegistry:
    """Test the ToolRegistry class."""
    
    @pytest.fixture
    def registry(self):
        """Create a fresh tool registry."""
        return ToolRegistry()
    
    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool instance."""
        return MockTool()
    
    def test_registry_initialization(self, registry):
        """Test registry initialization."""
        assert len(registry.list_tools()) == 0
        assert len(registry.list_categories()) == 0
    
    def test_tool_registration_instance(self, registry, mock_tool):
        """Test registering a tool instance."""
        registry.register_tool(mock_tool, category="test")
        
        assert registry.has_tool("mock_tool")
        assert "mock_tool" in registry.list_tools()
        assert "mock_tool" in registry.list_tools("test")
        assert "test" in registry.list_categories()
    
    def test_tool_registration_class(self, registry):
        """Test registering a tool class."""
        registry.register_tool(MockTool, category="test")
        
        assert registry.has_tool("mock_tool")
        tool = registry.get_tool("mock_tool")
        assert isinstance(tool, MockTool)
    
    def test_tool_registration_with_aliases(self, registry, mock_tool):
        """Test registering a tool with aliases."""
        registry.register_tool(mock_tool, aliases=["mock", "test_tool"])
        
        assert registry.has_tool("mock_tool")
        assert registry.has_tool("mock")
        assert registry.has_tool("test_tool")
        
        # All should return the same tool
        tool1 = registry.get_tool("mock_tool")
        tool2 = registry.get_tool("mock")
        tool3 = registry.get_tool("test_tool")
        
        assert tool1 is tool2 is tool3
    
    def test_tool_registration_duplicate_error(self, registry, mock_tool):
        """Test error when registering duplicate tool names."""
        registry.register_tool(mock_tool)
        
        with pytest.raises(LexoraError):
            registry.register_tool(mock_tool)
    
    def test_tool_unregistration(self, registry, mock_tool):
        """Test unregistering tools."""
        registry.register_tool(mock_tool, category="test", aliases=["mock"])
        
        assert registry.has_tool("mock_tool")
        assert registry.has_tool("mock")
        
        success = registry.unregister_tool("mock_tool")
        assert success
        
        assert not registry.has_tool("mock_tool")
        assert not registry.has_tool("mock")
        assert "mock_tool" not in registry.list_tools("test")
    
    def test_tool_unregistration_not_found(self, registry):
        """Test unregistering non-existent tool."""
        success = registry.unregister_tool("nonexistent")
        assert not success
    
    def test_get_tool_metadata(self, registry, mock_tool):
        """Test getting tool metadata."""
        registry.register_tool(mock_tool)
        
        metadata = registry.get_tool_metadata("mock_tool")
        assert metadata is not None
        assert metadata["name"] == "mock_tool"
        assert metadata["version"] == "1.0.0"
    
    @pytest.mark.asyncio
    async def test_execute_tool_via_registry(self, registry, mock_tool):
        """Test executing tool via registry."""
        registry.register_tool(mock_tool)
        
        result = await registry.execute_tool("mock_tool", text="hello", count=2)
        
        assert result.status == ToolStatus.SUCCESS
        assert result.data["result"] == "hellohello"
    
    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self, registry):
        """Test executing non-existent tool."""
        with pytest.raises(LexoraError):
            await registry.execute_tool("nonexistent")
    
    def test_search_tools(self, registry, mock_tool):
        """Test tool search functionality."""
        registry.register_tool(mock_tool, category="test")
        
        # Search by name
        results = registry.search_tools("mock")
        assert len(results) > 0
        assert results[0]["tool"] == "mock_tool"
        
        # Search by description
        results = registry.search_tools("testing")
        assert len(results) > 0
        assert results[0]["tool"] == "mock_tool"
    
    def test_registry_stats(self, registry, mock_tool):
        """Test registry statistics."""
        registry.register_tool(mock_tool, category="test")
        
        stats = registry.get_registry_stats()
        assert stats["total_tools"] == 1
        assert stats["categories"] == 1
        assert "test" in stats["tools_by_category"]
        assert stats["tools_by_category"]["test"] == 1
    
    def test_validate_all_tools(self, registry, mock_tool):
        """Test validating all tools in registry."""
        registry.register_tool(mock_tool)
        
        results = registry.validate_all_tools()
        assert "mock_tool" in results
        assert results["mock_tool"] is True


class TestGlobalRegistry:
    """Test global registry functions."""
    
    def setup_method(self):
        """Clear global registry before each test."""
        get_registry().clear()
    
    def test_global_registry_functions(self):
        """Test global registry convenience functions."""
        # Register tool
        register_tool(MockTool, category="global_test")
        
        # Check registration
        assert "mock_tool" in list_tools()
        assert "mock_tool" in list_tools("global_test")
        
        # Get tool
        tool = get_tool("mock_tool")
        assert isinstance(tool, MockTool)
    
    @pytest.mark.asyncio
    async def test_global_execute_tool(self):
        """Test global tool execution."""
        from tools import execute_tool
        
        register_tool(MockTool)
        
        result = await execute_tool("mock_tool", text="global", count=1)
        assert result.status == ToolStatus.SUCCESS
        assert result.data["result"] == "global"


class TestToolValidation:
    """Test tool validation functions."""
    
    def test_validate_tool_interface_success(self):
        """Test successful tool interface validation."""
        assert validate_tool_interface(MockTool) is True
    
    def test_validate_tool_interface_not_subclass(self):
        """Test validation failure for non-BaseTool class."""
        class NotATool:
            pass
        
        with pytest.raises(LexoraError):
            validate_tool_interface(NotATool)
    
    def test_create_tool_parameter(self):
        """Test tool parameter creation utility."""
        param = create_tool_parameter(
            "test_param",
            ParameterType.STRING,
            "Test parameter",
            required=True,
            default="default_value"
        )
        
        assert param.name == "test_param"
        assert param.type == ParameterType.STRING
        assert param.description == "Test parameter"
        assert param.required is True
        assert param.default == "default_value"


if __name__ == "__main__":
    pytest.main([__file__])