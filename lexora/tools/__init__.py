"""
Tool system for the Lexora Agentic RAG SDK.

This module provides the tool registry system and base classes for implementing
custom tools in the RAG agent system.
"""

import importlib
import inspect
import os
from typing import Any, Dict, List, Optional, Type, Union, Callable
from pathlib import Path
import asyncio
from datetime import datetime
from collections import defaultdict

from .base_tool import BaseTool, ToolResult, ToolParameter, ParameterType, validate_tool_interface
from .create_corpus import CreateCorpusTool, create_corpus_tool
from .add_data import AddDataTool, create_add_data_tool
from .rag_query import RAGQueryTool, create_rag_query_tool
from .list_corpora import ListCorporaTool, create_list_corpora_tool
from .get_corpus_info import GetCorpusInfoTool, create_get_corpus_info_tool
from .delete_document import DeleteDocumentTool, create_delete_document_tool
from .delete_corpus import DeleteCorpusTool, create_delete_corpus_tool
from .update_document import UpdateDocumentTool, create_update_document_tool
from .bulk_add_data import BulkAddDataTool, create_bulk_add_data_tool
from .health_check import HealthCheckTool, create_health_check_tool
from ..exceptions import LexoraError, ErrorCode, create_tool_error
from ..utils.logging import get_logger


class ToolRegistry:
    """
    Registry for managing and discovering RAG tools.
    
    This class provides centralized management of tools, including registration,
    discovery, validation, and execution coordination.
    """
    
    def __init__(self):
        """Initialize the tool registry."""
        self.logger = get_logger(self.__class__.__name__)
        self._tools: Dict[str, BaseTool] = {}
        self._tool_metadata: Dict[str, Dict[str, Any]] = {}
        self._tool_classes: Dict[str, Type[BaseTool]] = {}
        self._categories: Dict[str, List[str]] = defaultdict(list)
        self._aliases: Dict[str, str] = {}
        
    def register_tool(
        self,
        tool: Union[BaseTool, Type[BaseTool]],
        category: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """
        Register a tool with the registry.
        
        Args:
            tool: Tool instance or class to register
            category: Optional category for organization
            aliases: Optional list of alternative names
            **kwargs: Additional configuration for tool instantiation
            
        Raises:
            LexoraError: If registration fails
        """
        try:
            # Handle both instances and classes
            if inspect.isclass(tool):
                validate_tool_interface(tool)
                tool_instance = tool(**kwargs)
                self._tool_classes[tool_instance.name] = tool
            else:
                tool_instance = tool
                self._tool_classes[tool_instance.name] = tool.__class__
            
            # Validate tool instance
            if not isinstance(tool_instance, BaseTool):
                raise create_tool_error(
                    f"Tool must be an instance of BaseTool, got {type(tool_instance)}",
                    "registry",
                    ErrorCode.INVALID_CONFIG
                )
            
            # Check for name conflicts
            if tool_instance.name in self._tools:
                raise create_tool_error(
                    f"Tool with name '{tool_instance.name}' is already registered",
                    tool_instance.name,
                    ErrorCode.TOOL_VALIDATION_FAILED
                )
            
            # Register the tool
            self._tools[tool_instance.name] = tool_instance
            self._tool_metadata[tool_instance.name] = tool_instance.get_metadata()
            
            # Handle category
            if category:
                self._categories[category].append(tool_instance.name)
            
            # Handle aliases
            if aliases:
                for alias in aliases:
                    if alias in self._aliases:
                        raise create_tool_error(
                            f"Alias '{alias}' is already in use",
                            tool_instance.name,
                            ErrorCode.TOOL_VALIDATION_FAILED
                        )
                    self._aliases[alias] = tool_instance.name
            
            self.logger.info(f"Registered tool: {tool_instance.name} (version {tool_instance.version})")
            
        except Exception as e:
            if isinstance(e, LexoraError):
                raise
            raise create_tool_error(
                f"Failed to register tool: {str(e)}",
                "registry",
                ErrorCode.TOOL_VALIDATION_FAILED
            )
    
    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool from the registry.
        
        Args:
            name: Name of the tool to unregister
            
        Returns:
            True if tool was unregistered, False if not found
        """
        if name not in self._tools:
            return False
        
        # Remove from main registry
        del self._tools[name]
        del self._tool_metadata[name]
        del self._tool_classes[name]
        
        # Remove from categories
        for category, tools in self._categories.items():
            if name in tools:
                tools.remove(name)
        
        # Remove aliases
        aliases_to_remove = [alias for alias, tool_name in self._aliases.items() if tool_name == name]
        for alias in aliases_to_remove:
            del self._aliases[alias]
        
        self.logger.info(f"Unregistered tool: {name}")
        return True
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool by name or alias.
        
        Args:
            name: Tool name or alias
            
        Returns:
            Tool instance if found, None otherwise
        """
        # Check direct name
        if name in self._tools:
            return self._tools[name]
        
        # Check aliases
        if name in self._aliases:
            return self._tools[self._aliases[name]]
        
        return None
    
    def has_tool(self, name: str) -> bool:
        """
        Check if a tool is registered.
        
        Args:
            name: Tool name or alias
            
        Returns:
            True if tool exists, False otherwise
        """
        return name in self._tools or name in self._aliases
    
    def list_tools(self, category: Optional[str] = None) -> List[str]:
        """
        List registered tool names.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of tool names
        """
        if category:
            return self._categories.get(category, [])
        return list(self._tools.keys())
    
    def list_categories(self) -> List[str]:
        """
        List available tool categories.
        
        Returns:
            List of category names
        """
        return list(self._categories.keys())
    
    def get_tool_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a tool.
        
        Args:
            name: Tool name or alias
            
        Returns:
            Tool metadata if found, None otherwise
        """
        tool_name = self._aliases.get(name, name)
        return self._tool_metadata.get(tool_name)
    
    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for all registered tools.
        
        Returns:
            Dictionary mapping tool names to their metadata
        """
        return self._tool_metadata.copy()
    
    async def execute_tool(self, name: str, **kwargs) -> ToolResult:
        """
        Execute a tool by name.
        
        Args:
            name: Tool name or alias
            **kwargs: Parameters for tool execution
            
        Returns:
            ToolResult from tool execution
            
        Raises:
            LexoraError: If tool not found or execution fails
        """
        tool = self.get_tool(name)
        if not tool:
            raise create_tool_error(
                f"Tool '{name}' not found in registry",
                name,
                ErrorCode.TOOL_NOT_FOUND
            )
        
        return await tool.run(**kwargs)
    
    def search_tools(
        self,
        query: str,
        category: Optional[str] = None,
        fuzzy: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for tools by name or description.
        
        Args:
            query: Search query
            category: Optional category filter
            fuzzy: Whether to perform fuzzy matching
            
        Returns:
            List of matching tool metadata with relevance scores
        """
        results = []
        query_lower = query.lower()
        
        tools_to_search = self.list_tools(category) if category else self.list_tools()
        
        for tool_name in tools_to_search:
            metadata = self._tool_metadata[tool_name]
            score = 0
            
            # Exact name match
            if query_lower == tool_name.lower():
                score += 100
            # Name contains query
            elif query_lower in tool_name.lower():
                score += 50
            
            # Description contains query
            if query_lower in metadata.get('description', '').lower():
                score += 25
            
            # Fuzzy matching on name
            if fuzzy and self._fuzzy_match(query_lower, tool_name.lower()):
                score += 10
            
            if score > 0:
                results.append({
                    'tool': tool_name,
                    'metadata': metadata,
                    'score': score
                })
        
        # Sort by relevance score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def _fuzzy_match(self, query: str, target: str, threshold: float = 0.6) -> bool:
        """
        Perform fuzzy string matching.
        
        Args:
            query: Query string
            target: Target string
            threshold: Minimum similarity threshold
            
        Returns:
            True if strings are similar enough
        """
        # Simple fuzzy matching using character overlap
        if len(query) == 0 or len(target) == 0:
            return False
        
        # Calculate character overlap ratio
        query_chars = set(query)
        target_chars = set(target)
        overlap = len(query_chars & target_chars)
        total = len(query_chars | target_chars)
        
        similarity = overlap / total if total > 0 else 0
        return similarity >= threshold
    
    def load_tools_from_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        category: Optional[str] = None
    ) -> List[str]:
        """
        Load tools from a directory.
        
        Args:
            directory: Directory path to scan
            recursive: Whether to scan subdirectories
            category: Category to assign to loaded tools
            
        Returns:
            List of loaded tool names
            
        Raises:
            LexoraError: If loading fails
        """
        directory = Path(directory)
        if not directory.exists():
            raise create_tool_error(
                f"Directory not found: {directory}",
                "registry",
                ErrorCode.INVALID_CONFIG
            )
        
        loaded_tools = []
        
        # Find Python files
        pattern = "**/*.py" if recursive else "*.py"
        for py_file in directory.glob(pattern):
            if py_file.name.startswith('_'):
                continue  # Skip private files
            
            try:
                # Load module
                module_name = py_file.stem
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find tool classes
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, BaseTool) and 
                            obj != BaseTool and 
                            not inspect.isabstract(obj)):
                            
                            try:
                            try:
                                self.register_tool(obj, category=category)
                                # Get the registered tool's name
                                tool = self.get_tool(obj().name)
                                if tool:
                     except Exception as e:
                         tool_name = getattr(temp_instance, 'name', name) if 'temp_instance' in locals() else name
                         self.logger.warning(f"Failed to register tool {tool_name}: {str(e)}")
             except Exception as e:
                 self.logger.warning(f"Failed to load module {py_file}: {str(e)}")                                self.logger.warning(f"Failed to register tool {name}: {str(e)}")                                self.logger.warning(f"Failed to register tool {tool_name}: {str(e)}")            except Exception as e:                self.logger.warning(f"Failed to load module {py_file}: {str(e)}")
        
        self.logger.info(f"Loaded {len(loaded_tools)} tools from {directory}")
        return loaded_tools
    
    def load_tools_from_module(
        self,
        module_name: str,
        category: Optional[str] = None
    ) -> List[str]:
        """
        Load tools from a Python module.
        
        Args:
            module_name: Name of module to import
            category: Category to assign to loaded tools
            
        Returns:
            List of loaded tool names
            
        Raises:
            LexoraError: If loading fails
        """
        try:
            module = importlib.import_module(module_name)
            loaded_tools = []
            
            # Find tool classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, BaseTool) and 
                try:
                    validate_tool_interface(obj)
                    # Instantiate once and register the instance
                    tool_instance = obj()
                    self.register_tool(tool_instance, category=category)
                    loaded_tools.append(tool_instance.name)
                except Exception as e:
                    self.logger.warning(f"Failed to register tool {name}: {str(e)}")                         loaded_tools.append(tool_instance.name)
                    except Exception as e:
                        self.logger.warning(f"Failed to register tool {name}: {str(e)}")            
            self.logger.info(f"Loaded {len(loaded_tools)} tools from module {module_name}")
            return loaded_tools
            
        except ImportError as e:
            raise create_tool_error(
                f"Failed to import module {module_name}: {str(e)}",
                "registry",
                ErrorCode.INVALID_CONFIG
            )
    
    def validate_all_tools(self) -> Dict[str, bool]:
        """
        Validate all registered tools.
        
        Returns:
            Dictionary mapping tool names to validation results
        """
        results = {}
        
        for tool_name, tool in self._tools.items():
            try:
                # Validate interface
                validate_tool_interface(tool.__class__)
                
                # Validate schema
                schema = tool.get_schema()
                if not isinstance(schema, dict):
                    results[tool_name] = False
                    continue
                
                # Validate metadata
                metadata = tool.get_metadata()
                required_fields = ['name', 'description', 'version', 'schema']
                if not all(field in metadata for field in required_fields):
                    results[tool_name] = False
                    continue
                
                results[tool_name] = True
                
            except Exception as e:
                self.logger.error(f"Validation failed for tool {tool_name}: {str(e)}")
                results[tool_name] = False
        
        return results
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the registry.
        
        Returns:
            Dictionary containing registry statistics
        """
        return {
            "total_tools": len(self._tools),
            "categories": len(self._categories),
            "aliases": len(self._aliases),
            "tools_by_category": {cat: len(tools) for cat, tools in self._categories.items()},
            "validation_results": self.validate_all_tools()
        }
    
    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        self._tool_metadata.clear()
        self._tool_classes.clear()
        self._categories.clear()
        self._aliases.clear()
        self.logger.info("Registry cleared")


# Global registry instance
_global_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """
    Get the global tool registry instance.
    
    Returns:
        Global ToolRegistry instance
    """
    return _global_registry


def register_tool(
    tool: Union[BaseTool, Type[BaseTool]],
    category: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    **kwargs
) -> None:
    """
    Register a tool with the global registry.
    
    Args:
        tool: Tool instance or class to register
        category: Optional category for organization
        aliases: Optional list of alternative names
        **kwargs: Additional configuration for tool instantiation
    """
    _global_registry.register_tool(tool, category, aliases, **kwargs)


def get_tool(name: str) -> Optional[BaseTool]:
    """
    Get a tool from the global registry.
    
    Args:
        name: Tool name or alias
        
    Returns:
        Tool instance if found, None otherwise
    """
    return _global_registry.get_tool(name)


def list_tools(category: Optional[str] = None) -> List[str]:
    """
    List tools in the global registry.
    
    Args:
        category: Optional category filter
        
    Returns:
        List of tool names
    """
    return _global_registry.list_tools(category)


async def execute_tool(name: str, **kwargs) -> ToolResult:
    """
    Execute a tool from the global registry.
    
    Args:
        name: Tool name or alias
        **kwargs: Parameters for tool execution
        
    Returns:
        ToolResult from tool execution
    """
    return await _global_registry.execute_tool(name, **kwargs)


# Export public interface
__all__ = [
    # Base classes
    "BaseTool",
    "ToolResult",
    "ToolParameter",
    "ParameterType",
    
    # Registry classes
    "ToolRegistry",
    
    # Global registry functions
    "get_registry",
    "register_tool",
    "get_tool",
    "list_tools",
    "execute_tool",
    
    # Utility functions
    "validate_tool_interface",
    
    # RAG Tools
    "CreateCorpusTool",
    "AddDataTool",
    "RAGQueryTool",
    "ListCorporaTool",
    "GetCorpusInfoTool",
    "DeleteDocumentTool",
    "DeleteCorpusTool",
    "UpdateDocumentTool",
    "BulkAddDataTool",
    "HealthCheckTool",
    "create_corpus_tool",
    "create_add_data_tool",
    "create_rag_query_tool",
    "create_list_corpora_tool",
    "create_get_corpus_info_tool",
    "create_delete_document_tool",
    "create_delete_corpus_tool",
    "create_update_document_tool",
    "create_bulk_add_data_tool",
    "create_health_check_tool",
]