"""
Lexora - Agentic RAG SDK

A production-ready, plug-and-play Python SDK for building intelligent RAG systems
with minimal configuration. Features an agentic layer that can reason, plan, and
execute tasks using multiple LLMs and vector databases.

Quick Start:
    >>> from lexora import RAGAgent
    >>> agent = RAGAgent()
    >>> response = await agent.query("What is machine learning?")
    >>> print(response.answer)

For more examples, see the documentation at https://github.com/VesperAkshay/lexora
"""

# Core Agent
from .rag_agent.agent import RAGAgent, create_rag_agent, AgentResponse

# Configuration Models
from .models.config import (
    LLMConfig,
    VectorDBConfig,
    AgentConfig,
    RAGAgentConfig,
)

# Data Models
from .models.core import (
    Document,
    SearchResult,
    CorpusInfo,
)

# Response Models
from .models.responses import (
    QueryIntent,
    ToolCall,
    ExecutionPlan,
    ToolResult,
)

# Exceptions
from .exceptions import (
    LexoraError,
    ErrorCode,
    ConfigurationError,
    LLMError,
    VectorDBError,
    ToolExecutionError,
    PlanningError,
)

# Base Classes for Extension
from .tools.base_tool import BaseTool, ToolParameter, ToolStatus
from .llm.base_llm import BaseLLM
from .vector_db.base_vector_db import BaseVectorDB

# Utilities
from .utils.logging import setup_logging, get_logger

__version__ = "0.1.0"
__author__ = "Lexora Team"
__license__ = "MIT"
__description__ = "Production-ready Agentic RAG SDK with minimal configuration"

# Public API
__all__ = [
    # Core
    "RAGAgent",
    "create_rag_agent",
    "AgentResponse",
    
    # Configuration
    "LLMConfig",
    "VectorDBConfig",
    "AgentConfig",
    "RAGAgentConfig",
    
    # Data Models
    "Document",
    "SearchResult",
    "CorpusInfo",
    
    # Response Models
    "QueryIntent",
    "ToolCall",
    "ExecutionPlan",
    "ToolResult",
    
    # Exceptions
    "LexoraError",
    "ErrorCode",
    "ConfigurationError",
    "LLMError",
    "VectorDBError",
    "ToolExecutionError",
    "PlanningError",
    
    # Base Classes
    "BaseTool",
    "ToolParameter",
    "ToolStatus",
    "BaseLLM",
    "BaseVectorDB",
    
    # Utilities
    "setup_logging",
    "get_logger",
    
    # Package Info
    "get_version",
    "get_info",
]

def get_version() -> str:
    """Get the current version of the Lexora SDK."""
    return __version__


def get_info() -> dict:
    """Get information about the Lexora SDK."""
    return {
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "description": __description__,
    }
