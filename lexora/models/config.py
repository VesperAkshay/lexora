"""
Configuration models for the Lexora Agentic RAG SDK.

This module contains configuration classes for different components of the system
including LLM providers, vector databases, and the main agent configuration.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, validator


class LLMConfig(BaseModel):
    """
    Configuration for LLM provider.
    
    This model defines the configuration parameters for connecting to and using
    Large Language Model providers through the litellm interface.
    
    Attributes:
        provider: The LLM provider name (defaults to "litellm")
        model: The specific model to use (e.g., "gpt-3.5-turbo", "claude-3-sonnet")
        api_key: Optional API key for the provider
        temperature: Sampling temperature for response generation (0.0-2.0)
        max_tokens: Maximum number of tokens in the response
        additional_params: Additional provider-specific parameters
    """
    
    provider: str = Field(
        default="litellm",
        description="The LLM provider name"
    )
    model: str = Field(
        ...,
        description="The specific model to use (e.g., 'gpt-3.5-turbo', 'claude-3-sonnet')"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Optional API key for the provider"
    )
    temperature: float = Field(
        default=0.7,
        description="Sampling temperature for response generation (0.0-2.0)"
    )
    max_tokens: int = Field(
        default=2000,
        description="Maximum number of tokens in the response"
    )
    additional_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provider-specific parameters"
    )
    
    @validator('provider')
    def validate_provider(cls, v):
        """Validate that provider is not empty."""
        if not v or not v.strip():
            raise ValueError("Provider cannot be empty")
        return v.strip()
    
    @validator('model')
    def validate_model(cls, v):
        """Validate that model is not empty."""
        if not v or not v.strip():
            raise ValueError("Model cannot be empty")
        return v.strip()
    
    @validator('temperature')
    def validate_temperature(cls, v):
        """Validate temperature is within valid range."""
        if not isinstance(v, (int, float)):
            raise ValueError("Temperature must be a number")
        if v < 0.0 or v > 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return float(v)
    
    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        """Validate max_tokens is positive."""
        if not isinstance(v, int):
            raise ValueError("Max tokens must be an integer")
        if v <= 0:
            raise ValueError("Max tokens must be positive")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the LLM configuration to a dictionary representation.
        try:
            dimension = int(os.getenv(f"{prefix}DIMENSION", "1536"))
        except ValueError as e:
            raise ValueError(f"Invalid value for {prefix}DIMENSION: {e}")
        
        return cls(
            provider=provider,
            connection_params=connection_params,
            embedding_model=os.getenv(f"{prefix}EMBEDDING_MODEL", "text-embedding-ada-002"),
            dimension=dimension
        )        Returns:
            Dictionary representation of the LLM configuration
        """
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMConfig':
        """
        Create an LLMConfig instance from a dictionary.
        
        Args:
            data: Dictionary containing LLM configuration data
            
        Returns:
            LLMConfig instance
        """
        return cls(**data)
    
    @classmethod
    def from_env(cls, prefix: str = "LEXORA_LLM_") -> 'LLMConfig':
        """
        Create an LLMConfig instance from environment variables.
        
        Environment variables:
        - {prefix}PROVIDER: LLM provider name (default: "litellm")
        - {prefix}MODEL: Model name (required)
        - {prefix}API_KEY: API key for the provider
        - {prefix}TEMPERATURE: Temperature value (default: 0.7)
        - {prefix}MAX_TOKENS: Max tokens (default: 2000)
        
        Args:
            prefix: Prefix for environment variable names
            
        Returns:
            LLMConfig instance
            
        Raises:
            ValueError: If required environment variables are missing
        """
        model = os.getenv(f"{prefix}MODEL")
        if not model:
            raise ValueError(f"Environment variable {prefix}MODEL is required")
        
        return cls(
            provider=os.getenv(f"{prefix}PROVIDER", "litellm"),
            model=model,
            api_key=os.getenv(f"{prefix}API_KEY"),
        try:
            temperature = float(os.getenv(f"{prefix}TEMPERATURE", "0.7"))
        except ValueError as e:
            raise ValueError(f"Invalid value for {prefix}TEMPERATURE: {e}")

        try:
            max_tokens = int(os.getenv(f"{prefix}MAX_TOKENS", "2000"))
        except ValueError as e:
            raise ValueError(f"Invalid value for {prefix}MAX_TOKENS: {e}")

        return cls(
            provider=os.getenv(f"{prefix}PROVIDER", "litellm"),
            model=model,
            api_key=os.getenv(f"{prefix}API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens
        )        )


class VectorDBConfig(BaseModel):
    """
    Configuration for vector database.
    
    This model defines the configuration parameters for connecting to and using
    vector database providers like FAISS, Pinecone, or Chroma.
    
    Attributes:
        provider: The vector database provider ("faiss", "pinecone", "chroma")
        connection_params: Provider-specific connection parameters
        embedding_model: The embedding model to use for vector generation
        dimension: The dimension of the embedding vectors
    """
    
    provider: str = Field(
        ...,
        description="The vector database provider ('faiss', 'pinecone', 'chroma')"
    )
    connection_params: Dict[str, Any] = Field(
        ...,
        description="Provider-specific connection parameters"
    )
    embedding_model: str = Field(
        default="text-embedding-ada-002",
        description="The embedding model to use for vector generation"
    )
    dimension: int = Field(
        default=1536,
        description="The dimension of the embedding vectors"
    )
    
    @validator('provider')
    def validate_provider(cls, v):
        """Validate that provider is supported."""
        if not v or not v.strip():
            raise ValueError("Provider cannot be empty")
        
        supported_providers = {"faiss", "pinecone", "chroma"}
        provider_lower = v.strip().lower()
        
        if provider_lower not in supported_providers:
            raise ValueError(
                f"Provider must be one of {supported_providers}, got '{v}'"
            )
        
        return provider_lower
    
    @validator('connection_params')
    def validate_connection_params(cls, v):
        """Validate connection parameters are provided."""
        if not isinstance(v, dict):
            raise ValueError("Connection params must be a dictionary")
        if not v:
            raise ValueError("Connection params cannot be empty")
        return v
    
    @validator('embedding_model')
    def validate_embedding_model(cls, v):
        """Validate that embedding model is not empty."""
        if not v or not v.strip():
            raise ValueError("Embedding model cannot be empty")
        return v.strip()
    
    @validator('dimension')
    def validate_dimension(cls, v):
        """Validate dimension is positive."""
        if not isinstance(v, int):
            raise ValueError("Dimension must be an integer")
        if v <= 0:
            raise ValueError("Dimension must be positive")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the vector DB configuration to a dictionary representation.
        
        Returns:
            Dictionary representation of the vector DB configuration
        """
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VectorDBConfig':
        """
        Create a VectorDBConfig instance from a dictionary.
        
        Args:
            data: Dictionary containing vector DB configuration data
            
        Returns:
            VectorDBConfig instance
        """
        return cls(**data)
    
    @classmethod
    def from_env(cls, prefix: str = "LEXORA_VECTORDB_") -> 'VectorDBConfig':
        """
        Create a VectorDBConfig instance from environment variables.
        
        Environment variables:
        - {prefix}PROVIDER: Vector DB provider (required: faiss, pinecone, chroma)
        - {prefix}EMBEDDING_MODEL: Embedding model name (default: text-embedding-ada-002)
        - {prefix}DIMENSION: Vector dimension (default: 1536)
        - {prefix}CONNECTION_PARAMS: JSON string of connection parameters (required)
        
        Args:
            prefix: Prefix for environment variable names
            
        Returns:
            VectorDBConfig instance
            
        Raises:
            ValueError: If required environment variables are missing
        """
        provider = os.getenv(f"{prefix}PROVIDER")
        if not provider:
            raise ValueError(f"Environment variable {prefix}PROVIDER is required")
        
        connection_params_str = os.getenv(f"{prefix}CONNECTION_PARAMS")
        if not connection_params_str:
            raise ValueError(f"Environment variable {prefix}CONNECTION_PARAMS is required")
        
        try:
            connection_params = json.loads(connection_params_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {prefix}CONNECTION_PARAMS: {e}")
        
        return cls(
            provider=provider,
            connection_params=connection_params,
            embedding_model=os.getenv(f"{prefix}EMBEDDING_MODEL", "text-embedding-ada-002"),
            dimension=int(os.getenv(f"{prefix}DIMENSION", "1536"))
        )


class AgentConfig(BaseModel):
    """
    Configuration for RAG agent.
    
    This model defines the main configuration parameters for the RAG agent
    including context management, tool execution limits, and system behavior.
    
    Attributes:
        max_context_length: Maximum length of context to maintain
        max_tool_calls: Maximum number of tool calls per query
        reasoning_model: Optional specific model for reasoning tasks
        enable_memory: Whether to enable conversation memory
        log_level: Logging level for the agent
    """
    
    max_context_length: int = Field(
        default=8000,
        description="Maximum length of context to maintain"
    )
    max_tool_calls: int = Field(
        default=10,
        description="Maximum number of tool calls per query"
    )
    reasoning_model: Optional[str] = Field(
        default=None,
        description="Optional specific model for reasoning tasks"
    )
    enable_memory: bool = Field(
        default=True,
        description="Whether to enable conversation memory"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level for the agent"
    )
    
    @validator('max_context_length')
    def validate_max_context_length(cls, v):
        """Validate max context length is positive."""
        if not isinstance(v, int):
            raise ValueError("Max context length must be an integer")
        if v <= 0:
            raise ValueError("Max context length must be positive")
        return v
    
    @validator('max_tool_calls')
    def validate_max_tool_calls(cls, v):
        """Validate max tool calls is positive."""
        if not isinstance(v, int):
            raise ValueError("Max tool calls must be an integer")
        if v <= 0:
            raise ValueError("Max tool calls must be positive")
        return v
    
    @validator('reasoning_model')
    def validate_reasoning_model(cls, v):
        """Validate reasoning model if provided."""
        if v is not None and (not v or not v.strip()):
            raise ValueError("Reasoning model cannot be empty string")
        return v.strip() if v else None
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level is supported."""
        if not v or not v.strip():
            raise ValueError("Log level cannot be empty")
        
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        level_upper = v.strip().upper()
        
        if level_upper not in valid_levels:
            raise ValueError(
                f"Log level must be one of {valid_levels}, got '{v}'"
            )
        
        return level_upper
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the agent configuration to a dictionary representation.
        
        Returns:
            Dictionary representation of the agent configuration
        """
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfig':
        """
        Create an AgentConfig instance from a dictionary.
        
        Args:
            data: Dictionary containing agent configuration data
            
        Returns:
            AgentConfig instance
        """
        return cls(**data)
    
    @classmethod
    def from_env(cls, prefix: str = "LEXORA_AGENT_") -> 'AgentConfig':
        """
        Create an AgentConfig instance from environment variables.
        
        Environment variables:
        - {prefix}MAX_CONTEXT_LENGTH: Max context length (default: 8000)
        - {prefix}MAX_TOOL_CALLS: Max tool calls per query (default: 10)
        - {prefix}REASONING_MODEL: Optional reasoning model name
        - {prefix}ENABLE_MEMORY: Enable memory (default: true)
        - {prefix}LOG_LEVEL: Logging level (default: INFO)
        
        Args:
            prefix: Prefix for environment variable names
            
        Returns:
            AgentConfig instance
        """
        enable_memory_str = os.getenv(f"{prefix}ENABLE_MEMORY", "true").lower()
        enable_memory = enable_memory_str in ("true", "1", "yes", "on")
        
        return cls(
        try:
            max_context_length = int(os.getenv(f"{prefix}MAX_CONTEXT_LENGTH", "8000"))
        except ValueError as e:
            raise ValueError(f"Invalid value for {prefix}MAX_CONTEXT_LENGTH: {e}")

        try:
            max_tool_calls = int(os.getenv(f"{prefix}MAX_TOOL_CALLS", "10"))
        except ValueError as e:
            raise ValueError(f"Invalid value for {prefix}MAX_TOOL_CALLS: {e}")

        return cls(
            max_context_length=max_context_length,
            max_tool_calls=max_tool_calls,
            reasoning_model=os.getenv(f"{prefix}REASONING_MODEL"),
            enable_memory=enable_memory,
            log_level=os.getenv(f"{prefix}LOG_LEVEL", "INFO")
        )            reasoning_model=os.getenv(f"{prefix}REASONING_MODEL"),
            enable_memory=enable_memory,
            log_level=os.getenv(f"{prefix}LOG_LEVEL", "INFO")
        )



class RAGAgentConfig(BaseModel):
    """
    Complete configuration for RAGAgent including all components.
    
    This model combines LLM, vector DB, and agent configurations into a single
    configuration object that can be loaded from files or environment variables.
    
    Attributes:
        llm: LLM provider configuration
        vector_db: Vector database configuration
        agent: Agent behavior configuration
    """
    
    llm: LLMConfig = Field(
        ...,
        description="LLM provider configuration"
    )
    vector_db: VectorDBConfig = Field(
        ...,
        description="Vector database configuration"
    )
    agent: AgentConfig = Field(
        default_factory=AgentConfig,
        description="Agent behavior configuration"
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the complete configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            "llm": self.llm.to_dict(),
            "vector_db": self.vector_db.to_dict(),
            "agent": self.agent.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RAGAgentConfig':
        """
        Create a RAGAgentConfig instance from a dictionary.
        
        Args:
            data: Dictionary containing complete configuration
            
        Returns:
            RAGAgentConfig instance
        """
        return cls(
            llm=LLMConfig.from_dict(data.get("llm", {})),
            vector_db=VectorDBConfig.from_dict(data.get("vector_db", {})),
            agent=AgentConfig.from_dict(data.get("agent", {}))
        )
    
    @classmethod
    def from_yaml(cls, file_path: str) -> 'RAGAgentConfig':
        """
        Load configuration from a YAML file.
        
        Args:
            file_path: Path to YAML configuration file
            
        Returns:
            RAGAgentConfig instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is invalid
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            return cls.from_dict(data)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {file_path}: {e}")
    
    @classmethod
    def from_json(cls, file_path: str) -> 'RAGAgentConfig':
        """
        Load configuration from a JSON file.
        
        Args:
            file_path: Path to JSON configuration file
            
        Returns:
            RAGAgentConfig instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {file_path}: {e}")
    
    @classmethod
    def from_env(cls) -> 'RAGAgentConfig':
        """
        Load configuration from environment variables.
        
        Uses standard prefixes:
        - LEXORA_LLM_* for LLM configuration
        - LEXORA_VECTORDB_* for vector DB configuration
        - LEXORA_AGENT_* for agent configuration
        
        Returns:
            RAGAgentConfig instance
            
        Raises:
            ValueError: If required environment variables are missing
        """
        return cls(
            llm=LLMConfig.from_env(),
            vector_db=VectorDBConfig.from_env(),
            agent=AgentConfig.from_env()
        )
    
    def save_yaml(self, file_path: str) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            file_path: Path where to save the configuration
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def save_json(self, file_path: str, indent: int = 2) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            file_path: Path where to save the configuration
            indent: Indentation level for JSON formatting
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)
