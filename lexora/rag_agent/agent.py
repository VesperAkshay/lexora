"""RAGAgent - Main orchestrator for the Lexora Agentic RAG SDK.

This module implements the main RAGAgent class that coordinates all components
of the agentic RAG system including planning, execution, and reasoning.
"""

import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field

from .planner import AgentPlanner, ExecutionPlan, create_agent_planner
from .executor import AgentExecutor, ExecutionContext, create_agent_executor
from .reasoning import ReasoningEngine, ReasoningResult, ReasoningStrategy, create_reasoning_engine
from ..tools import ToolRegistry
from ..tools.base_tool import BaseTool
from ..llm.base_llm import BaseLLM, MockLLMProvider
from ..llm.litellm_provider import LitellmProvider
from ..vector_db.base_vector_db import BaseVectorDB, MockVectorDB
from ..vector_db.faiss_db import FAISSVectorDB
from ..models.config import LLMConfig, VectorDBConfig, AgentConfig, RAGAgentConfig
from ..exceptions import LexoraError, ErrorCode, create_tool_error
from ..utils.logging import get_logger, setup_logging


@dataclass
class AgentResponse:
    """Response from the RAG agent."""
    answer: str
    confidence: float
    sources: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent response to dictionary."""
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "sources": self.sources,
            "reasoning_chain": self.reasoning_chain,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class RAGAgent:
    """
    Main orchestrator for the agentic RAG system.
    
    The RAGAgent coordinates all components including the planner, executor,
    and reasoning engine to process user queries and generate responses.
    """
    
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        vector_db_config: Optional[VectorDBConfig] = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[List[BaseTool]] = None,
        **kwargs
    ):
        """
        Initialize the RAG agent with required configurations.
        
        Args:
            llm_config: Configuration for LLM provider
            vector_db_config: Configuration for vector database
            agent_config: Configuration for agent behavior
            tools: Optional list of custom tools to register
            **kwargs: Additional configuration options
            
        Raises:
            LexoraError: If configuration validation fails
        """
        self.logger = get_logger(self.__class__.__name__)
        
        try:
            # Validate and store configurations
            self.llm_config = llm_config or self._get_default_llm_config()
            self.vector_db_config = vector_db_config or self._get_default_vector_db_config()
            self.agent_config = agent_config or self._get_default_agent_config()
            
            # Setup logging
            setup_logging(level=self.agent_config.log_level)
            
            self.logger.info("Initializing RAGAgent...")
            
            # Initialize LLM provider
            self.llm = self._initialize_llm(self.llm_config)
            
            # Initialize vector database
            self.vector_db = self._initialize_vector_db(self.vector_db_config)
            
            # Initialize tool registry
            self.tool_registry = ToolRegistry()
            
            # Register default tools
            self._register_default_tools()
            
            # Register custom tools if provided
            if tools:
                for tool in tools:
                    self.add_tool(tool)
            
            # Initialize components
            self.planner = create_agent_planner(
                llm=self.llm,
                tool_registry=self.tool_registry,
                max_plan_steps=self.agent_config.max_tool_calls
            )
            
            self.executor = create_agent_executor(
                tool_registry=self.tool_registry,
                max_context_size=self.agent_config.max_context_length,
                enable_step_retry=True
            )
            
            self.reasoning_engine = create_reasoning_engine(
                llm=self.llm,
                default_strategy=ReasoningStrategy.SYNTHESIS
            )
            
            # Agent state
            self.query_history: List[Dict[str, Any]] = []
            self.is_initialized = True
            
            self.logger.info("RAGAgent initialized successfully")
            
        except Exception as e:
            raise create_tool_error(
                f"Failed to initialize RAGAgent: {str(e)}",
                "rag_agent",
                {"error_type": type(e).__name__},
                ErrorCode.INVALID_CONFIG,
                e
            )
    
    @classmethod
    async def create(
        cls,
        llm_config: Optional[LLMConfig] = None,
        vector_db_config: Optional[VectorDBConfig] = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[List[BaseTool]] = None,
        **kwargs
    ) -> "RAGAgent":
        """
        Async factory method to create and initialize a RAGAgent.
        
        This is the recommended way to create a RAGAgent as it properly
        handles async initialization of the vector database connection.
        
        Args:
            llm_config: Configuration for LLM provider
            vector_db_config: Configuration for vector database
            agent_config: Configuration for agent behavior
            tools: Optional list of custom tools to register
            **kwargs: Additional configuration options
            
        Returns:
            Fully initialized RAGAgent instance
            
        Example:
            agent = await RAGAgent.create(
                llm_config=my_llm_config,
                vector_db_config=my_db_config
            )
        """
        # Create instance using synchronous __init__
        instance = cls(llm_config, vector_db_config, agent_config, tools, **kwargs)
        
        # Perform async initialization
        await instance.initialize()
        
        return instance
    
    async def initialize(self) -> None:
        """
        Perform async initialization tasks.
        
        This method connects to the vector database and performs any other
        async setup required. It's called automatically by the create() factory
        method, but can also be called manually if using the constructor directly.
        
        Example:
            agent = RAGAgent()  # Synchronous construction
            await agent.initialize()  # Async initialization
        """
        try:
            # Connect to vector database if it has a connect method
            if hasattr(self.vector_db, 'connect') and callable(self.vector_db.connect):
                self.logger.info("Connecting to vector database...")
                await self.vector_db.connect()
                self.logger.info("Vector database connected successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize async components: {e}")
            raise create_tool_error(
                f"Failed to initialize async components: {str(e)}",
                "rag_agent",
                {"error_type": type(e).__name__},
                ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                e
            )
    
    async def query(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
        reasoning_strategy: Optional[ReasoningStrategy] = None
    ) -> AgentResponse:
        """
        Process a user query through the agentic pipeline.
        
        This method orchestrates the entire RAG pipeline:
        1. Plan: Analyze query and create execution plan
        2. Execute: Execute the plan using available tools
        3. Reason: Synthesize results into coherent response
        
        Args:
            input_text: User query text
            context: Optional context information
            reasoning_strategy: Optional reasoning strategy to use
            
        Returns:
            AgentResponse: Generated response with metadata
            
        Raises:
            LexoraError: If query processing fails
        """
        import time
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing query: '{input_text[:100]}...'")
            
            # Validate input
            if not input_text or not input_text.strip():
                raise ValueError("Query text cannot be empty")
            
            # Step 1: Create execution plan
            self.logger.info("Step 1: Creating execution plan...")
            plan = await self.planner.create_plan(input_text, context)
            
            self.logger.info(f"Created plan with {len(plan.steps)} steps")
            
            # Step 2: Execute the plan
            self.logger.info("Step 2: Executing plan...")
            execution_context = ExecutionContext(
                plan_id=plan.id,
                max_context_size=self.agent_config.max_context_length
            )
            
            execution_result = await self.executor.execute_plan(plan, execution_context)
            
            self.logger.info(
                f"Plan execution completed (success: {execution_result.success})"
            )
            
            # Step 3: Generate response using reasoning engine
            self.logger.info("Step 3: Generating response...")
            reasoning_result = await self.reasoning_engine.generate_response(
                query=input_text,
                plan=plan,
                execution_result=execution_result,
                context=execution_context,
                strategy=reasoning_strategy
            )
            
            # Calculate total execution time
            execution_time = time.time() - start_time
            
            # Create agent response
            response = AgentResponse(
                answer=reasoning_result.answer,
                confidence=reasoning_result.confidence,
                sources=[source.to_dict() for source in reasoning_result.sources],
                reasoning_chain=reasoning_result.reasoning_chain,
                execution_time=execution_time,
                metadata={
                    "plan_id": plan.id,
                    "steps_executed": len(plan.steps),
                    "reasoning_strategy": reasoning_result.metadata.get("strategy"),
                    "confidence_level": reasoning_result.confidence_level.value,
                    **reasoning_result.metadata
                }
            )
            
            # Add to query history
            self.query_history.append({
                "query": input_text,
                "response": response.to_dict(),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            self.logger.info(
                f"Query processed successfully in {execution_time:.2f}s "
                f"(confidence: {response.confidence:.2f})"
            )
            
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Query processing failed: {e}")
            
            # Return error response
            return AgentResponse(
                answer=f"I encountered an error processing your query: {str(e)}",
                confidence=0.0,
                execution_time=execution_time,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
    
    def add_tool(self, tool: BaseTool, category: Optional[str] = None) -> None:
        """
        Register a custom tool with the agent.
        
        Args:
            tool: Tool instance to register
            category: Optional category for the tool
            
        Raises:
            LexoraError: If tool registration fails
        """
        try:
            self.logger.info(f"Registering tool: {tool.name}")
            
            # Validate tool
            if not hasattr(tool, 'name') or not tool.name:
                raise ValueError("Tool must have a valid name")
            
            if not hasattr(tool, 'run'):
                raise ValueError("Tool must implement run() method")
            
            # Register with tool registry
            self.tool_registry.register_tool(tool, category=category or "custom")
            
            self.logger.info(f"Tool '{tool.name}' registered successfully")
            
        except Exception as e:
            raise create_tool_error(
                f"Failed to register tool: {str(e)}",
                "rag_agent",
                {"tool_name": getattr(tool, 'name', 'unknown')},
                ErrorCode.TOOL_VALIDATION_FAILED,
                e
            )
    
    def get_available_tools(self) -> List[str]:
        """
        Return list of available tool names.
        
        Returns:
            List of tool names registered with the agent
        """
        return self.tool_registry.list_tools()
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Dictionary containing tool information
            
        Raises:
            LexoraError: If tool not found
        """
        try:
            tool = self.tool_registry.get_tool(tool_name)
            schema = tool.get_schema()
            
            return {
                "name": tool.name,
                "description": tool.description,
                "version": tool.version,
                "parameters": schema.get("properties", {}),
                "required_parameters": schema.get("required", [])
            }
        except Exception as e:
            raise create_tool_error(
                f"Failed to get tool info: {str(e)}",
                "rag_agent",
                {"tool_name": tool_name},
                ErrorCode.TOOL_NOT_FOUND,
                e
            )
    
    def get_query_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent query history.
        
        Args:
            limit: Maximum number of queries to return
            
        Returns:
            List of recent query records
        """
        return self.query_history[-limit:] if self.query_history else []
    
    def clear_history(self) -> None:
        """Clear query history."""
        self.query_history.clear()
        self.logger.info("Query history cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all components.
        
        Returns:
            Dictionary with health status of all components
        """
        health_status = {
            "agent": "healthy",
            "components": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Check LLM
            try:
                await self.llm.generate(prompt="test", max_tokens=5)
                health_status["components"]["llm"] = "healthy"
            except Exception as e:
                health_status["components"]["llm"] = f"unhealthy: {str(e)}"
                health_status["agent"] = "degraded"
            
            # Check vector database
            try:
                await self.vector_db.list_corpora()
                health_status["components"]["vector_db"] = "healthy"
            except Exception as e:
                health_status["components"]["vector_db"] = f"unhealthy: {str(e)}"
                health_status["agent"] = "degraded"
            
            # Check tool registry
            tool_count = len(self.get_available_tools())
            health_status["components"]["tool_registry"] = f"healthy ({tool_count} tools)"
            
            # Check components
            health_status["components"]["planner"] = "healthy" if self.planner else "not initialized"
            health_status["components"]["executor"] = "healthy" if self.executor else "not initialized"
            health_status["components"]["reasoning_engine"] = "healthy" if self.reasoning_engine else "not initialized"
            
        except Exception as e:
            health_status["agent"] = "unhealthy"
            health_status["error"] = str(e)
        
        return health_status
    
    def _initialize_llm(self, config: LLMConfig) -> BaseLLM:
        """Initialize LLM provider from configuration."""
        try:
            self.logger.info(f"Initializing LLM provider: {config.provider}")
            
            if config.provider.lower() == "mock":
                return MockLLMProvider(model=config.model)
            else:
                return LitellmProvider(
                    model=config.model,
                    api_key=config.api_key,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens
                )
        except Exception as e:
            raise create_tool_error(
                f"Failed to initialize LLM: {str(e)}",
                "rag_agent",
                {"provider": config.provider},
                ErrorCode.LLM_CONNECTION_FAILED,
                e
            )
    
    def _initialize_vector_db(self, config: VectorDBConfig) -> BaseVectorDB:
        """Initialize vector database from configuration."""
        try:
            self.logger.info(f"Initializing vector database: {config.provider}")
            
            if config.provider.lower() == "faiss":
                # Import embedding components
                from ..utils.embeddings import (
                    EmbeddingManager,
                    OpenAIEmbeddingProvider,
                    MockEmbeddingProvider
                )
                
                # Determine which embedding provider to use based on model name
                if "openai" in config.embedding_model.lower() or "ada" in config.embedding_model.lower():
                    # Use OpenAI provider for OpenAI models
                    provider = OpenAIEmbeddingProvider(
                        model=config.embedding_model,
                        api_key=config.connection_params.get("openai_api_key")
                    )
                elif "mock" in config.embedding_model.lower() or config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2":
                    # Use mock provider for testing or when no API key is available
                    provider = MockEmbeddingProvider(
                        dimension=config.dimension,
                        model_name=config.embedding_model
                    )
                else:
                    # Default to mock provider with a warning
                    self.logger.warning(
                        f"Unknown embedding model '{config.embedding_model}', using mock provider. "
                        "For production, use OpenAI models or configure a proper provider."
                    )
                    provider = MockEmbeddingProvider(
                        dimension=config.dimension,
                        model_name=config.embedding_model
                    )
                
                embedding_manager = EmbeddingManager(
                    provider=provider,
                    enable_caching=True
                )
                
                db = FAISSVectorDB(
                    config=config,
                    embedding_manager=embedding_manager
                )
                # Note: Connection is deferred until first use or explicit connect() call
                # This avoids unsafe async operations in __init__
                return db
            else:
                raise ValueError(f"Unsupported vector database provider: {config.provider}")
        except Exception as e:
            raise create_tool_error(
                f"Failed to initialize vector database: {str(e)}",
                "rag_agent",
                {"provider": config.provider},
                ErrorCode.VECTOR_DB_CONNECTION_FAILED,
                e
            )
    
    def _register_default_tools(self) -> None:
        """Register default RAG tools."""
        try:
            from ..tools import (
                CreateCorpusTool,
                AddDataTool,
                RAGQueryTool,
                ListCorporaTool,
                DeleteCorpusTool,
                GetCorpusInfoTool
            )
            from ..utils.embeddings import (
                EmbeddingManager,
                OpenAIEmbeddingProvider,
                MockEmbeddingProvider
            )
            from ..utils.chunking import TextChunker
            
            # Create embedding provider based on model
            if "openai" in self.vector_db_config.embedding_model.lower() or "ada" in self.vector_db_config.embedding_model.lower():
                provider = OpenAIEmbeddingProvider(
                    model=self.vector_db_config.embedding_model,
                    api_key=self.vector_db_config.connection_params.get("openai_api_key")
                )
            else:
                # Use mock provider for testing
                provider = MockEmbeddingProvider(
                    dimension=self.vector_db_config.dimension,
                    model_name=self.vector_db_config.embedding_model
                )
            
            # Create embedding manager
            embedding_manager = EmbeddingManager(
                provider=provider,
                enable_caching=True
            )
            text_chunker = TextChunker()
            
            # Register tools
            self.tool_registry.register_tool(
                CreateCorpusTool(vector_db=self.vector_db),
                category="corpus_management"
            )
            
            self.tool_registry.register_tool(
                AddDataTool(
                    vector_db=self.vector_db,
                    embedding_manager=embedding_manager,
                    text_chunker=text_chunker
                ),
                category="data_management"
            )
            
            self.tool_registry.register_tool(
                RAGQueryTool(
                    vector_db=self.vector_db,
                    embedding_manager=embedding_manager
                ),
                category="search"
            )
            
            self.tool_registry.register_tool(
                ListCorporaTool(vector_db=self.vector_db),
                category="corpus_management"
            )
            
            self.tool_registry.register_tool(
                DeleteCorpusTool(vector_db=self.vector_db),
                category="corpus_management"
            )
            
            self.tool_registry.register_tool(
                GetCorpusInfoTool(vector_db=self.vector_db),
                category="corpus_management"
            )
            
            self.logger.info(f"Registered {len(self.get_available_tools())} default tools")
            
        except Exception as e:
            self.logger.warning(f"Failed to register some default tools: {e}")
    
    def _get_default_llm_config(self) -> LLMConfig:
        """Get default LLM configuration."""
        return LLMConfig(
            provider="mock",
            model="mock-model",
            temperature=0.7,
            max_tokens=2000
        )
    
    def _get_default_vector_db_config(self) -> VectorDBConfig:
        """
        Get default vector database configuration.
        
        Note: This uses a mock embedding model for testing. For production:
        - Use OpenAI: embedding_model="text-embedding-ada-002"
        - Provide API key in connection_params: {"openai_api_key": "your-key"}
        """
        # Use FAISS as default but with minimal config
        return VectorDBConfig(
            provider="faiss",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Mock model for testing
            dimension=384,
            connection_params={"index_path": "./faiss_index", "metric": "cosine"}
        )
    
    def _get_default_agent_config(self) -> AgentConfig:
        """Get default agent configuration."""
        return AgentConfig(
            max_context_length=50000,
            max_tool_calls=20,
            log_level="INFO"
        )
    
    @classmethod
    def from_config(cls, config: RAGAgentConfig, **kwargs) -> 'RAGAgent':
        """
        Create a RAGAgent from a RAGAgentConfig object.
        
        Args:
            config: Complete RAGAgent configuration
            **kwargs: Additional initialization options
            
        Returns:
            Configured RAGAgent instance
        """
        return cls(
            llm_config=config.llm,
            vector_db_config=config.vector_db,
            agent_config=config.agent,
            **kwargs
        )
    
    @classmethod
    def from_yaml(cls, file_path: str, **kwargs) -> 'RAGAgent':
        """
        Create a RAGAgent from a YAML configuration file.
        
        Example YAML structure:
        ```yaml
        llm:
          provider: openai
          model: gpt-3.5-turbo
          api_key: ${OPENAI_API_KEY}
          temperature: 0.7
          max_tokens: 2000
        
        vector_db:
          provider: faiss
          embedding_model: text-embedding-ada-002
          dimension: 1536
          connection_params:
            index_path: ./faiss_index
            metric: cosine
        
        agent:
          max_context_length: 8000
          max_tool_calls: 10
          log_level: INFO
        ```
        
        Args:
            file_path: Path to YAML configuration file
            **kwargs: Additional initialization options
            
        Returns:
            Configured RAGAgent instance
        """
        config = RAGAgentConfig.from_yaml(file_path)
        return cls.from_config(config, **kwargs)
    
    @classmethod
    def from_json(cls, file_path: str, **kwargs) -> 'RAGAgent':
        """
        Create a RAGAgent from a JSON configuration file.
        
        Example JSON structure:
        ```json
        {
          "llm": {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "api_key": "${OPENAI_API_KEY}",
            "temperature": 0.7,
            "max_tokens": 2000
          },
          "vector_db": {
            "provider": "faiss",
            "embedding_model": "text-embedding-ada-002",
            "dimension": 1536,
            "connection_params": {
              "index_path": "./faiss_index",
              "metric": "cosine"
            }
          },
          "agent": {
            "max_context_length": 8000,
            "max_tool_calls": 10,
            "log_level": "INFO"
          }
        }
        ```
        
        Args:
            file_path: Path to JSON configuration file
            **kwargs: Additional initialization options
            
        Returns:
            Configured RAGAgent instance
        """
        config = RAGAgentConfig.from_json(file_path)
        return cls.from_config(config, **kwargs)
    
    @classmethod
    def from_env(cls, **kwargs) -> 'RAGAgent':
        """
        Create a RAGAgent from environment variables.
        
        Environment variables:
        - LEXORA_LLM_PROVIDER: LLM provider name
        - LEXORA_LLM_MODEL: Model name (required)
        - LEXORA_LLM_API_KEY: API key
        - LEXORA_LLM_TEMPERATURE: Temperature (default: 0.7)
        - LEXORA_LLM_MAX_TOKENS: Max tokens (default: 2000)
        
        - LEXORA_VECTORDB_PROVIDER: Vector DB provider (required)
        - LEXORA_VECTORDB_EMBEDDING_MODEL: Embedding model
        - LEXORA_VECTORDB_DIMENSION: Vector dimension
        - LEXORA_VECTORDB_CONNECTION_PARAMS: JSON string of connection params
        
        - LEXORA_AGENT_MAX_CONTEXT_LENGTH: Max context length
        - LEXORA_AGENT_MAX_TOOL_CALLS: Max tool calls
        - LEXORA_AGENT_LOG_LEVEL: Log level
        
        Args:
            **kwargs: Additional initialization options
            
        Returns:
            Configured RAGAgent instance
            
        Example:
            ```bash
            export LEXORA_LLM_MODEL="gpt-3.5-turbo"
            export LEXORA_LLM_API_KEY="sk-..."
            export LEXORA_VECTORDB_PROVIDER="faiss"
            export LEXORA_VECTORDB_CONNECTION_PARAMS='{"index_path": "./faiss_index"}'
            ```
            
            ```python
            agent = RAGAgent.from_env()
            ```
        """
        config = RAGAgentConfig.from_env()
        return cls.from_config(config, **kwargs)
    
    def save_config(self, file_path: str, format: str = "yaml") -> None:
        """
        Save current agent configuration to a file.
        
        Args:
            file_path: Path where to save the configuration
            format: File format ("yaml" or "json")
            
        Raises:
            ValueError: If format is not supported
        """
        config = RAGAgentConfig(
            llm=self.llm_config,
            vector_db=self.vector_db_config,
            agent=self.agent_config
        )
        
        if format.lower() == "yaml":
            config.save_yaml(file_path)
            self.logger.info(f"Configuration saved to {file_path} (YAML)")
        elif format.lower() == "json":
            config.save_json(file_path)
            self.logger.info(f"Configuration saved to {file_path} (JSON)")
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'")
    
    def get_config(self) -> RAGAgentConfig:
        """
        Get the current agent configuration.
        
        Returns:
            RAGAgentConfig object with current settings
        """
        return RAGAgentConfig(
            llm=self.llm_config,
            vector_db=self.vector_db_config,
            agent=self.agent_config
        )
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return (
            f"RAGAgent("
            f"llm={self.llm_config.provider}, "
            f"vector_db={self.vector_db_config.provider}, "
            f"tools={len(self.get_available_tools())}"
            f")"
        )


# Convenience function for creating the agent
def create_rag_agent(
    llm_config: Optional[LLMConfig] = None,
    vector_db_config: Optional[VectorDBConfig] = None,
    agent_config: Optional[AgentConfig] = None,
    **kwargs
) -> RAGAgent:
    """
    Create a RAGAgent instance.
    
    Args:
        llm_config: Configuration for LLM provider
        vector_db_config: Configuration for vector database
        agent_config: Configuration for agent behavior
        **kwargs: Additional configuration options
        
    Returns:
        Configured RAGAgent instance
    """
    return RAGAgent(
        llm_config=llm_config,
        vector_db_config=vector_db_config,
        agent_config=agent_config,
        **kwargs
    )
