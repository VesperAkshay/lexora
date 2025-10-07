# Lexora Agentic RAG SDK - API Reference

## Table of Contents

- [Core Classes](#core-classes)
  - [RAGAgent](#ragagent)
- [Configuration](#configuration)
  - [LLMConfig](#llmconfig)
  - [VectorDBConfig](#vectordbconfig)
  - [AgentConfig](#agentconfig)
  - [RAGAgentConfig](#ragagentconfig)
- [Data Models](#data-models)
  - [Document](#document)
  - [SearchResult](#searchresult)
  - [CorpusInfo](#corpusinfo)
- [Response Models](#response-models)
  - [AgentResponse](#agentresponse)
  - [ToolResult](#toolresult)
- [Base Classes](#base-classes)
  - [BaseTool](#basetool)
  - [BaseLLM](#basellm)
  - [BaseVectorDB](#basevectordb)
- [Exceptions](#exceptions)
- [Utilities](#utilities)

---

## Core Classes

### RAGAgent

Main orchestrator for the agentic RAG system.

```python
class RAGAgent:
    """
    Main orchestrator for the agentic RAG system.
    
    The RAGAgent coordinates all components including the planner, executor,
    and reasoning engine to process user queries and generate responses.
    """
```

#### Constructor

```python
def __init__(
    self,
    llm_config: Optional[LLMConfig] = None,
    vector_db_config: Optional[VectorDBConfig] = None,
    agent_config: Optional[AgentConfig] = None,
    tools: Optional[List[BaseTool]] = None,
    **kwargs
) -> None
```

**Parameters:**
- `llm_config` (Optional[LLMConfig]): Configuration for LLM provider
- `vector_db_config` (Optional[VectorDBConfig]): Configuration for vector database
- `agent_config` (Optional[AgentConfig]): Configuration for agent behavior
- `tools` (Optional[List[BaseTool]]): Optional list of custom tools to register
- `**kwargs`: Additional configuration options

**Raises:**
- `LexoraError`: If configuration validation fails

**Example:**
```python
from lexora import RAGAgent, LLMConfig, VectorDBConfig

llm_config = LLMConfig(
    provider="openai",
    model="gpt-3.5-turbo",
    api_key="your-api-key"
)

vector_db_config = VectorDBConfig(
    provider="faiss",
    embedding_model="text-embedding-ada-002",
    dimension=1536,
    connection_params={"index_path": "./faiss_index"}
)

agent = RAGAgent(
    llm_config=llm_config,
    vector_db_config=vector_db_config
)
```

#### Methods

##### query()

Process a user query through the agentic pipeline.

```python
async def query(
    self,
    input_text: str,
    context: Optional[Dict[str, Any]] = None,
    reasoning_strategy: Optional[ReasoningStrategy] = None
) -> AgentResponse
```

**Parameters:**
- `input_text` (str): The user's query text
- `context` (Optional[Dict[str, Any]]): Additional context for the query
- `reasoning_strategy` (Optional[ReasoningStrategy]): Strategy for response generation

**Returns:**
- `AgentResponse`: Complete response with answer, sources, and metadata

**Example:**
```python
response = await agent.query("What is machine learning?")
print(response.answer)
print(f"Confidence: {response.confidence}")
print(f"Sources: {len(response.sources)}")
```

##### add_tool()

Register a custom tool with the agent.

```python
def add_tool(
    self,
    tool: BaseTool,
    category: Optional[str] = None
) -> None
```

**Parameters:**
- `tool` (BaseTool): Tool instance to register
- `category` (Optional[str]): Optional category for the tool

**Raises:**
- `LexoraError`: If tool registration fails

**Example:**
```python
from lexora import BaseTool

class CustomTool(BaseTool):
    name = "custom_tool"
    description = "My custom tool"
    
    async def _execute(self, **kwargs):
        return {"result": "custom result"}

agent.add_tool(CustomTool())
```

##### get_available_tools()

Return list of available tool names.

```python
def get_available_tools(self) -> List[str]
```

**Returns:**
- `List[str]`: List of tool names registered with the agent

**Example:**
```python
tools = agent.get_available_tools()
print(f"Available tools: {', '.join(tools)}")
```

##### get_tool_info()

Get detailed information about a specific tool.

```python
def get_tool_info(self, tool_name: str) -> Dict[str, Any]
```

**Parameters:**
- `tool_name` (str): Name of the tool

**Returns:**
- `Dict[str, Any]`: Dictionary containing tool information

**Raises:**
- `LexoraError`: If tool not found

**Example:**
```python
info = agent.get_tool_info("rag_query")
print(f"Tool: {info['name']}")
print(f"Description: {info['description']}")
print(f"Parameters: {info['parameters']}")
```

#### Class Methods

##### from_yaml()

Create a RAGAgent from a YAML configuration file.

```python
@classmethod
def from_yaml(cls, file_path: str, **kwargs) -> 'RAGAgent'
```

**Parameters:**
- `file_path` (str): Path to YAML configuration file
- `**kwargs`: Additional initialization options

**Returns:**
- `RAGAgent`: Configured RAGAgent instance

**Example:**
```python
agent = RAGAgent.from_yaml("config.yaml")
```

##### from_json()

Create a RAGAgent from a JSON configuration file.

```python
@classmethod
def from_json(cls, file_path: str, **kwargs) -> 'RAGAgent'
```

**Parameters:**
- `file_path` (str): Path to JSON configuration file
- `**kwargs`: Additional initialization options

**Returns:**
- `RAGAgent`: Configured RAGAgent instance

**Example:**
```python
agent = RAGAgent.from_json("config.json")
```

##### from_env()

Create a RAGAgent from environment variables.

```python
@classmethod
def from_env(cls, **kwargs) -> 'RAGAgent'
```

**Parameters:**
- `**kwargs`: Additional initialization options

**Returns:**
- `RAGAgent`: Configured RAGAgent instance

**Example:**
```python
# Set environment variables first
# LEXORA_LLM_MODEL=gpt-3.5-turbo
# LEXORA_VECTORDB_PROVIDER=faiss
# etc.

agent = RAGAgent.from_env()
```

##### from_config()

Create a RAGAgent from a RAGAgentConfig object.

```python
@classmethod
def from_config(cls, config: RAGAgentConfig, **kwargs) -> 'RAGAgent'
```

**Parameters:**
- `config` (RAGAgentConfig): Complete RAGAgent configuration
- `**kwargs`: Additional initialization options

**Returns:**
- `RAGAgent`: Configured RAGAgent instance

**Example:**
```python
from lexora import RAGAgentConfig

config = RAGAgentConfig.from_yaml("config.yaml")
agent = RAGAgent.from_config(config)
```

##### save_config()

Save current agent configuration to a file.

```python
def save_config(self, file_path: str, format: str = "yaml") -> None
```

**Parameters:**
- `file_path` (str): Path where to save the configuration
- `format` (str): File format ("yaml" or "json")

**Raises:**
- `ValueError`: If format is not supported

**Example:**
```python
agent.save_config("my_config.yaml", format="yaml")
agent.save_config("my_config.json", format="json")
```

##### get_config()

Get the current agent configuration.

```python
def get_config(self) -> RAGAgentConfig
```

**Returns:**
- `RAGAgentConfig`: Object with current settings

**Example:**
```python
config = agent.get_config()
print(config.to_dict())
```

---

## Configuration

### LLMConfig

Configuration for LLM provider.

```python
class LLMConfig(BaseModel):
    """Configuration for LLM provider."""
    
    provider: str = "litellm"
    model: str
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    additional_params: Dict[str, Any] = {}
```

**Attributes:**
- `provider` (str): The LLM provider name (default: "litellm")
- `model` (str): The specific model to use (required)
- `api_key` (Optional[str]): Optional API key for the provider
- `temperature` (float): Sampling temperature (0.0-2.0, default: 0.7)
- `max_tokens` (int): Maximum tokens in response (default: 2000)
- `additional_params` (Dict): Additional provider-specific parameters

**Example:**
```python
from lexora import LLMConfig

config = LLMConfig(
    provider="openai",
    model="gpt-3.5-turbo",
    api_key="sk-...",
    temperature=0.7,
    max_tokens=2000
)
```

**Class Methods:**
- `from_dict(data: Dict[str, Any]) -> LLMConfig`: Create from dictionary
- `from_env(prefix: str = "LEXORA_LLM_") -> LLMConfig`: Create from environment variables
- `to_dict() -> Dict[str, Any]`: Convert to dictionary

### VectorDBConfig

Configuration for vector database.

```python
class VectorDBConfig(BaseModel):
    """Configuration for vector database."""
    
    provider: str
    connection_params: Dict[str, Any]
    embedding_model: str = "text-embedding-ada-002"
    dimension: int = 1536
```

**Attributes:**
- `provider` (str): Vector DB provider ("faiss", "pinecone", "chroma")
- `connection_params` (Dict): Provider-specific connection parameters
- `embedding_model` (str): Embedding model name (default: "text-embedding-ada-002")
- `dimension` (int): Vector dimension (default: 1536)

**Example:**
```python
from lexora import VectorDBConfig

config = VectorDBConfig(
    provider="faiss",
    embedding_model="text-embedding-ada-002",
    dimension=1536,
    connection_params={
        "index_path": "./faiss_index",
        "metric": "cosine"
    }
)
```

**Class Methods:**
- `from_dict(data: Dict[str, Any]) -> VectorDBConfig`: Create from dictionary
- `from_env(prefix: str = "LEXORA_VECTORDB_") -> VectorDBConfig`: Create from environment variables
- `to_dict() -> Dict[str, Any]`: Convert to dictionary

### AgentConfig

Configuration for RAG agent behavior.

```python
class AgentConfig(BaseModel):
    """Configuration for RAG agent."""
    
    max_context_length: int = 8000
    max_tool_calls: int = 10
    reasoning_model: Optional[str] = None
    enable_memory: bool = True
    log_level: str = "INFO"
```

**Attributes:**
- `max_context_length` (int): Maximum context length (default: 8000)
- `max_tool_calls` (int): Maximum tool calls per query (default: 10)
- `reasoning_model` (Optional[str]): Optional specific model for reasoning
- `enable_memory` (bool): Enable conversation memory (default: True)
- `log_level` (str): Logging level (default: "INFO")

**Example:**
```python
from lexora import AgentConfig

config = AgentConfig(
    max_context_length=8000,
    max_tool_calls=10,
    log_level="INFO"
)
```

**Class Methods:**
- `from_dict(data: Dict[str, Any]) -> AgentConfig`: Create from dictionary
- `from_env(prefix: str = "LEXORA_AGENT_") -> AgentConfig`: Create from environment variables
- `to_dict() -> Dict[str, Any]`: Convert to dictionary

### RAGAgentConfig

Complete configuration for RAGAgent.

```python
class RAGAgentConfig(BaseModel):
    """Complete configuration for RAGAgent."""
    
    llm: LLMConfig
    vector_db: VectorDBConfig
    agent: AgentConfig = AgentConfig()
```

**Attributes:**
- `llm` (LLMConfig): LLM provider configuration
- `vector_db` (VectorDBConfig): Vector database configuration
- `agent` (AgentConfig): Agent behavior configuration

**Example:**
```python
from lexora import RAGAgentConfig, LLMConfig, VectorDBConfig

config = RAGAgentConfig(
    llm=LLMConfig(model="gpt-3.5-turbo"),
    vector_db=VectorDBConfig(
        provider="faiss",
        connection_params={"index_path": "./faiss_index"}
    )
)
```

**Class Methods:**
- `from_dict(data: Dict[str, Any]) -> RAGAgentConfig`: Create from dictionary
- `from_yaml(file_path: str) -> RAGAgentConfig`: Load from YAML file
- `from_json(file_path: str) -> RAGAgentConfig`: Load from JSON file
- `from_env() -> RAGAgentConfig`: Load from environment variables
- `to_dict() -> Dict[str, Any]`: Convert to dictionary
- `save_yaml(file_path: str) -> None`: Save to YAML file
- `save_json(file_path: str, indent: int = 2) -> None`: Save to JSON file

---

## Data Models

### Document

Represents a document in the RAG system.

```python
class Document(BaseModel):
    """Represents a document in the RAG system."""
    
    id: str
    content: str
    metadata: Dict[str, Any] = {}
    embedding: Optional[List[float]] = None
```

**Attributes:**
- `id` (str): Unique document identifier
- `content` (str): Document content/text
- `metadata` (Dict): Additional metadata
- `embedding` (Optional[List[float]]): Vector embedding

### SearchResult

Represents a search result from vector database.

```python
class SearchResult(BaseModel):
    """Represents a search result."""
    
    document: Document
    score: float
    rank: int
```

**Attributes:**
- `document` (Document): The retrieved document
- `score` (float): Similarity/relevance score
- `rank` (int): Result ranking

### CorpusInfo

Information about a corpus.

```python
class CorpusInfo(BaseModel):
    """Information about a corpus."""
    
    name: str
    description: str
    document_count: int
    created_at: str
    metadata: Dict[str, Any] = {}
```

**Attributes:**
- `name` (str): Corpus name
- `description` (str): Corpus description
- `document_count` (int): Number of documents
- `created_at` (str): Creation timestamp
- `metadata` (Dict): Additional metadata

---

## Response Models

### AgentResponse

Response from the RAG agent.

```python
class AgentResponse:
    """Response from the RAG agent."""
    
    answer: str
    confidence: float
    sources: List[Dict[str, Any]] = []
    reasoning_chain: List[str] = []
    execution_time: float = 0.0
    metadata: Dict[str, Any] = {}
    timestamp: datetime
```

**Attributes:**
- `answer` (str): Generated answer
- `confidence` (float): Confidence score (0.0-1.0)
- `sources` (List[Dict]): Source documents used
- `reasoning_chain` (List[str]): Reasoning steps
- `execution_time` (float): Time taken in seconds
- `metadata` (Dict): Additional metadata
- `timestamp` (datetime): Response timestamp

**Methods:**
- `to_dict() -> Dict[str, Any]`: Convert to dictionary

### ToolResult

Result from tool execution.

```python
class ToolResult(BaseModel):
    """Result from tool execution."""
    
    status: ToolStatus
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
```

**Attributes:**
- `status` (ToolStatus): Execution status
- `data` (Optional[Dict]): Result data
- `message` (Optional[str]): Human-readable message
- `error` (Optional[str]): Error message if failed
- `execution_time` (Optional[float]): Execution time
- `metadata` (Optional[Dict]): Additional metadata

---

## Base Classes

### BaseTool

Abstract base class for tools.

```python
class BaseTool(ABC):
    """Abstract base class for tools."""
    
    name: str
    description: str
    version: str = "1.0.0"
    parameters: List[ToolParameter] = []
```

**Abstract Methods:**
- `async def _execute(self, **kwargs) -> Dict[str, Any]`: Execute tool logic

**Methods:**
- `async def run(self, **kwargs) -> ToolResult`: Execute tool with validation
- `def get_schema() -> Dict[str, Any]`: Get tool schema
- `def validate_parameters(**kwargs) -> Dict[str, Any]`: Validate parameters

### BaseLLM

Abstract base class for LLM providers.

```python
class BaseLLM(ABC):
    """Abstract base class for LLM providers."""
```

**Abstract Methods:**
- `async def generate(prompt: str, **kwargs) -> str`: Generate text
- `async def generate_structured(prompt: str, schema: Dict, **kwargs) -> Dict`: Generate structured output

### BaseVectorDB

Abstract base class for vector databases.

```python
class BaseVectorDB(ABC):
    """Abstract base class for vector databases."""
```

**Abstract Methods:**
- `async def create_corpus(name: str, **kwargs) -> None`: Create corpus
- `async def add_documents(corpus_name: str, documents: List[Document]) -> None`: Add documents
- `async def search(corpus_name: str, query_embedding: List[float], top_k: int) -> List[SearchResult]`: Search
- `async def delete_corpus(corpus_name: str) -> None`: Delete corpus

---

## Exceptions

### LexoraError

Base exception for all Lexora errors.

```python
class LexoraError(Exception):
    """Base exception for Lexora SDK."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        context: Dict[str, Any] = None,
        suggestions: List[str] = None,
        original_error: Exception = None
    )
```

### ConfigurationError

Configuration-related errors.

```python
class ConfigurationError(LexoraError):
    """Configuration-related errors."""
```

### LLMError

LLM provider errors.

```python
class LLMError(LexoraError):
    """LLM provider errors."""
```

### VectorDBError

Vector database errors.

```python
class VectorDBError(LexoraError):
    """Vector database errors."""
```

### ToolExecutionError

Tool execution errors.

```python
class ToolExecutionError(LexoraError):
    """Tool execution errors."""
```

### PlanningError

Query planning errors.

```python
class PlanningError(LexoraError):
    """Query planning errors."""
```

---

## Utilities

### setup_logging()

Configure logging for the SDK.

```python
def setup_logging(
    level: str = "INFO",
    format_type: str = "structured",
    log_file: Optional[str] = None,
    correlation_ids: bool = True
) -> None
```

**Parameters:**
- `level` (str): Log level (default: "INFO")
- `format_type` (str): Format type (default: "structured")
- `log_file` (Optional[str]): Optional log file path
- `correlation_ids` (bool): Enable correlation IDs (default: True)

### get_logger()

Get a logger instance.

```python
def get_logger(name: str) -> logging.Logger
```

**Parameters:**
- `name` (str): Logger name

**Returns:**
- `logging.Logger`: Logger instance

---

## Type Hints

All public APIs include complete type hints for type checking with mypy, pyright, or other type checkers.

Example:
```python
from typing import Optional, List, Dict, Any
from lexora import RAGAgent, AgentResponse

async def process_query(
    agent: RAGAgent,
    query: str,
    context: Optional[Dict[str, Any]] = None
) -> AgentResponse:
    """Process a query with the agent."""
    response: AgentResponse = await agent.query(query, context=context)
    return response
```

---

## Version Information

```python
import lexora

print(lexora.__version__)  # "0.1.0"
print(lexora.get_version())  # "0.1.0"
print(lexora.get_info())  # Full SDK information
```

---

For more examples and tutorials, see the [Examples Documentation](EXAMPLES.md).
