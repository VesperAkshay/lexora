# RAGAgent Configuration Management Guide

## Overview

The RAGAgent now supports comprehensive configuration management with multiple loading methods:
- YAML files
- JSON files
- Environment variables
- Programmatic configuration
- Dictionary-based configuration

## Configuration Methods

### 1. Loading from YAML File

```python
from rag_agent import RAGAgent

# Load agent from YAML configuration
agent = RAGAgent.from_yaml("config.yaml")
```

**Example YAML (`config.yaml`):**
```yaml
llm:
  provider: openai
  model: gpt-3.5-turbo
  api_key: ${OPENAI_API_KEY}
  temperature: 0.7
  max_tokens: 2000

vector_db:
  provider: faiss
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  dimension: 384
  connection_params:
    index_path: ./faiss_index
    metric: cosine

agent:
  max_context_length: 8000
  max_tool_calls: 10
  log_level: INFO
```

### 2. Loading from JSON File

```python
# Load agent from JSON configuration
agent = RAGAgent.from_json("config.json")
```

**Example JSON (`config.json`):**
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
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "dimension": 384,
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

### 3. Loading from Environment Variables

```python
# Load agent from environment variables
agent = RAGAgent.from_env()
```

**Environment Variables:**
```bash
# LLM Configuration
export LEXORA_LLM_PROVIDER=openai
export LEXORA_LLM_MODEL=gpt-3.5-turbo
export LEXORA_LLM_API_KEY=your-api-key
export LEXORA_LLM_TEMPERATURE=0.7
export LEXORA_LLM_MAX_TOKENS=2000

# Vector DB Configuration
export LEXORA_VECTORDB_PROVIDER=faiss
export LEXORA_VECTORDB_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
export LEXORA_VECTORDB_DIMENSION=384
export LEXORA_VECTORDB_CONNECTION_PARAMS='{"index_path": "./faiss_index", "metric": "cosine"}'

# Agent Configuration
export LEXORA_AGENT_MAX_CONTEXT_LENGTH=8000
export LEXORA_AGENT_MAX_TOOL_CALLS=10
export LEXORA_AGENT_LOG_LEVEL=INFO
export LEXORA_AGENT_ENABLE_MEMORY=true
```

### 4. Programmatic Configuration

```python
from rag_agent import RAGAgent
from models.config import LLMConfig, VectorDBConfig, AgentConfig

# Create configuration objects
llm_config = LLMConfig(
    provider="openai",
    model="gpt-3.5-turbo",
    api_key="your-api-key",
    temperature=0.7,
    max_tokens=2000
)

vector_db_config = VectorDBConfig(
    provider="faiss",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    dimension=384,
    connection_params={
        "index_path": "./faiss_index",
        "metric": "cosine"
    }
)

agent_config = AgentConfig(
    max_context_length=8000,
    max_tool_calls=10,
    log_level="INFO"
)

# Create agent with config objects
agent = RAGAgent(
    llm_config=llm_config,
    vector_db_config=vector_db_config,
    agent_config=agent_config
)
```

### 5. Loading from Dictionary

```python
from models.config import RAGAgentConfig

config_dict = {
    "llm": {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 2000
    },
    "vector_db": {
        "provider": "faiss",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "connection_params": {
            "index_path": "./faiss_index"
        }
    },
    "agent": {
        "max_context_length": 8000,
        "max_tool_calls": 10,
        "log_level": "INFO"
    }
}

# Create config from dictionary
config = RAGAgentConfig.from_dict(config_dict)
agent = RAGAgent.from_config(config)
```

## Saving Configuration

### Save to YAML

```python
agent.save_config("my_config.yaml", format="yaml")
```

### Save to JSON

```python
agent.save_config("my_config.json", format="json")
```

### Get Current Configuration

```python
current_config = agent.get_config()
print(current_config.to_dict())
```

## Individual Config Loading

You can also load individual configuration components from environment variables:

```python
from models.config import LLMConfig, VectorDBConfig, AgentConfig

# Load individual configs from environment
llm_config = LLMConfig.from_env()
vector_db_config = VectorDBConfig.from_env()
agent_config = AgentConfig.from_env()

# Create agent with individual configs
agent = RAGAgent(
    llm_config=llm_config,
    vector_db_config=vector_db_config,
    agent_config=agent_config
)
```

## Configuration Parameters

### LLM Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | str | "litellm" | LLM provider name |
| `model` | str | required | Model name (e.g., "gpt-3.5-turbo") |
| `api_key` | str | None | API key for the provider |
| `temperature` | float | 0.7 | Sampling temperature (0.0-2.0) |
| `max_tokens` | int | 2000 | Maximum tokens in response |
| `additional_params` | dict | {} | Additional provider-specific parameters |

### Vector DB Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | str | required | Vector DB provider ("faiss", "pinecone", "chroma") |
| `embedding_model` | str | "text-embedding-ada-002" | Embedding model name |
| `dimension` | int | 1536 | Vector dimension |
| `connection_params` | dict | required | Provider-specific connection parameters |

### Agent Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_context_length` | int | 8000 | Maximum context length |
| `max_tool_calls` | int | 10 | Maximum tool calls per query |
| `reasoning_model` | str | None | Optional specific model for reasoning |
| `enable_memory` | bool | True | Enable conversation memory |
| `log_level` | str | "INFO" | Logging level |

## Environment Variable Prefixes

All environment variables use specific prefixes:

- **LLM Config**: `LEXORA_LLM_*`
- **Vector DB Config**: `LEXORA_VECTORDB_*`
- **Agent Config**: `LEXORA_AGENT_*`

## Best Practices

1. **Use Environment Variables for Sensitive Data**
   - Store API keys in environment variables
   - Never commit API keys to version control
   - Use `.env` files for local development

2. **Use Configuration Files for Deployment**
   - YAML for human-readable configs
   - JSON for programmatic generation
   - Version control your config templates

3. **Validate Configuration**
   - All configs use Pydantic for validation
   - Invalid configs raise descriptive errors
   - Check error messages for suggestions

4. **Configuration Hierarchy**
   - Environment variables override file configs
   - Programmatic configs override everything
   - Use the method that fits your deployment

## Examples

See the `examples/` directory for complete examples:

- `examples/config_example.yaml` - YAML configuration template
- `examples/config_example.json` - JSON configuration template
- `examples/.env.example` - Environment variables template
- `examples/configuration_demo.py` - Comprehensive demo of all methods

## Troubleshooting

### Missing Required Parameters

```
ValueError: Environment variable LEXORA_LLM_MODEL is required
```

**Solution**: Set the required environment variable or use a different configuration method.

### Invalid JSON in Connection Params

```
ValueError: Invalid JSON in LEXORA_VECTORDB_CONNECTION_PARAMS
```

**Solution**: Ensure the JSON string is properly formatted:
```bash
export LEXORA_VECTORDB_CONNECTION_PARAMS='{"index_path": "./faiss_index"}'
```

### Configuration File Not Found

```
FileNotFoundError: Configuration file not found: config.yaml
```

**Solution**: Check the file path is correct and the file exists.

## Migration Guide

### From Direct Initialization

**Before:**
```python
agent = RAGAgent()  # Uses defaults
```

**After:**
```python
# Option 1: Use config file
agent = RAGAgent.from_yaml("config.yaml")

# Option 2: Use environment variables
agent = RAGAgent.from_env()

# Option 3: Keep using defaults
agent = RAGAgent()  # Still works!
```

### From Manual Config Objects

**Before:**
```python
llm_config = LLMConfig(...)
vector_db_config = VectorDBConfig(...)
agent = RAGAgent(llm_config=llm_config, vector_db_config=vector_db_config)
```

**After:**
```python
# Still works exactly the same!
llm_config = LLMConfig(...)
vector_db_config = VectorDBConfig(...)
agent = RAGAgent(llm_config=llm_config, vector_db_config=vector_db_config)

# Or use the new methods
config = RAGAgentConfig(llm=llm_config, vector_db=vector_db_config)
agent = RAGAgent.from_config(config)
```

## Summary

The RAGAgent configuration system provides:

✅ Multiple configuration methods (YAML, JSON, env vars, programmatic)
✅ Environment variable support for sensitive data
✅ Configuration file loading and saving
✅ Individual component configuration
✅ Pydantic validation for all configs
✅ Backward compatibility with existing code
✅ Clear error messages and suggestions

Choose the method that best fits your deployment scenario!
