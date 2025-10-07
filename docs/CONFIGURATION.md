# Configuration Guide

This guide covers all configuration options for Lexora.

## Configuration Methods

Lexora supports 4 configuration methods, listed from simplest to most flexible:

1. **Default Configuration** - Zero config, just works
2. **File-Based** - YAML or JSON configuration files
3. **Environment Variables** - For secrets and deployment
4. **Programmatic** - Full control in code

---

## 1. Default Configuration

The easiest way to get started:

```python
from lexora import RAGAgent

agent = RAGAgent()
```

**Default Settings:**
- LLM: Mock provider (for testing)
- Vector DB: FAISS (local storage)
- Embeddings: Mock embeddings (512 dimensions)
- Storage: `./faiss_storage`

---

## 2. File-Based Configuration

### YAML Configuration

Create `config.yaml`:

```yaml
llm:
  provider: "openai"
  model: "gpt-4"
  api_key: "${OPENAI_API_KEY}"  # Reads from environment
  temperature: 0.7
  max_tokens: 2000

vector_db:
  provider: "faiss"
  embedding_model: "text-embedding-ada-002"
  dimension: 1536
  connection_params:
    storage_path: "./vector_storage"
    index_type: "IndexFlatIP"

agent:
  max_iterations: 5
  enable_reasoning: true
  timeout_seconds: 30
  log_level: "INFO"
```

Load it:

```python
from lexora import RAGAgent

agent = RAGAgent.from_yaml("config.yaml")
```

### JSON Configuration

Create `config.json`:

```json
{
  "llm": {
    "provider": "openai",
    "model": "gpt-4",
    "api_key": "${OPENAI_API_KEY}",
    "temperature": 0.7
  },
  "vector_db": {
    "provider": "faiss",
    "embedding_model": "text-embedding-ada-002",
    "dimension": 1536,
    "connection_params": {
      "storage_path": "./vector_storage"
    }
  },
  "agent": {
    "max_iterations": 5,
    "enable_reasoning": true
  }
}
```

Load it:

```python
agent = RAGAgent.from_json("config.json")
```

---

## 3. Environment Variables

### Setting Environment Variables

Create a `.env` file:

```bash
# LLM Configuration
LEXORA_LLM_PROVIDER=openai
LEXORA_LLM_MODEL=gpt-4
LEXORA_LLM_API_KEY=sk-your-api-key-here
LEXORA_LLM_TEMPERATURE=0.7
LEXORA_LLM_MAX_TOKENS=2000

# Vector Database Configuration
LEXORA_VECTORDB_PROVIDER=faiss
LEXORA_VECTORDB_EMBEDDING_MODEL=text-embedding-ada-002
LEXORA_VECTORDB_DIMENSION=1536
LEXORA_VECTORDB_STORAGE_PATH=./vector_storage

# Agent Configuration
LEXORA_AGENT_MAX_ITERATIONS=5
LEXORA_AGENT_ENABLE_REASONING=true
LEXORA_AGENT_LOG_LEVEL=INFO
```

Load from environment:

```python
from lexora import RAGAgent

agent = RAGAgent.from_env()
```

### Environment Variable Reference

#### LLM Variables
- `LEXORA_LLM_PROVIDER` - LLM provider (openai, anthropic, azure, etc.)
- `LEXORA_LLM_MODEL` - Model name
- `LEXORA_LLM_API_KEY` - API key
- `LEXORA_LLM_API_BASE` - API base URL (for Azure, custom endpoints)
- `LEXORA_LLM_TEMPERATURE` - Temperature (0.0-1.0)
- `LEXORA_LLM_MAX_TOKENS` - Maximum tokens

#### Vector DB Variables
- `LEXORA_VECTORDB_PROVIDER` - Provider (faiss, pinecone, chroma)
- `LEXORA_VECTORDB_EMBEDDING_MODEL` - Embedding model name
- `LEXORA_VECTORDB_DIMENSION` - Embedding dimension
- `LEXORA_VECTORDB_STORAGE_PATH` - Storage path (FAISS, Chroma)
- `LEXORA_VECTORDB_API_KEY` - API key (Pinecone)
- `LEXORA_VECTORDB_ENVIRONMENT` - Environment (Pinecone)

#### Agent Variables
- `LEXORA_AGENT_MAX_ITERATIONS` - Max reasoning iterations
- `LEXORA_AGENT_ENABLE_REASONING` - Enable/disable reasoning
- `LEXORA_AGENT_TIMEOUT_SECONDS` - Operation timeout
- `LEXORA_AGENT_LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)

---

## 4. Programmatic Configuration

For maximum flexibility:

```python
from lexora import RAGAgent, LLMConfig, VectorDBConfig, AgentConfig

# Configure LLM
llm_config = LLMConfig(
    provider="openai",
    model="gpt-4",
    api_key="your-api-key",
    temperature=0.7,
    max_tokens=2000
)

# Configure Vector Database
vector_db_config = VectorDBConfig(
    provider="faiss",
    embedding_model="text-embedding-ada-002",
    dimension=1536,
    connection_params={
        "storage_path": "./vector_storage",
        "index_type": "IndexFlatIP"
    }
)

# Configure Agent
agent_config = AgentConfig(
    max_iterations=5,
    enable_reasoning=True,
    timeout_seconds=30,
    log_level="INFO"
)

# Create agent
agent = RAGAgent(
    llm_config=llm_config,
    vector_db_config=vector_db_config,
    agent_config=agent_config
)
```

---

## Configuration Options

### LLM Configuration

```python
LLMConfig(
    provider: str,              # LLM provider name
    model: str,                 # Model identifier
    api_key: str = None,        # API key (optional for mock)
    api_base: str = None,       # Custom API endpoint
    temperature: float = 0.7,   # Randomness (0.0-1.0)
    max_tokens: int = 1000,     # Max response tokens
    timeout: int = 30           # Request timeout (seconds)
)
```

### Vector Database Configuration

```python
VectorDBConfig(
    provider: str,              # "faiss", "pinecone", or "chroma"
    embedding_model: str,       # Embedding model name
    dimension: int,             # Embedding dimension
    connection_params: dict     # Provider-specific parameters
)
```

#### FAISS Connection Params

```python
connection_params={
    "storage_path": "./faiss_storage",  # Where to store indices
    "index_type": "IndexFlatIP"         # FAISS index type
}
```

#### Pinecone Connection Params

```python
connection_params={
    "api_key": "your-pinecone-key",
    "environment": "us-west1-gcp",
    "index_name": "my-index"
}
```

#### Chroma Connection Params

```python
connection_params={
    "persist_directory": "./chroma_storage",
    "collection_name": "my_collection"
}
```

### Agent Configuration

```python
AgentConfig(
    max_iterations: int = 5,        # Max reasoning steps
    enable_reasoning: bool = True,  # Enable AI reasoning
    timeout_seconds: int = 30,      # Operation timeout
    log_level: str = "INFO",        # Logging level
    enable_caching: bool = True     # Enable result caching
)
```

---

## Production Configuration

### Recommended Setup for Production

```yaml
# production.yaml
llm:
  provider: "openai"
  model: "gpt-4"
  api_key: "${OPENAI_API_KEY}"
  temperature: 0.3  # Lower for more consistent results
  max_tokens: 2000

vector_db:
  provider: "pinecone"  # Managed service
  embedding_model: "text-embedding-ada-002"
  dimension: 1536
  connection_params:
    api_key: "${PINECONE_API_KEY}"
    environment: "${PINECONE_ENVIRONMENT}"
    index_name: "production-index"

agent:
  max_iterations: 3
  enable_reasoning: true
  timeout_seconds: 60
  log_level: "WARNING"  # Less verbose in production
  enable_caching: true
```

### Security Best Practices

1. **Never commit API keys** - Use environment variables
2. **Use .env files** - Keep secrets separate
3. **Rotate keys regularly** - Update API keys periodically
4. **Limit permissions** - Use least-privilege API keys

Example `.env` file:

```bash
# .env (add to .gitignore!)
OPENAI_API_KEY=sk-your-key-here
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=us-west1-gcp
```

Load with:

```python
from dotenv import load_dotenv
load_dotenv()

agent = RAGAgent.from_yaml("production.yaml")
```

---

## Saving and Loading Configurations

### Save Current Configuration

```python
# Save as YAML
agent.save_config("my_config.yaml", format="yaml")

# Save as JSON
agent.save_config("my_config.json", format="json")
```

### Load Saved Configuration

```python
# Load YAML
agent = RAGAgent.from_yaml("my_config.yaml")

# Load JSON
agent = RAGAgent.from_json("my_config.json")
```

---

## Configuration Examples

### Example 1: Development Setup

```python
from lexora import RAGAgent

# Quick setup for development
agent = RAGAgent()  # Uses mock LLM and local FAISS
```

### Example 2: Testing with OpenAI

```python
from lexora import RAGAgent, LLMConfig

agent = RAGAgent(
    llm_config=LLMConfig(
        provider="openai",
        model="gpt-3.5-turbo",  # Cheaper for testing
        api_key="your-key"
    )
)
```

### Example 3: Production with Pinecone

```python
from lexora import RAGAgent, LLMConfig, VectorDBConfig
import os

agent = RAGAgent(
    llm_config=LLMConfig(
        provider="openai",
        model="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.3
    ),
    vector_db_config=VectorDBConfig(
        provider="pinecone",
        embedding_model="text-embedding-ada-002",
        dimension=1536,
        connection_params={
            "api_key": os.getenv("PINECONE_API_KEY"),
            "environment": os.getenv("PINECONE_ENVIRONMENT")
        }
    )
)
```

---

## Troubleshooting

### Configuration Not Loading

**Problem**: Configuration file not found

**Solution**: Use absolute paths or check working directory
```python
from pathlib import Path

config_path = Path(__file__).parent / "config.yaml"
agent = RAGAgent.from_yaml(str(config_path))
```

### API Key Errors

**Problem**: "Invalid API key"

**Solution**: Verify your API key is set correctly
```python
import os
print(f"API Key set: {bool(os.getenv('OPENAI_API_KEY'))}")
```

### Embedding Dimension Mismatch

**Problem**: "Dimension mismatch"

**Solution**: Ensure dimension matches your embedding model:
- `text-embedding-ada-002`: 1536
- `text-embedding-3-small`: 1536
- `text-embedding-3-large`: 3072

---

## Next Steps

- [RAG Tools Guide](./RAG_TOOLS.md) - Learn about available tools
- [Custom Tools](./CUSTOM_TOOLS.md) - Build your own tools
- [Deployment Guide](./DEPLOYMENT.md) - Deploy to production
- [API Reference](./API_REFERENCE.md) - Complete API documentation
