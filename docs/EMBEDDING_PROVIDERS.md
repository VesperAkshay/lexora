# Embedding Providers Guide

## Overview

The Lexora Agentic RAG SDK supports multiple embedding providers to convert text into vector representations. This guide explains how to configure and use different embedding providers for development and production.

## Available Providers

### 1. OpenAI Embedding Provider (Recommended for Production)

The OpenAI provider uses OpenAI's embedding models for high-quality vector representations.

**Supported Models:**
- `text-embedding-ada-002` (1536 dimensions) - Recommended
- `text-embedding-3-small` (1536 dimensions)
- `text-embedding-3-large` (3072 dimensions)

**Configuration:**

```python
from rag_agent import RAGAgent
from models.config import VectorDBConfig
import os

vector_db_config = VectorDBConfig(
    provider="faiss",
    embedding_model="text-embedding-ada-002",
    dimension=1536,
    connection_params={
        "index_path": "./faiss_index",
        "openai_api_key": os.getenv("OPENAI_API_KEY")
    }
)

agent = RAGAgent(vector_db_config=vector_db_config)
```

**Advantages:**
- High-quality embeddings
- Consistent performance
- Well-tested and reliable
- Good for semantic search

**Requirements:**
- OpenAI API key
- Internet connection
- API usage costs apply

### 2. Mock Embedding Provider (For Testing/Development)

The Mock provider generates random embeddings for testing without requiring API keys.

**Configuration:**

```python
from rag_agent import RAGAgent
from models.config import VectorDBConfig

vector_db_config = VectorDBConfig(
    provider="faiss",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Any name with "mock"
    dimension=384,
    connection_params={"index_path": "./faiss_index"}
)

agent = RAGAgent(vector_db_config=vector_db_config)
```

**Advantages:**
- No API key required
- Fast initialization
- Free to use
- Good for testing

**Limitations:**
- ⚠️ **NOT suitable for production**
- Random embeddings (not semantic)
- No real similarity matching
- Only for development/testing

## Automatic Provider Selection

The SDK automatically selects the appropriate provider based on the model name:

| Model Name Contains | Provider Selected |
|---------------------|-------------------|
| `openai` or `ada` | OpenAIEmbeddingProvider |
| `mock` | MockEmbeddingProvider |
| Default (unknown) | MockEmbeddingProvider (with warning) |

## Best Practices

### For Production

1. **Always use OpenAI embeddings:**
   ```python
   embedding_model="text-embedding-ada-002"
   ```

2. **Store API keys securely:**
   ```python
   # Use environment variables
   api_key=os.getenv("OPENAI_API_KEY")
   
   # Never hardcode API keys in source code!
   ```

3. **Enable caching:**
   ```python
   # Caching is enabled by default
   # Reduces API calls and costs
   ```

4. **Monitor usage:**
   - Track API calls
   - Set up billing alerts
   - Monitor embedding generation costs

### For Development/Testing

1. **Use mock embeddings:**
   ```python
   # Default configuration uses mock
   agent = RAGAgent()
   ```

2. **Switch to real embeddings before production:**
   ```python
   # Test with mock
   if os.getenv("ENV") == "production":
       embedding_model = "text-embedding-ada-002"
   else:
       embedding_model = "mock-embedding"
   ```

## Migration from Mock to Production

When moving from development to production:

1. **Update configuration:**
   ```python
   # Before (Development)
   vector_db_config = VectorDBConfig(
       provider="faiss",
       embedding_model="sentence-transformers/all-MiniLM-L6-v2",
       dimension=384
   )
   
   # After (Production)
   vector_db_config = VectorDBConfig(
       provider="faiss",
       embedding_model="text-embedding-ada-002",
       dimension=1536,  # Note: dimension changed!
       connection_params={
           "openai_api_key": os.getenv("OPENAI_API_KEY")
       }
   )
   ```

2. **Re-index your data:**
   - Embeddings from different models are not compatible
   - You must regenerate embeddings with the new model
   - Delete old FAISS index and recreate

3. **Update dimension:**
   - Mock: 384 dimensions
   - OpenAI Ada-002: 1536 dimensions
   - Ensure consistency across your system

## Troubleshooting

### "Using mock provider" Warning

**Problem:** You see a warning about using mock provider in production.

**Solution:**
```python
# Ensure your model name triggers OpenAI provider
embedding_model="text-embedding-ada-002"  # Contains "ada"
```

### API Key Not Found

**Problem:** OpenAI provider fails with authentication error.

**Solution:**
```python
# Set environment variable
export OPENAI_API_KEY="your-key-here"

# Or pass directly (not recommended)
connection_params={"openai_api_key": "your-key"}
```

### Dimension Mismatch

**Problem:** Error about dimension mismatch when loading index.

**Solution:**
- Ensure dimension matches your embedding model
- OpenAI Ada-002: 1536
- Mock: 384 (or custom)
- Delete and recreate index if dimension changed

## Example: Complete Production Setup

```python
#!/usr/bin/env python3
import os
from rag_agent import RAGAgent
from models.config import LLMConfig, VectorDBConfig, AgentConfig

# Production configuration
llm_config = LLMConfig(
    provider="openai",
    model="gpt-3.5-turbo",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7
)

vector_db_config = VectorDBConfig(
    provider="faiss",
    embedding_model="text-embedding-ada-002",  # Production embedding model
    dimension=1536,
    connection_params={
        "index_path": "./production_index",
        "metric": "cosine",
        "openai_api_key": os.getenv("OPENAI_API_KEY")
    }
)

agent_config = AgentConfig(
    max_context_length=50000,
    max_tool_calls=20,
    log_level="INFO"
)

# Initialize agent
agent = RAGAgent(
    llm_config=llm_config,
    vector_db_config=vector_db_config,
    agent_config=agent_config
)

# Use agent
response = await agent.query("Your query here")
print(response.answer)
```

## Summary

- ✅ **Production:** Use OpenAI embeddings (`text-embedding-ada-002`)
- ✅ **Development:** Use mock embeddings (default)
- ✅ **Security:** Store API keys in environment variables
- ✅ **Migration:** Re-index data when changing embedding models
- ⚠️ **Warning:** Mock embeddings are NOT suitable for production use

For more examples, see `examples/rag_agent_with_real_embeddings.py`.
