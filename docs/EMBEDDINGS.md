# Embedding Options in Lexora SDK

Lexora provides flexible embedding options to suit different production needs, from free local models to enterprise-grade APIs.

## 📋 Table of Contents

- [Overview](#overview)
- [Available Embedding Providers](#available-embedding-providers)
- [Quick Start Examples](#quick-start-examples)
- [Production Recommendations](#production-recommendations)
- [Custom Embedding Providers](#custom-embedding-providers)
- [Performance Comparison](#performance-comparison)

---

## Overview

Embeddings are vector representations of text that enable semantic search. Lexora supports multiple embedding providers, giving you flexibility in cost, performance, and privacy.

**You are NOT limited to OpenAI embeddings!** Lexora's architecture allows you to use:
- Free local models (HuggingFace, sentence-transformers)
- Cloud APIs (OpenAI, Cohere, Google)
- Custom implementations
- Mock embeddings for testing

---

## Available Embedding Providers

### 1. OpenAI Embeddings (Paid)

**Best for:** Production applications with budget for API costs

```python
from lexora.utils.embeddings import create_openai_embedding_manager

embedding_manager = create_openai_embedding_manager(
    model="text-embedding-ada-002",  # or "text-embedding-3-small"
    api_key="your-openai-api-key",
    enable_caching=True
)
```

**Pros:**
- High quality embeddings
- Reliable API
- 1536 dimensions

**Cons:**
- Costs money per token
- Requires internet connection
- Data sent to external API

**Pricing:** ~$0.0001 per 1K tokens

---

### 2. Mock Embeddings (Free)

**Best for:** Development, testing, and prototyping

```python
from lexora.utils.embeddings import create_mock_embedding_manager

embedding_manager = create_mock_embedding_manager(
    dimension=384,  # Match your vector DB
    model_name="mock-embedding",
    enable_caching=True
)
```

**Pros:**
- Completely free
- No API key required
- Fast and deterministic
- Good for testing

**Cons:**
- Not suitable for production
- Lower quality than real embeddings

---

### 3. HuggingFace / Sentence-Transformers (Free, Recommended)

**Best for:** Production use without API costs

```python
from lexora.utils.embeddings import BaseEmbeddingProvider, EmbeddingManager
from sentence_transformers import SentenceTransformer
from typing import List

class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """Free, local embedding provider using sentence-transformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    def get_model_name(self) -> str:
        """Get model name."""
        return self.model_name

# Usage
provider = HuggingFaceEmbeddingProvider()
embedding_manager = EmbeddingManager(provider=provider, enable_caching=True)
```

**Pros:**
- Completely free
- Runs locally (no API calls)
- Privacy-friendly (data stays on-premise)
- High quality embeddings
- Fast on GPU, acceptable on CPU
- Many models available

**Cons:**
- Requires installing sentence-transformers
- Uses local compute resources
- Initial model download required

**Popular Models:**
- `all-MiniLM-L6-v2` (384 dim) - Fast, good quality
- `all-mpnet-base-v2` (768 dim) - Better quality, slower
- `multi-qa-MiniLM-L6-cos-v1` (384 dim) - Optimized for Q&A

**Installation:**
```bash
pip install sentence-transformers
```

---

### 4. Google Gemini Embeddings (Free Tier Available)

**Best for:** Using Gemini ecosystem

```python
from lexora.utils.embeddings import BaseEmbeddingProvider
import google.generativeai as genai
from typing import List

class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Google Gemini embedding provider."""
    
    def __init__(self, api_key: str, model: str = "models/embedding-001"):
        genai.configure(api_key=api_key)
        self.model = model
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Gemini API."""
        result = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    
    def get_dimension(self) -> int:
        return 768  # Gemini embedding dimension
    
    def get_model_name(self) -> str:
        return self.model

# Usage
provider = GeminiEmbeddingProvider(api_key="your-gemini-api-key")
embedding_manager = EmbeddingManager(provider=provider)
```

**Pros:**
- Free tier available
- Good quality
- Integrates with Gemini LLM

**Cons:**
- Requires API key
- Rate limits on free tier

---

### 5. Cohere Embeddings

**Best for:** Enterprise applications

```python
from lexora.utils.embeddings import BaseEmbeddingProvider
import cohere
from typing import List

class CohereEmbeddingProvider(BaseEmbeddingProvider):
    """Cohere embedding provider."""
    
    def __init__(self, api_key: str, model: str = "embed-english-v3.0"):
        self.client = cohere.Client(api_key)
        self.model = model
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Cohere API."""
        response = self.client.embed(
            texts=[text],
            model=self.model,
            input_type="search_document"
        )
        return response.embeddings[0]
    
    def get_dimension(self) -> int:
        return 1024  # Cohere v3 dimension
    
    def get_model_name(self) -> str:
        return self.model
```

---

## Quick Start Examples

### Example 1: Free Local Embeddings (Recommended for Production)

```python
from lexora import RAGAgent
from lexora.models.config import VectorDBConfig
from sentence_transformers import SentenceTransformer

# Use free local embeddings
agent = RAGAgent(
    vector_db_config=VectorDBConfig(
        provider="faiss",
        dimension=384,  # all-MiniLM-L6-v2 dimension
        connection_params={
            "index_type": "Flat",
            "persist_directory": "./vector_db"
        }
    )
)

# The agent will use mock embeddings by default
# To use HuggingFace, configure the embedding manager
```

### Example 2: OpenAI Embeddings

```python
from lexora import RAGAgent
from lexora.models.config import LLMConfig, VectorDBConfig

agent = RAGAgent(
    llm_config=LLMConfig(
        provider="litellm",
        model="gpt-4",
        api_key="your-openai-key"
    ),
    vector_db_config=VectorDBConfig(
        provider="faiss",
        dimension=1536,  # OpenAI dimension
        connection_params={
            "embedding_model": "text-embedding-ada-002",
            "openai_api_key": "your-openai-key"
        }
    )
)
```

### Example 3: Testing with Mock Embeddings

```python
from lexora import RAGAgent

# Perfect for unit tests and development
agent = RAGAgent()  # Uses mock embeddings by default
```

---

## Production Recommendations

### For Cost-Conscious Applications
**Use HuggingFace/Sentence-Transformers**
- Zero API costs
- Good quality
- Privacy-friendly
- Scalable

### For Enterprise Applications
**Use OpenAI or Cohere**
- Highest quality
- Managed infrastructure
- Support available
- Predictable costs

### For Gemini Users
**Use Gemini Embeddings**
- Consistent ecosystem
- Free tier available
- Good integration

### For Development/Testing
**Use Mock Embeddings**
- Fast iteration
- No setup required
- Deterministic results

---

## Custom Embedding Providers

You can create custom embedding providers for any service:

```python
from lexora.utils.embeddings import BaseEmbeddingProvider
from typing import List

class CustomEmbeddingProvider(BaseEmbeddingProvider):
    """Your custom embedding provider."""
    
    def __init__(self, **config):
        # Initialize your embedding service
        pass
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        # Your implementation here
        embedding = your_embedding_function(text)
        return embedding
    
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return 768  # Your dimension
    
    def get_model_name(self) -> str:
        """Return model name."""
        return "custom-model"
```

---

## Performance Comparison

| Provider | Cost | Speed | Quality | Privacy | Setup |
|----------|------|-------|---------|---------|-------|
| **HuggingFace** | Free | Fast (GPU) | High | ✅ Local | Easy |
| **OpenAI** | $0.0001/1K | Fast | Highest | ❌ Cloud | Easy |
| **Gemini** | Free tier | Fast | High | ❌ Cloud | Easy |
| **Cohere** | Paid | Fast | Highest | ❌ Cloud | Easy |
| **Mock** | Free | Fastest | Low | ✅ Local | None |

---

## Best Practices

1. **Use caching** - Enable embedding caching to avoid regenerating embeddings
2. **Match dimensions** - Ensure your vector DB dimension matches your embedding model
3. **Batch processing** - Generate embeddings in batches for better performance
4. **Monitor costs** - Track API usage if using paid providers
5. **Test locally** - Use mock embeddings for development, real embeddings for production

---

## Troubleshooting

### "OpenAI authentication failed"
- Check your API key is correct
- Ensure you have credits in your OpenAI account
- Consider using free alternatives like HuggingFace

### "Dimension mismatch"
- Ensure your vector DB dimension matches your embedding model
- OpenAI: 1536, HuggingFace (MiniLM): 384, Gemini: 768

### "Out of memory"
- Use smaller batch sizes
- Use a smaller embedding model
- Consider cloud-based embeddings

---

## Additional Resources

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Google Gemini API](https://ai.google.dev/docs)
- [Cohere Embeddings](https://docs.cohere.com/docs/embeddings)

---

**Need help?** Open an issue on GitHub or check our [Getting Started Guide](./GETTING_STARTED.md).
