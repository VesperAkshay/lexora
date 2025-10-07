# Best Practices Guide

This guide covers best practices for building production-ready applications with Lexora.

## Table of Contents

- [Error Handling](#error-handling)
- [Performance Optimization](#performance-optimization)
- [Security](#security)
- [Testing](#testing)
- [Monitoring and Logging](#monitoring-and-logging)
- [Deployment](#deployment)
- [Data Management](#data-management)
- [Scalability](#scalability)

## Error Handling

### Comprehensive Error Handling

```python
from lexora import RAGAgent
from lexora.exceptions import (
    LexoraError,
    CorpusNotFoundError,
    ToolExecutionError,
    VectorDBError,
    LLMError
)
import logging

logger = logging.getLogger(__name__)

async def safe_query(agent: RAGAgent, query: str, corpus_name: str, max_retries: int = 3):
    """Query with comprehensive error handling and retries."""
    
    for attempt in range(max_retries):
        try:
            response = await agent.query(query, corpus_name=corpus_name)
            return response
            
        except CorpusNotFoundError as e:
            logger.error(f"Corpus not found: {corpus_name}")
            raise  # Don't retry for this error
            
        except VectorDBError as e:
            logger.warning(f"Vector DB error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
        except LLMError as e:
            logger.warning(f"LLM error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                # Fallback to simpler response
                return await fallback_response(query, corpus_name)
            await asyncio.sleep(1)
            
        except ToolExecutionError as e:
            logger.error(f"Tool execution failed: {e}")
            # Try without problematic tool
            return await agent.query(query, corpus_name=corpus_name, skip_tools=True)
            
        except LexoraError as e:
            logger.error(f"Lexora error: {e}")
            raise
            
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            raise

async def fallback_response(query: str, corpus_name: str):
    """Provide fallback response when LLM fails."""
    return {
        "answer": "I'm having trouble generating a response right now. Please try again later.",
        "confidence": 0.0,
        "sources": [],
        "fallback": True
    }
```

### Graceful Degradation

```python
class ResilientRAGAgent:
    """RAG agent with graceful degradation."""
    
    def __init__(self, agent: RAGAgent):
        self.agent = agent
        self.fallback_enabled = True
    
    async def query(self, query: str, corpus_name: str = None):
        """Query with fallback strategies."""
        try:
            # Try primary query
            return await self.agent.query(query, corpus_name=corpus_name)
            
        except VectorDBError:
            if self.fallback_enabled:
                # Fallback to keyword search
                logger.warning("Vector DB unavailable, using keyword search")
                return await self._keyword_search(query, corpus_name)
            raise
            
        except LLMError:
            if self.fallback_enabled:
                # Return raw documents without generation
                logger.warning("LLM unavailable, returning raw documents")
                return await self._raw_document_search(query, corpus_name)
            raise
    
    async def _keyword_search(self, query: str, corpus_name: str):
        """Fallback keyword-based search."""
        # Implement simple keyword matching
        pass
    
    async def _raw_document_search(self, query: str, corpus_name: str):
        """Return raw documents without LLM generation."""
        results = await self.agent.search_documents(corpus_name, query)
        return {
            "answer": "Here are relevant documents (LLM unavailable):",
            "sources": results,
            "confidence": 0.5,
            "fallback": True
        }
```

## Performance Optimization

### Connection Pooling

```python
from lexora import RAGAgent
from functools import lru_cache

@lru_cache(maxsize=1)
def get_agent():
    """Singleton agent instance with connection pooling."""
    return RAGAgent(
        vector_db_config={
            "connection_pool_size": 10,
            "connection_timeout": 30
        }
    )

# Reuse the same agent instance
agent = get_agent()
```

### Caching Strategies

```python
from functools import lru_cache
from datetime import datetime, timedelta
import hashlib

class CachedRAGAgent:
    """RAG agent with response caching."""
    
    def __init__(self, agent: RAGAgent, cache_ttl: int = 3600):
        self.agent = agent
        self.cache = {}
        self.cache_ttl = cache_ttl
    
    def _cache_key(self, query: str, corpus_name: str) -> str:
        """Generate cache key."""
        key_str = f"{query}:{corpus_name}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def query(self, query: str, corpus_name: str = None):
        """Query with caching."""
        cache_key = self._cache_key(query, corpus_name)
        
        # Check cache
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached_data
        
        # Query and cache
        response = await self.agent.query(query, corpus_name=corpus_name)
        self.cache[cache_key] = (response, datetime.now())
        
        return response
    
    def clear_cache(self):
        """Clear the cache."""
        self.cache.clear()
```

### Batch Processing

```python
async def optimized_batch_add(agent: RAGAgent, corpus_name: str, documents: list):
    """Optimized batch document addition."""
    
    # Optimal batch size based on testing
    BATCH_SIZE = 100
    
    # Pre-process documents
    processed_docs = []
    for doc in documents:
        # Clean and validate
        if len(doc['content']) < 10:
            continue
        processed_docs.append(doc)
    
    # Add in parallel batches
    tasks = []
    for i in range(0, len(processed_docs), BATCH_SIZE):
        batch = processed_docs[i:i + BATCH_SIZE]
        task = agent.add_documents(corpus_name, batch)
        tasks.append(task)
        
        # Limit concurrent batches
        if len(tasks) >= 5:
            await asyncio.gather(*tasks)
            tasks = []
    
    # Process remaining
    if tasks:
        await asyncio.gather(*tasks)
```

### Query Optimization

```python
class OptimizedQueryStrategy:
    """Optimize queries for better performance."""
    
    def __init__(self, agent: RAGAgent):
        self.agent = agent
    
    async def smart_query(self, query: str, corpus_name: str):
        """Intelligently optimize query execution."""
        
        # Short queries: use fewer results
        if len(query.split()) < 5:
            top_k = 3
        else:
            top_k = 5
        
        # Check if query needs full LLM generation
        if self._is_simple_query(query):
            # Direct document retrieval
            results = await self.agent.search_documents(
                corpus_name=corpus_name,
                query=query,
                top_k=top_k
            )
            return self._format_simple_response(results)
        
        # Full RAG pipeline
        return await self.agent.query(
            query=query,
            corpus_name=corpus_name,
            top_k=top_k
        )
    
    def _is_simple_query(self, query: str) -> bool:
        """Check if query is simple enough for direct retrieval."""
        simple_patterns = ['what is', 'define', 'who is', 'when was']
        return any(pattern in query.lower() for pattern in simple_patterns)
    
    def _format_simple_response(self, results):
        """Format simple response from documents."""
        if results:
            return {
                "answer": results[0].content,
                "confidence": results[0].score,
                "sources": results,
                "simple": True
            }
        return None
```

## Security

### Input Validation

```python
import re
from typing import Optional

class InputValidator:
    """Validate and sanitize user inputs."""
    
    @staticmethod
    def validate_query(query: str, max_length: int = 1000) -> str:
        """Validate and sanitize query input."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if len(query) > max_length:
            raise ValueError(f"Query too long (max {max_length} characters)")
        
        # Remove potentially harmful characters
        sanitized = re.sub(r'[<>{}]', '', query)
        
        return sanitized.strip()
    
    @staticmethod
    def validate_corpus_name(name: str) -> str:
        """Validate corpus name."""
        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
            raise ValueError("Corpus name can only contain letters, numbers, hyphens, and underscores")
        
        if len(name) > 50:
            raise ValueError("Corpus name too long (max 50 characters)")
        
        return name
    
    @staticmethod
    def validate_metadata(metadata: dict) -> dict:
        """Validate metadata dictionary."""
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary")
        
        # Limit metadata size
        if len(str(metadata)) > 10000:
            raise ValueError("Metadata too large")
        
        # Sanitize values
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                sanitized[key] = value[:500]  # Limit string length
            else:
                sanitized[key] = value
        
        return sanitized

# Usage
validator = InputValidator()

async def safe_query_endpoint(query: str, corpus_name: str):
    """Secure query endpoint with validation."""
    try:
        # Validate inputs
        clean_query = validator.validate_query(query)
        clean_corpus = validator.validate_corpus_name(corpus_name)
        
        # Execute query
        response = await agent.query(clean_query, corpus_name=clean_corpus)
        return response
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise
```

### API Key Management

```python
import os
from cryptography.fernet import Fernet

class SecureConfig:
    """Secure configuration management."""
    
    def __init__(self):
        self.cipher = Fernet(self._get_encryption_key())
    
    def _get_encryption_key(self) -> bytes:
        """Get or create encryption key."""
        key_file = ".encryption_key"
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
    
    def encrypt_api_key(self, api_key: str) -> str:
        """Encrypt API key."""
        return self.cipher.encrypt(api_key.encode()).decode()
    
    def decrypt_api_key(self, encrypted_key: str) -> str:
        """Decrypt API key."""
        return self.cipher.decrypt(encrypted_key.encode()).decode()
    
    def get_api_key(self, key_name: str) -> Optional[str]:
        """Get API key from environment or encrypted storage."""
        # Try environment variable first
        key = os.getenv(key_name)
        if key:
            return key
        
        # Try encrypted storage
        encrypted_file = f".{key_name.lower()}.enc"
        if os.path.exists(encrypted_file):
            with open(encrypted_file, 'r') as f:
                encrypted = f.read()
            return self.decrypt_api_key(encrypted)
        
        return None

# Usage
config = SecureConfig()
api_key = config.get_api_key("OPENAI_API_KEY")
```

### Rate Limiting

```python
from collections import defaultdict
from datetime import datetime, timedelta
import asyncio

class RateLimiter:
    """Rate limiter for API endpoints."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    async def check_rate_limit(self, user_id: str) -> bool:
        """Check if user has exceeded rate limit."""
        now = datetime.now()
        window_start = now - timedelta(seconds=self.window_seconds)
        
        # Clean old requests
        self.requests[user_id] = [
            req_time for req_time in self.requests[user_id]
            if req_time > window_start
        ]
        
        # Check limit
        if len(self.requests[user_id]) >= self.max_requests:
            return False
        
        # Record request
        self.requests[user_id].append(now)
        return True
    
    async def wait_if_needed(self, user_id: str):
        """Wait if rate limit exceeded."""
        while not await self.check_rate_limit(user_id):
            await asyncio.sleep(1)

# Usage
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

async def rate_limited_query(user_id: str, query: str):
    """Query with rate limiting."""
    if not await rate_limiter.check_rate_limit(user_id):
        raise Exception("Rate limit exceeded. Please try again later.")
    
    return await agent.query(query)
```

## Testing

### Unit Testing

```python
import pytest
from lexora import RAGAgent

@pytest.fixture
async def agent():
    """Create test agent."""
    agent = RAGAgent()
    yield agent
    # Cleanup
    await agent.cleanup()

@pytest.fixture
async def test_corpus(agent):
    """Create test corpus with sample data."""
    corpus_name = "test_corpus"
    await agent.create_corpus(corpus_name)
    
    documents = [
        {"content": "Python is a programming language"},
        {"content": "Machine learning uses algorithms"}
    ]
    await agent.add_documents(corpus_name, documents)
    
    yield corpus_name
    
    # Cleanup
    await agent.delete_corpus(corpus_name, confirm_deletion=corpus_name)

@pytest.mark.asyncio
async def test_query(agent, test_corpus):
    """Test basic query functionality."""
    response = await agent.query("What is Python?", corpus_name=test_corpus)
    
    assert response.answer is not None
    assert len(response.answer) > 0
    assert response.confidence >= 0.0
    assert response.confidence <= 1.0
    assert len(response.sources) > 0

@pytest.mark.asyncio
async def test_query_with_filters(agent, test_corpus):
    """Test query with metadata filters."""
    response = await agent.query(
        "programming",
        corpus_name=test_corpus,
        metadata_filters={"topic": "programming"}
    )
    
    assert response is not None
```

### Integration Testing

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_workflow():
    """Test complete workflow."""
    agent = RAGAgent()
    corpus_name = "integration_test"
    
    try:
        # Create corpus
        await agent.create_corpus(corpus_name)
        
        # Add documents
        documents = [
            {"content": "Test document 1", "metadata": {"id": 1}},
            {"content": "Test document 2", "metadata": {"id": 2}}
        ]
        result = await agent.add_documents(corpus_name, documents)
        assert result.documents_added == 2
        
        # Query
        response = await agent.query("test", corpus_name=corpus_name)
        assert response.answer is not None
        
        # List corpora
        corpora = await agent.list_corpora()
        assert any(c["name"] == corpus_name for c in corpora)
        
    finally:
        # Cleanup
        await agent.delete_corpus(corpus_name, confirm_deletion=corpus_name)
```

## Monitoring and Logging

### Structured Logging

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    """Structured logging for better observability."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(self.JSONFormatter())
        self.logger.addHandler(handler)
    
    class JSONFormatter(logging.Formatter):
        """Format logs as JSON."""
        
        def format(self, record):
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName
            }
            
            # Add extra fields
            if hasattr(record, 'extra'):
                log_data.update(record.extra)
            
            return json.dumps(log_data)
    
    def log_query(self, query: str, corpus_name: str, response_time: float, confidence: float):
        """Log query with metrics."""
        self.logger.info(
            "Query executed",
            extra={
                "query_length": len(query),
                "corpus_name": corpus_name,
                "response_time_ms": response_time * 1000,
                "confidence": confidence
            }
        )

# Usage
logger = StructuredLogger("lexora")

async def monitored_query(query: str, corpus_name: str):
    """Query with monitoring."""
    start_time = time.time()
    
    try:
        response = await agent.query(query, corpus_name=corpus_name)
        response_time = time.time() - start_time
        
        logger.log_query(query, corpus_name, response_time, response.confidence)
        
        return response
        
    except Exception as e:
        logger.logger.error(f"Query failed: {e}", extra={
            "query": query,
            "corpus_name": corpus_name,
            "error_type": type(e).__name__
        })
        raise
```

### Performance Metrics

```python
from dataclasses import dataclass
from typing import List
import statistics

@dataclass
class QueryMetrics:
    """Query performance metrics."""
    response_time: float
    confidence: float
    num_sources: int
    corpus_name: str

class MetricsCollector:
    """Collect and analyze performance metrics."""
    
    def __init__(self):
        self.metrics: List[QueryMetrics] = []
    
    def record_query(self, metrics: QueryMetrics):
        """Record query metrics."""
        self.metrics.append(metrics)
    
    def get_stats(self) -> dict:
        """Get aggregated statistics."""
        if not self.metrics:
            return {}
        
        response_times = [m.response_time for m in self.metrics]
        confidences = [m.confidence for m in self.metrics]
        
        return {
            "total_queries": len(self.metrics),
            "avg_response_time": statistics.mean(response_times),
            "p95_response_time": statistics.quantiles(response_times, n=20)[18],
            "avg_confidence": statistics.mean(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences)
        }
    
    def get_corpus_stats(self, corpus_name: str) -> dict:
        """Get stats for specific corpus."""
        corpus_metrics = [m for m in self.metrics if m.corpus_name == corpus_name]
        
        if not corpus_metrics:
            return {}
        
        response_times = [m.response_time for m in corpus_metrics]
        
        return {
            "queries": len(corpus_metrics),
            "avg_response_time": statistics.mean(response_times),
            "avg_confidence": statistics.mean([m.confidence for m in corpus_metrics])
        }

# Usage
metrics = MetricsCollector()

async def tracked_query(query: str, corpus_name: str):
    """Query with metrics tracking."""
    start_time = time.time()
    response = await agent.query(query, corpus_name=corpus_name)
    response_time = time.time() - start_time
    
    metrics.record_query(QueryMetrics(
        response_time=response_time,
        confidence=response.confidence,
        num_sources=len(response.sources),
        corpus_name=corpus_name
    ))
    
    return response

# Get stats
stats = metrics.get_stats()
print(f"Average response time: {stats['avg_response_time']:.3f}s")
print(f"P95 response time: {stats['p95_response_time']:.3f}s")
```

## Deployment

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Install Lexora
RUN pip install -e .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from lexora import RAGAgent; agent = RAGAgent(); print('healthy')"

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  lexora-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LEXORA_LLM_PROVIDER=openai
      - LEXORA_LLM_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./faiss_storage:/app/faiss_storage
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lexora-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: lexora-api
  template:
    metadata:
      labels:
        app: lexora-api
    spec:
      containers:
      - name: lexora-api
        image: lexora/rag-sdk:latest
        ports:
        - containerPort: 8000
        env:
        - name: LEXORA_LLM_API_KEY
          valueFrom:
            secretKeyRef:
              name: lexora-secrets
              key: llm-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

---

For more best practices and patterns, check our [GitHub Discussions](https://github.com/your-org/lexora-rag-sdk/discussions).
