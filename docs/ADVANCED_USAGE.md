# Advanced Usage Guide

This guide covers advanced features and patterns for building sophisticated applications with Lexora.

## Table of Contents

- [Custom Tools](#custom-tools)
- [Metadata Filtering](#metadata-filtering)
- [Batch Operations](#batch-operations)
- [Custom Embeddings](#custom-embeddings)
- [Query Optimization](#query-optimization)
- [Streaming Responses](#streaming-responses)
- [Multi-Corpus Queries](#multi-corpus-queries)
- [Document Chunking Strategies](#document-chunking-strategies)

## Custom Tools

Create custom tools to extend Lexora's capabilities.

### Basic Custom Tool

```python
from lexora.tools.base import BaseTool, ToolParameter
from typing import Dict, Any

class CustomSearchTool(BaseTool):
    """Custom tool for specialized search operations."""
    
    @property
    def name(self) -> str:
        return "custom_search"
    
    @property
    def description(self) -> str:
        return "Performs custom search with advanced filters"
    
    def _setup_parameters(self) -> None:
        self._parameters = [
            ToolParameter(
                name="query",
                type="string",
                description="Search query",
                required=True
            ),
            ToolParameter(
                name="filters",
                type="object",
                description="Advanced filter criteria",
                required=False
            ),
            ToolParameter(
                name="max_results",
                type="integer",
                description="Maximum number of results",
                required=False,
                default=10
            )
        ]
    
    async def _execute(self, query: str, filters: Dict = None, max_results: int = 10, **kwargs) -> Dict[str, Any]:
        """Execute the custom search."""
        # Your custom search logic here
        results = []
        
        # Example: Apply custom filtering logic
        if filters:
            # Process filters
            pass
        
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }

# Register and use the tool
agent = RAGAgent()
custom_tool = CustomSearchTool()
agent.add_tool(custom_tool)

# Execute the tool
result = await agent.execute_tool(
    "custom_search",
    query="machine learning",
    filters={"difficulty": "advanced"},
    max_results=5
)
```

### Tool with External API Integration

```python
import aiohttp
from lexora.tools.base import BaseTool, ToolParameter

class WeatherTool(BaseTool):
    """Tool that fetches weather data from external API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        super().__init__()
    
    @property
    def name(self) -> str:
        return "get_weather"
    
    @property
    def description(self) -> str:
        return "Fetches current weather information for a location"
    
    def _setup_parameters(self) -> None:
        self._parameters = [
            ToolParameter(
                name="location",
                type="string",
                description="City name or coordinates",
                required=True
            ),
            ToolParameter(
                name="units",
                type="string",
                description="Temperature units (metric/imperial)",
                required=False,
                default="metric"
            )
        ]
    
    async def _execute(self, location: str, units: str = "metric", **kwargs):
        """Fetch weather data from API."""
        async with aiohttp.ClientSession() as session:
            url = f"https://api.weather.com/v1/current"
            params = {
                "location": location,
                "units": units,
                "apikey": self.api_key
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "location": location,
                        "temperature": data.get("temp"),
                        "condition": data.get("condition"),
                        "humidity": data.get("humidity")
                    }
                else:
                    raise Exception(f"Weather API error: {response.status}")

# Usage
weather_tool = WeatherTool(api_key="your-api-key")
agent.add_tool(weather_tool)

result = await agent.execute_tool("get_weather", location="New York")
```

## Metadata Filtering

Advanced filtering techniques for precise document retrieval.

### Basic Metadata Filters

```python
# Exact match
response = await agent.query(
    "Python tutorials",
    metadata_filters={"difficulty": "beginner"}
)

# Multiple conditions (AND)
response = await agent.query(
    "web development",
    metadata_filters={
        "topic": "programming",
        "language": "python",
        "difficulty": "intermediate"
    }
)
```

### Complex Filters

```python
# OR conditions
response = await agent.query(
    "programming concepts",
    metadata_filters={
        "$or": [
            {"language": "python"},
            {"language": "javascript"}
        ]
    }
)

# Nested conditions
response = await agent.query(
    "advanced topics",
    metadata_filters={
        "$and": [
            {"topic": "ai"},
            {
                "$or": [
                    {"difficulty": "advanced"},
                    {"difficulty": "expert"}
                ]
            }
        ]
    }
)

# Range queries
response = await agent.query(
    "recent articles",
    metadata_filters={
        "year": {"$gte": 2020, "$lte": 2024},
        "views": {"$gt": 1000}
    }
)

# Array operations
response = await agent.query(
    "multi-tag search",
    metadata_filters={
        "tags": {"$in": ["python", "machine-learning", "tutorial"]}
    }
)
```

### Dynamic Filter Builder

```python
class FilterBuilder:
    """Helper class for building complex filters."""
    
    def __init__(self):
        self.filters = {}
    
    def add_exact(self, field: str, value: Any):
        """Add exact match filter."""
        self.filters[field] = value
        return self
    
    def add_range(self, field: str, min_val=None, max_val=None):
        """Add range filter."""
        range_filter = {}
        if min_val is not None:
            range_filter["$gte"] = min_val
        if max_val is not None:
            range_filter["$lte"] = max_val
        self.filters[field] = range_filter
        return self
    
    def add_in(self, field: str, values: list):
        """Add IN filter."""
        self.filters[field] = {"$in": values}
        return self
    
    def build(self):
        """Build the final filter dictionary."""
        return self.filters

# Usage
filters = (FilterBuilder()
    .add_exact("topic", "programming")
    .add_range("year", min_val=2020)
    .add_in("language", ["python", "javascript"])
    .build())

response = await agent.query("coding tutorials", metadata_filters=filters)
```

## Batch Operations

Efficiently process large volumes of documents and queries.

### Batch Document Addition

```python
async def batch_add_documents(agent: RAGAgent, corpus_name: str, documents: list, batch_size: int = 100):
    """Add documents in batches for better performance."""
    total_added = 0
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        result = await agent.add_documents(corpus_name, batch)
        total_added += result.documents_added
        print(f"Progress: {total_added}/{len(documents)} documents added")
    
    return total_added

# Usage
documents = [...]  # Your large document collection
total = await batch_add_documents(agent, "large_corpus", documents, batch_size=50)
print(f"Total documents added: {total}")
```

### Parallel Query Processing

```python
import asyncio

async def process_queries_parallel(agent: RAGAgent, queries: list, corpus_name: str, max_concurrent: int = 10):
    """Process multiple queries in parallel with concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_query(query: str):
        async with semaphore:
            return await agent.query(query, corpus_name=corpus_name)
    
    tasks = [process_query(q) for q in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return results

# Usage
queries = ["What is Python?", "Explain ML", "What is FastAPI?"]
results = await process_queries_parallel(agent, queries, "my_corpus", max_concurrent=5)

for query, result in zip(queries, results):
    if isinstance(result, Exception):
        print(f"Error for '{query}': {result}")
    else:
        print(f"Q: {query}\nA: {result.answer}\n")
```

### Bulk Document Processing with Progress

```python
from tqdm.asyncio import tqdm

async def bulk_process_with_progress(agent: RAGAgent, corpus_name: str, file_paths: list):
    """Process files with progress bar."""
    documents = []
    
    # Read files with progress
    for file_path in tqdm(file_paths, desc="Reading files"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            documents.append({
                "content": content,
                "metadata": {
                    "source": file_path,
                    "filename": os.path.basename(file_path)
                }
            })
    
    # Add documents in batches with progress
    batch_size = 100
    total_added = 0
    
    for i in tqdm(range(0, len(documents), batch_size), desc="Adding documents"):
        batch = documents[i:i + batch_size]
        result = await agent.add_documents(corpus_name, batch)
        total_added += result.documents_added
    
    return total_added
```

## Custom Embeddings

Implement custom embedding strategies for specialized use cases.

### Custom Embedding Provider

```python
from lexora.vector_db.base import BaseEmbeddingProvider
import numpy as np

class CustomEmbeddingProvider(BaseEmbeddingProvider):
    """Custom embedding provider with specialized logic."""
    
    def __init__(self, model_name: str, dimension: int):
        self.model_name = model_name
        self.dimension = dimension
        # Initialize your model here
    
    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        # Your custom embedding logic
        embedding = np.random.rand(self.dimension)  # Replace with actual logic
        return embedding
    
    async def embed_batch(self, texts: list) -> list:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embedding = await self.embed_text(text)
            embeddings.append(embedding)
        return embeddings

# Usage
custom_embedder = CustomEmbeddingProvider(model_name="custom-model", dimension=512)
agent = RAGAgent(embedding_provider=custom_embedder)
```

### Hybrid Embedding Strategy

```python
class HybridEmbeddingProvider(BaseEmbeddingProvider):
    """Combines multiple embedding strategies."""
    
    def __init__(self, providers: list, weights: list = None):
        self.providers = providers
        self.weights = weights or [1.0 / len(providers)] * len(providers)
    
    async def embed_text(self, text: str) -> np.ndarray:
        """Generate hybrid embedding."""
        embeddings = []
        
        for provider in self.providers:
            emb = await provider.embed_text(text)
            embeddings.append(emb)
        
        # Weighted combination
        hybrid = np.zeros_like(embeddings[0])
        for emb, weight in zip(embeddings, self.weights):
            hybrid += emb * weight
        
        # Normalize
        hybrid = hybrid / np.linalg.norm(hybrid)
        return hybrid

# Usage
provider1 = CustomEmbeddingProvider("model1", 512)
provider2 = CustomEmbeddingProvider("model2", 512)

hybrid_provider = HybridEmbeddingProvider(
    providers=[provider1, provider2],
    weights=[0.6, 0.4]
)

agent = RAGAgent(embedding_provider=hybrid_provider)
```

## Query Optimization

Techniques for improving query performance and relevance.

### Query Rewriting

```python
class QueryOptimizer:
    """Optimize queries for better retrieval."""
    
    def __init__(self, agent: RAGAgent):
        self.agent = agent
    
    async def expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms."""
        # Use LLM to expand query
        expansion_prompt = f"Expand this query with related terms: {query}"
        expanded = await self.agent.llm.generate(expansion_prompt)
        return expanded
    
    async def rewrite_query(self, query: str) -> str:
        """Rewrite query for better matching."""
        rewrite_prompt = f"Rewrite this query for better search: {query}"
        rewritten = await self.agent.llm.generate(rewrite_prompt)
        return rewritten
    
    async def optimized_query(self, query: str, corpus_name: str):
        """Execute optimized query."""
        # Try original query
        response1 = await self.agent.query(query, corpus_name=corpus_name)
        
        # Try expanded query
        expanded = await self.expand_query(query)
        response2 = await self.agent.query(expanded, corpus_name=corpus_name)
        
        # Return best result
        return response1 if response1.confidence > response2.confidence else response2

# Usage
optimizer = QueryOptimizer(agent)
result = await optimizer.optimized_query("ML algorithms", "tech_kb")
```

### Result Re-ranking

```python
class ResultReranker:
    """Re-rank search results for better relevance."""
    
    def __init__(self, agent: RAGAgent):
        self.agent = agent
    
    async def rerank_results(self, query: str, results: list, top_k: int = 5):
        """Re-rank results using cross-encoder or other methods."""
        scored_results = []
        
        for result in results:
            # Calculate relevance score
            score = await self._calculate_relevance(query, result.content)
            scored_results.append((score, result))
        
        # Sort by score
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        return [result for _, result in scored_results[:top_k]]
    
    async def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score."""
        # Implement your scoring logic
        # Could use cross-encoder, BM25, or other methods
        return 0.0  # Placeholder

# Usage
reranker = ResultReranker(agent)
results = await agent.search_documents("my_corpus", "query")
reranked = await reranker.rerank_results("query", results, top_k=5)
```

## Multi-Corpus Queries

Query across multiple corpora simultaneously.

```python
async def multi_corpus_query(agent: RAGAgent, query: str, corpus_names: list, top_k: int = 5):
    """Query multiple corpora and combine results."""
    all_results = []
    
    for corpus_name in corpus_names:
        try:
            response = await agent.query(query, corpus_name=corpus_name, top_k=top_k)
            all_results.extend(response.sources)
        except Exception as e:
            print(f"Error querying {corpus_name}: {e}")
    
    # Sort by relevance score
    all_results.sort(key=lambda x: x.score, reverse=True)
    
    # Take top results
    top_results = all_results[:top_k]
    
    # Generate combined answer
    combined_context = "\n\n".join([r.content for r in top_results])
    answer = await agent.llm.generate(
        f"Based on this context, answer: {query}\n\nContext:\n{combined_context}"
    )
    
    return {
        "answer": answer,
        "sources": top_results,
        "corpora_searched": corpus_names
    }

# Usage
result = await multi_corpus_query(
    agent,
    "What is machine learning?",
    corpus_names=["tech_docs", "research_papers", "tutorials"],
    top_k=3
)
```

## Document Chunking Strategies

Advanced strategies for splitting documents into optimal chunks.

### Semantic Chunking

```python
class SemanticChunker:
    """Chunk documents based on semantic boundaries."""
    
    def __init__(self, max_chunk_size: int = 1000, overlap: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
    
    def chunk_by_paragraphs(self, text: str) -> list:
        """Chunk by paragraph boundaries."""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def chunk_by_sentences(self, text: str) -> list:
        """Chunk by sentence boundaries."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def chunk_with_overlap(self, text: str) -> list:
        """Create overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.max_chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.overlap
        
        return chunks

# Usage
chunker = SemanticChunker(max_chunk_size=500, overlap=50)
chunks = chunker.chunk_by_paragraphs(long_document)

documents = [
    {"content": chunk, "metadata": {"chunk_index": i}}
    for i, chunk in enumerate(chunks)
]

await agent.add_documents("my_corpus", documents)
```

---

For more examples and patterns, check out our [Examples Repository](https://github.com/your-org/lexora-examples).
