# Frequently Asked Questions (FAQ)

## General Questions

### What is Lexora?

Lexora is a production-ready Agentic RAG (Retrieval-Augmented Generation) SDK that enables developers to build intelligent applications with minimal configuration. It combines vector databases, large language models, and intelligent agents to provide seamless document retrieval and generation capabilities.

### What are the main use cases for Lexora?

- **Knowledge Base Systems**: Build intelligent Q&A systems for documentation
- **Customer Support**: Create AI-powered support chatbots
- **Research Tools**: Enable semantic search across research papers
- **Content Management**: Organize and retrieve content intelligently
- **Enterprise Search**: Build internal search systems for company knowledge
- **Educational Platforms**: Create interactive learning assistants

### Is Lexora free to use?

Yes, Lexora is open-source and free to use under the MIT License. However, you may incur costs from third-party services like OpenAI API or cloud hosting.

### What programming languages does Lexora support?

Lexora is built in Python and currently supports Python 3.8+. We're considering support for other languages in the future.

## Installation & Setup

### What are the system requirements?

- **Python**: 3.8 or higher
- **RAM**: 2GB minimum, 4GB+ recommended
- **Storage**: 1GB free space
- **OS**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)

### How do I install Lexora?

```bash
# Clone the repository
git clone https://github.com/your-org/lexora-rag-sdk.git
cd lexora-rag-sdk

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Do I need an OpenAI API key?

No, Lexora works with a mock LLM provider by default for testing. However, for production use, you'll want to configure a real LLM provider like OpenAI.

### How do I configure Lexora?

You can configure Lexora in three ways:

1. **Programmatically**:
```python
from lexora import RAGAgent, LLMConfig

agent = RAGAgent(
    llm_config=LLMConfig(provider="openai", api_key="your-key")
)
```

2. **Configuration File**:
```python
agent = RAGAgent.from_yaml("config.yaml")
```

3. **Environment Variables**:
```bash
export LEXORA_LLM_PROVIDER=openai
export LEXORA_LLM_API_KEY=your-key
```

## Usage Questions

### How do I create a knowledge base?

```python
from lexora import RAGAgent

agent = RAGAgent()

# Create a corpus (knowledge base)
await agent.create_corpus(
    corpus_name="my_kb",
    description="My knowledge base"
)

# Add documents
documents = [
    {"content": "Your document content here"},
    {"content": "Another document"}
]
await agent.add_documents("my_kb", documents)
```

### How many documents can I add to a corpus?

There's no hard limit, but performance considerations apply:
- **Small corpus** (< 1K documents): Excellent performance
- **Medium corpus** (1K-10K documents): Good performance
- **Large corpus** (10K-100K documents): Requires optimization
- **Very large corpus** (> 100K documents): Consider sharding or external vector DB

### How do I query my knowledge base?

```python
response = await agent.query(
    "What is machine learning?",
    corpus_name="my_kb"
)

print(response.answer)
print(f"Confidence: {response.confidence}")
```

### Can I use multiple corpora?

Yes! You can create multiple corpora and query them separately or together:

```python
# Create multiple corpora
await agent.create_corpus("tech_docs")
await agent.create_corpus("research_papers")

# Query specific corpus
response = await agent.query("Python", corpus_name="tech_docs")

# Query across multiple corpora (custom implementation)
# See Advanced Usage guide for multi-corpus queries
```

### How do I filter results by metadata?

```python
response = await agent.query(
    "Python tutorials",
    metadata_filters={
        "difficulty": "beginner",
        "topic": "programming"
    }
)
```

### What document formats are supported?

Lexora accepts documents as dictionaries with `content` and optional `metadata`:

```python
document = {
    "content": "Your text content",
    "metadata": {
        "source": "file.txt",
        "author": "John Doe",
        "date": "2024-01-01"
    }
}
```

For file uploads, you'll need to extract text first (using libraries like PyPDF2, python-docx, etc.).

## Performance Questions

### How fast is Lexora?

Performance depends on several factors:
- **Query Response Time**: < 100ms for small corpora (< 1K documents)
- **Document Addition**: 10,000+ documents/second with batching
- **Concurrent Queries**: 100+ queries/second

### How can I improve query performance?

1. **Use batching** for document addition
2. **Enable caching** for frequent queries
3. **Optimize chunk size** (500-1000 characters recommended)
4. **Use metadata filters** to narrow search scope
5. **Implement connection pooling**
6. **Consider GPU acceleration** for embeddings

### Does Lexora support GPU acceleration?

Yes, if you're using embedding models that support GPU (like sentence-transformers), Lexora will automatically use available GPUs.

### How much memory does Lexora use?

- **Base memory**: ~50MB
- **Per 1K documents**: ~1MB
- **During query**: Additional 10-50MB depending on context size

## Integration Questions

### Can I use Lexora with FastAPI?

Yes! See our [FastAPI Integration Guide](INTEGRATIONS.md#fastapi).

### Does Lexora work with Streamlit?

Absolutely! Check out our [Streamlit Integration Guide](INTEGRATIONS.md#streamlit).

### Can I integrate Lexora with Django?

Yes, see our [Django Integration Guide](INTEGRATIONS.md#django).

### Is there a REST API?

Lexora doesn't provide a built-in REST API, but you can easily create one using FastAPI or Flask. See our [Integration Examples](INTEGRATIONS.md).

### Can I use Lexora with LangChain?

Yes! We provide a LangChain retriever implementation. See [LangChain Integration](INTEGRATIONS.md#langchain).

## Vector Database Questions

### What vector databases does Lexora support?

Currently:
- **FAISS** (default, local)

Coming soon:
- **Pinecone** (cloud)
- **Chroma** (local/cloud)
- **Weaviate** (cloud)

### Can I use my own vector database?

Yes! You can implement a custom vector database provider by extending `BaseVectorDB`. See [Advanced Usage](ADVANCED_USAGE.md#custom-embeddings).

### Where are vectors stored?

By default, FAISS stores vectors locally in the `./faiss_storage` directory. You can configure this:

```python
agent = RAGAgent(
    vector_db_config=VectorDBConfig(
        provider="faiss",
        connection_params={"storage_path": "/custom/path"}
    )
)
```

### How do I backup my vector database?

For FAISS, simply backup the storage directory:

```bash
# Backup
cp -r ./faiss_storage ./faiss_storage_backup

# Restore
cp -r ./faiss_storage_backup ./faiss_storage
```

## LLM Questions

### What LLM providers are supported?

Currently:
- **Mock LLM** (for testing)
- **OpenAI** (GPT-3.5, GPT-4)

Coming soon:
- **Anthropic** (Claude)
- **Cohere**
- **Local models** (via Ollama)

### Can I use local LLMs?

Not yet, but we're working on support for local models via Ollama and Hugging Face. This is on our roadmap.

### How much does it cost to use OpenAI?

Costs depend on usage:
- **GPT-3.5-turbo**: ~$0.002 per 1K tokens
- **GPT-4**: ~$0.03 per 1K tokens
- **Embeddings**: ~$0.0001 per 1K tokens

A typical query might cost $0.001-0.01 depending on context size.

### Can I control the LLM temperature?

Yes:

```python
agent = RAGAgent(
    agent_config=AgentConfig(temperature=0.7)
)
```

## Error Handling

### What should I do if I get a "Corpus not found" error?

This means the corpus doesn't exist. Create it first:

```python
await agent.create_corpus("my_corpus")
```

### Why am I getting "Rate limit exceeded" errors?

You're hitting API rate limits. Solutions:
1. Implement rate limiting in your application
2. Add retry logic with exponential backoff
3. Upgrade your API plan
4. Use batching to reduce API calls

### How do I handle connection errors?

Implement retry logic:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def resilient_query(query: str):
    return await agent.query(query)
```

### What if the LLM returns low-confidence answers?

1. **Add more relevant documents** to your corpus
2. **Improve document quality** and chunking
3. **Use metadata filters** to narrow search
4. **Adjust query phrasing** for better matching
5. **Increase `top_k`** to retrieve more context

## Security Questions

### Is my data secure?

- **Local storage**: Data is stored locally by default (FAISS)
- **API keys**: Never hardcode API keys; use environment variables
- **Encryption**: Implement encryption for sensitive data
- **Access control**: Implement authentication in your application layer

### How do I secure my API endpoints?

Implement authentication and authorization:

```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/query")
async def query(request: QueryRequest, token=Depends(security)):
    # Verify token
    if not verify_token(token.credentials):
        raise HTTPException(status_code=401)
    # Process query
    return await agent.query(request.query)
```

### Can I use Lexora in production?

Yes! Lexora is designed for production use. Follow our [Best Practices Guide](BEST_PRACTICES.md) for:
- Error handling
- Performance optimization
- Security
- Monitoring
- Deployment

## Troubleshooting

### Lexora is running slowly. What can I do?

1. **Check document count**: Large corpora need optimization
2. **Enable caching**: Cache frequent queries
3. **Use batching**: Batch document additions
4. **Optimize chunk size**: Try 500-1000 characters
5. **Check system resources**: Ensure adequate RAM/CPU
6. **Profile your code**: Identify bottlenecks

### I'm getting memory errors. How do I fix this?

1. **Reduce batch size** when adding documents
2. **Use smaller embedding dimensions**
3. **Process documents in chunks**
4. **Increase system RAM**
5. **Use external vector DB** (Pinecone, Chroma)

### My queries return irrelevant results. Why?

1. **Improve document quality**: Clean and structure your data
2. **Better chunking**: Adjust chunk size and overlap
3. **Add metadata**: Use metadata for filtering
4. **Tune embeddings**: Try different embedding models
5. **Query rewriting**: Rephrase queries for better matching

### How do I debug issues?

Enable debug logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("lexora")
logger.setLevel(logging.DEBUG)
```

## Contributing

### How can I contribute to Lexora?

We welcome contributions! See our [Contributing Guide](../CONTRIBUTING.md) for:
- Code contributions
- Bug reports
- Feature requests
- Documentation improvements

### Where can I report bugs?

Report bugs on our [GitHub Issues](https://github.com/your-org/lexora-rag-sdk/issues) page.

### How do I request a feature?

Open a feature request on [GitHub Discussions](https://github.com/your-org/lexora-rag-sdk/discussions).

## Community & Support

### Where can I get help?

- **Documentation**: [Full documentation](../README.md)
- **GitHub Issues**: [Report bugs](https://github.com/your-org/lexora-rag-sdk/issues)
- **GitHub Discussions**: [Ask questions](https://github.com/your-org/lexora-rag-sdk/discussions)
- **Discord**: [Join our community](https://discord.gg/lexora)
- **Email**: support@lexora.com

### Is there a community forum?

Yes! Join our [GitHub Discussions](https://github.com/your-org/lexora-rag-sdk/discussions) or [Discord server](https://discord.gg/lexora).

### How do I stay updated?

- **Star** our [GitHub repository](https://github.com/your-org/lexora-rag-sdk)
- **Watch** for releases
- Follow us on Twitter: [@LexoraAI](https://twitter.com/LexoraAI)
- Subscribe to our [newsletter](https://lexora.com/newsletter)

## Roadmap

### What features are coming next?

- PyPI package distribution
- Additional vector DB support (Pinecone, Chroma)
- Local LLM support (Ollama, Hugging Face)
- Real-time document updates
- Advanced query analytics
- Multi-language support
- GraphQL API
- Kubernetes operators

### When will feature X be available?

Check our [Roadmap](../README.md#roadmap) for planned features and timelines.

---

**Didn't find your answer?** Ask on [GitHub Discussions](https://github.com/your-org/lexora-rag-sdk/discussions) or join our [Discord](https://discord.gg/lexora)!
