# Getting Started with Lexora

This guide will help you get up and running with Lexora in minutes.

## Table of Contents

1. [Installation](#installation)
2. [Your First RAG Application](#your-first-rag-application)
3. [Understanding the Basics](#understanding-the-basics)
4. [Next Steps](#next-steps)

---

## Installation

### Step 1: Install Python

Lexora requires Python 3.8 or higher. Check your Python version:

```bash
python --version
```

If you need to install Python, visit [python.org](https://www.python.org/downloads/).

### Step 2: Create a Virtual Environment

It's recommended to use a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Lexora

```bash
# Clone the repository
git clone https://github.com/yourusername/lexora.git
cd lexora

# Install dependencies
pip install -r requirements.txt

# Install Lexora
pip install -e .
```

### Step 4: Verify Installation

```python
python -c "from lexora import RAGAgent; print('Lexora installed successfully!')"
```

---

## Your First RAG Application

Let's build a simple knowledge base application.

### Step 1: Create Your First Script

Create a file called `my_first_rag.py`:

```python
import asyncio
from lexora import RAGAgent

async def main():
    # Initialize the agent
    print("Initializing Lexora...")
    agent = RAGAgent()
    
    # Create a corpus
    print("Creating corpus...")
    await agent.tool_registry.get_tool("create_corpus").run(
        corpus_name="my_knowledge_base",
        description="My first knowledge base"
    )
    
    # Add some documents
    print("Adding documents...")
    documents = [
        {
            "content": "Python is a high-level programming language known for its simplicity.",
            "metadata": {"topic": "programming"}
        },
        {
            "content": "Machine learning is a branch of AI that learns from data.",
            "metadata": {"topic": "ai"}
        },
        {
            "content": "Neural networks are inspired by the human brain.",
            "metadata": {"topic": "ai"}
        }
    ]
    
    await agent.tool_registry.get_tool("add_data").run(
        corpus_name="my_knowledge_base",
        documents=documents
    )
    
    # Query the knowledge base
    print("\nQuerying knowledge base...")
    result = await agent.tool_registry.get_tool("rag_query").run(
        corpus_name="my_knowledge_base",
        query="What is machine learning?",
        top_k=2
    )
    
    # Display results
    print("\nResults:")
    for i, doc in enumerate(result.data["results"], 1):
        print(f"\n{i}. {doc['content']}")
        print(f"   Score: {doc['score']:.3f}")
        print(f"   Topic: {doc['metadata'].get('topic', 'N/A')}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 2: Run Your Application

```bash
python my_first_rag.py
```

You should see output like:

```
Initializing Lexora...
Creating corpus...
Adding documents...

Querying knowledge base...

Results:

1. Machine learning is a branch of AI that learns from data.
   Score: 0.892
   Topic: ai

2. Neural networks are inspired by the human brain.
   Score: 0.745
   Topic: ai
```

Congratulations! You've built your first RAG application! 🎉

---

## Understanding the Basics

### What Just Happened?

1. **Initialized the Agent**: Created a RAGAgent with default settings
2. **Created a Corpus**: Set up a container for your documents
3. **Added Documents**: Stored documents with automatic embedding generation
4. **Queried**: Performed semantic search to find relevant documents

### Key Concepts

#### 1. RAGAgent

The main interface to Lexora. It orchestrates all operations.

```python
agent = RAGAgent()  # Uses sensible defaults
```

#### 2. Corpus

A collection of documents that can be searched together.

```python
await agent.tool_registry.get_tool("create_corpus").run(
    corpus_name="my_corpus",
    description="Description of what this corpus contains"
)
```

#### 3. Documents

The basic unit of information. Each document has:
- **content**: The text content
- **metadata**: Optional key-value pairs for filtering

```python
document = {
    "content": "Your text here",
    "metadata": {"author": "John", "date": "2024-01-01"}
}
```

#### 4. Semantic Search

Unlike keyword search, semantic search understands meaning:

```python
# These queries would find similar results:
# - "What is ML?"
# - "Explain machine learning"
# - "Tell me about artificial intelligence learning"
```

---

## Next Steps

### 1. Learn About Configuration

See [CONFIGURATION.md](./CONFIGURATION.md) to learn how to:
- Use real LLMs (OpenAI, Claude, etc.)
- Configure vector databases (Pinecone, Chroma)
- Set up production environments

### 2. Explore RAG Tools

Check out [RAG_TOOLS.md](./RAG_TOOLS.md) for:
- Complete tool reference
- Advanced querying techniques
- Corpus management

### 3. Build Custom Tools

Read [CUSTOM_TOOLS.md](./CUSTOM_TOOLS.md) to:
- Create your own tools
- Extend Lexora's functionality
- Integrate with external APIs

### 4. Production Deployment

See [DEPLOYMENT.md](./DEPLOYMENT.md) for:
- Best practices
- Performance optimization
- Monitoring and logging

### 5. Try the Examples

Explore the [examples/](../examples/) directory:
- `01_quick_start.py` - Basic usage
- `02_custom_configuration.py` - Configuration options
- `03_corpus_management.py` - Managing corpora
- `04_custom_tools.py` - Creating custom tools

---

## Common Issues

### Issue: "Module not found"

**Solution**: Make sure you've installed Lexora:
```bash
pip install -e .
```

### Issue: "Async function not awaited"

**Solution**: Make sure you're using `asyncio.run()`:
```python
import asyncio

async def main():
    # Your async code here
    pass

if __name__ == "__main__":
    asyncio.run(main())
```

### Issue: "Corpus already exists"

**Solution**: Either delete the existing corpus or use `overwrite_existing=True`:
```python
await agent.tool_registry.get_tool("create_corpus").run(
    corpus_name="my_corpus",
    overwrite_existing=True
)
```

---

## Getting Help

- **Documentation**: Check the [docs/](.) directory
- **Examples**: See [examples/](../examples/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/lexora/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/lexora/discussions)

---

## What's Next?

Now that you've built your first RAG application, you're ready to:

1. ✅ Add more documents to your corpus
2. ✅ Experiment with different queries
3. ✅ Add metadata filtering
4. ✅ Configure a real LLM for better results
5. ✅ Build a production application

Happy building! 🚀
