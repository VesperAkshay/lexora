# Lexora Agentic RAG SDK - Examples

This directory contains comprehensive examples demonstrating various features of the Lexora Agentic RAG SDK.

## Quick Start

### Prerequisites

```bash
# Install the SDK
pip install -e .

# Set environment variables (optional)
export OPENAI_API_KEY="your-api-key"
```

## Examples Overview

### 1. Quick Start (`01_quick_start.py`)

The simplest way to get started with the SDK.

**What you'll learn:**
- Basic agent initialization
- Processing simple queries
- Checking available tools

**Run:**
```bash
python examples/01_quick_start.py
```

**Key Code:**
```python
from lexora import RAGAgent

agent = RAGAgent()
response = await agent.query("What is machine learning?")
print(response.answer)
```

---

### 2. Custom Configuration (`02_custom_configuration.py`)

Configure the agent with custom LLM and vector database settings.

**What you'll learn:**
- Configuring LLM providers
- Setting up vector databases
- Customizing agent behavior
- Saving and loading configurations

**Run:**
```bash
python examples/02_custom_configuration.py
```

**Key Code:**
```python
from lexora import RAGAgent, LLMConfig, VectorDBConfig

llm_config = LLMConfig(
    provider="openai",
    model="gpt-3.5-turbo",
    api_key="your-key"
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

---

### 3. Corpus Management (`03_corpus_management.py`)

Create and manage document corpora for RAG queries.

**What you'll learn:**
- Creating corpora
- Adding documents
- Performing RAG queries
- Managing corpus lifecycle

**Run:**
```bash
python examples/03_corpus_management.py
```

**Key Code:**
```python
# Create corpus
create_tool = agent.tool_registry.get_tool("create_corpus")
await create_tool.run(
    corpus_name="my_docs",
    description="My document collection"
)

# Add documents
add_tool = agent.tool_registry.get_tool("add_data")
await add_tool.run(
    corpus_name="my_docs",
    documents=[
        {"content": "Document 1 content", "metadata": {"type": "doc"}},
        {"content": "Document 2 content", "metadata": {"type": "doc"}}
    ]
)

# Query
query_tool = agent.tool_registry.get_tool("rag_query")
results = await query_tool.run(
    corpus_name="my_docs",
    query="Find relevant information",
    top_k=5
)
```

---

### 4. Custom Tools (`04_custom_tools.py`)

Create and register custom tools with the agent.

**What you'll learn:**
- Creating custom tools
- Defining tool parameters
- Registering tools with the agent
- Using custom tools

**Run:**
```bash
python examples/04_custom_tools.py
```

**Key Code:**
```python
from lexora import BaseTool, ToolParameter

class MyCustomTool(BaseTool):
    name = "my_tool"
    description = "My custom tool"
    
    def __init__(self):
        super().__init__()
        self.parameters = [
            ToolParameter(
                name="input",
                type="string",
                description="Input parameter",
                required=True
            )
        ]
    
    async def _execute(self, input: str, **kwargs):
        return {"result": f"Processed: {input}"}

# Register with agent
agent.add_tool(MyCustomTool())
```

---

### 5. Configuration Management (`configuration_demo.py`)

Comprehensive demonstration of all configuration methods.

**What you'll learn:**
- Loading from YAML files
- Loading from JSON files
- Using environment variables
- Programmatic configuration
- Saving configurations

**Run:**
```bash
python examples/configuration_demo.py
```

---

### 6. RAG Tools Demo (`rag_tools_demo.py`)

Demonstration of all built-in RAG tools.

**What you'll learn:**
- Using all RAG tools
- Tool chaining
- Error handling
- Best practices

**Run:**
```bash
python examples/rag_tools_demo.py
```

---

### 7. Real Embeddings (`rag_agent_with_real_embeddings.py`)

Using real embedding providers (OpenAI) for production.

**What you'll learn:**
- Configuring OpenAI embeddings
- Production deployment
- API key management
- Performance considerations

**Run:**
```bash
export OPENAI_API_KEY="your-key"
python examples/rag_agent_with_real_embeddings.py
```

---

## Configuration Files

### YAML Configuration (`config_example.yaml`)

Example YAML configuration file:

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

**Usage:**
```python
agent = RAGAgent.from_yaml("config_example.yaml")
```

### JSON Configuration (`config_example.json`)

Example JSON configuration file (same structure as YAML).

**Usage:**
```python
agent = RAGAgent.from_json("config_example.json")
```

### Environment Variables (`.env.example`)

Example environment variables file:

```bash
LEXORA_LLM_PROVIDER=openai
LEXORA_LLM_MODEL=gpt-3.5-turbo
LEXORA_LLM_API_KEY=your-key
LEXORA_VECTORDB_PROVIDER=faiss
LEXORA_VECTORDB_CONNECTION_PARAMS={"index_path": "./faiss_index"}
```

**Usage:**
```python
agent = RAGAgent.from_env()
```

---

## Common Patterns

### Pattern 1: Simple Query

```python
from lexora import RAGAgent

agent = RAGAgent()
response = await agent.query("Your question here")
print(response.answer)
```

### Pattern 2: With Context

```python
response = await agent.query(
    "Your question",
    context={"corpus_name": "my_docs"}
)
```

### Pattern 3: Custom Configuration

```python
from lexora import RAGAgent, LLMConfig

agent = RAGAgent(
    llm_config=LLMConfig(
        model="gpt-4",
        temperature=0.5
    )
)
```

### Pattern 4: From Configuration File

```python
agent = RAGAgent.from_yaml("config.yaml")
```

### Pattern 5: Tool Usage

```python
tool = agent.tool_registry.get_tool("rag_query")
result = await tool.run(
    corpus_name="docs",
    query="search query",
    top_k=5
)
```

---

## Troubleshooting

### Issue: Import Error

```python
ModuleNotFoundError: No module named 'lexora'
```

**Solution:**
```bash
# Install in development mode
pip install -e .
```

### Issue: API Key Error

```python
OpenAIError: The api_key client option must be set
```

**Solution:**
```bash
export OPENAI_API_KEY="your-api-key"
```

Or use mock embeddings for testing:
```python
vector_db_config = VectorDBConfig(
    provider="faiss",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    dimension=384,
    connection_params={"index_path": "./faiss_index"}
)
```

### Issue: Configuration Error

```python
ValueError: Provider must be one of {'faiss', 'pinecone', 'chroma'}
```

**Solution:** Check your configuration matches supported providers.

---

## Next Steps

1. **Read the Documentation:**
   - [API Reference](../docs/API_REFERENCE.md)
   - [Configuration Guide](../CONFIGURATION_GUIDE.md)

2. **Try the Examples:**
   - Start with `01_quick_start.py`
   - Progress through numbered examples
   - Experiment with custom configurations

3. **Build Your Application:**
   - Use examples as templates
   - Customize for your use case
   - Refer to API documentation

4. **Get Help:**
   - Check the documentation
   - Review example code
   - Open an issue on GitHub

---

## Example Output

### Quick Start Output

```
======================================================================
Lexora RAG Agent - Quick Start
======================================================================

1. Initializing RAGAgent with defaults...
   Agent initialized: RAGAgent(llm=mock, vector_db=faiss, tools=6)
   Available tools: 6

2. Processing a query...
   Query: What is machine learning?

3. Response:
   Answer: No results available to synthesize.
   Confidence: 0.00
   Execution time: 0.23s
   Sources: 0

4. Available tools:
   - create_corpus
   - add_data
   - rag_query
   - list_corpora
   - delete_corpus
   - get_corpus_info

======================================================================
Quick start complete!
======================================================================
```

---

## Contributing Examples

Have a useful example? Contributions are welcome!

1. Create a new example file
2. Follow the naming convention: `##_description.py`
3. Include comprehensive comments
4. Add to this README
5. Submit a pull request

---

## License

All examples are provided under the same license as the Lexora SDK (MIT).
