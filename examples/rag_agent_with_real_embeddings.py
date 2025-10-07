#!/usr/bin/env python3
"""
Example: Using RAGAgent with Real Embedding Providers

This example demonstrates how to configure RAGAgent with real embedding
providers (OpenAI) instead of mock providers for production use.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lexora import RAGAgent, LLMConfig, VectorDBConfig, AgentConfig


async def example_with_openai_embeddings():
    """
    Example using OpenAI embeddings for production.
    
    This is the recommended approach for production deployments.
    """
    print("=" * 60)
    print("Example 1: RAGAgent with OpenAI Embeddings")
    print("=" * 60)
    
    # Configure with OpenAI embeddings
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY"),  # Get from environment
        temperature=0.7,
        max_tokens=2000
    )
    
    vector_db_config = VectorDBConfig(
        provider="faiss",
        embedding_model="text-embedding-ada-002",  # OpenAI embedding model
        dimension=1536,  # Ada-002 dimension
        connection_params={
            "index_path": "./faiss_index",
            "metric": "cosine",
            "openai_api_key": os.getenv("OPENAI_API_KEY")  # API key for embeddings
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
    
    print(f"\nAgent initialized: {agent}")
    print(f"Available tools: {len(agent.get_available_tools())}")
    
    # Process a query
    response = await agent.query("What is machine learning?")
    
    print(f"\nQuery: What is machine learning?")
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Execution time: {response.execution_time:.2f}s")


async def example_with_mock_for_testing():
    """
    Example using mock embeddings for testing/development.
    
    This is useful for local development without API keys.
    """
    print("\n" + "=" * 60)
    print("Example 2: RAGAgent with Mock Embeddings (Testing)")
    print("=" * 60)
    
    # Use default configuration (mock embeddings)
    agent = RAGAgent()
    
    print(f"\nAgent initialized: {agent}")
    print("Note: Using mock embeddings - suitable for testing only")
    
    # Process a query
    response = await agent.query("Explain neural networks")
    
    print(f"\nQuery: Explain neural networks")
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence:.2f}")


async def example_with_custom_embedding_provider():
    """
    Example showing how the system automatically selects the right provider.
    """
    print("\n" + "=" * 60)
    print("Example 3: Automatic Provider Selection")
    print("=" * 60)
    
    # Configuration with OpenAI model name
    vector_db_config = VectorDBConfig(
        provider="faiss",
        embedding_model="text-embedding-ada-002",  # OpenAI model
        dimension=1536,
        connection_params={
            "index_path": "./faiss_index",
            "openai_api_key": os.getenv("OPENAI_API_KEY", "your-api-key-here")
        }
    )
    
    agent = RAGAgent(vector_db_config=vector_db_config)
    
    print("\nEmbedding Provider Selection:")
    print("- Model name contains 'openai' or 'ada' → OpenAIEmbeddingProvider")
    print("- Model name contains 'mock' → MockEmbeddingProvider")
    print("- Unknown model → MockEmbeddingProvider (with warning)")
    
    print(f"\nConfigured model: {vector_db_config.embedding_model}")
    print("Provider: OpenAIEmbeddingProvider (automatically selected)")


def print_configuration_guide():
    """Print a guide for configuring embedding providers."""
    print("\n" + "=" * 60)
    print("Configuration Guide: Embedding Providers")
    print("=" * 60)
    
    print("""
1. OpenAI Embeddings (Production Recommended):
   
   vector_db_config = VectorDBConfig(
       provider="faiss",
       embedding_model="text-embedding-ada-002",
       dimension=1536,
       connection_params={
           "openai_api_key": "your-api-key"
       }
   )

2. Mock Embeddings (Testing/Development):
   
   vector_db_config = VectorDBConfig(
       provider="faiss",
       embedding_model="sentence-transformers/all-MiniLM-L6-v2",
       dimension=384,
       connection_params={"index_path": "./faiss_index"}
   )

3. Environment Variables (Recommended):
   
   # Set in your environment
   export OPENAI_API_KEY="your-api-key"
   
   # Use in code
   api_key=os.getenv("OPENAI_API_KEY")

Key Points:
- Mock embeddings are for testing only
- OpenAI embeddings provide better quality for production
- Always use environment variables for API keys
- The system automatically selects the right provider based on model name
""")


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("RAGAgent Embedding Provider Examples")
    print("=" * 60)
    
    # Check if OpenAI API key is available
    has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
    
    if has_openai_key:
        print("\n✓ OpenAI API key found - running production examples")
        await example_with_openai_embeddings()
    else:
        print("\n⚠ No OpenAI API key found - skipping production examples")
        print("Set OPENAI_API_KEY environment variable to run with real embeddings")
    
    # Always run mock example (doesn't need API key)
    await example_with_mock_for_testing()
    
    # Show automatic provider selection
    await example_with_custom_embedding_provider()
    
    # Print configuration guide
    print_configuration_guide()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
