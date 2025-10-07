#!/usr/bin/env python3
"""
Custom Configuration Example

This example demonstrates how to configure the RAGAgent with custom
LLM and vector database settings.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lexora import RAGAgent, LLMConfig, VectorDBConfig, AgentConfig


async def main():
    """Custom configuration example."""
    print("=" * 70)
    print("Custom Configuration Example")
    print("=" * 70)
    
    # Configure LLM
    print("\n1. Configuring LLM...")
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY", "test-key"),
        temperature=0.7,
        max_tokens=2000
    )
    print(f"   Provider: {llm_config.provider}")
    print(f"   Model: {llm_config.model}")
    
    # Configure Vector Database
    print("\n2. Configuring Vector Database...")
    vector_db_config = VectorDBConfig(
        provider="faiss",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384,
        connection_params={
            "index_path": "./custom_faiss_index",
            "metric": "cosine"
        }
    )
    print(f"   Provider: {vector_db_config.provider}")
    print(f"   Embedding model: {vector_db_config.embedding_model}")
    print(f"   Dimension: {vector_db_config.dimension}")
    
    # Configure Agent
    print("\n3. Configuring Agent...")
    agent_config = AgentConfig(
        max_context_length=10000,
        max_tool_calls=15,
        log_level="INFO",
        enable_memory=True
    )
    print(f"   Max context: {agent_config.max_context_length}")
    print(f"   Max tool calls: {agent_config.max_tool_calls}")
    print(f"   Log level: {agent_config.log_level}")
    
    # Initialize agent with custom configuration
    print("\n4. Initializing RAGAgent with custom config...")
    agent = RAGAgent(
        llm_config=llm_config,
        vector_db_config=vector_db_config,
        agent_config=agent_config
    )
    print(f"   Agent initialized: {agent}")
    
    # Process a query
    print("\n5. Processing query...")
    response = await agent.query("Explain neural networks")
    print(f"   Answer: {response.answer[:100]}...")
    print(f"   Confidence: {response.confidence:.2f}")
    
    # Save configuration for reuse
    print("\n6. Saving configuration...")
    agent.save_config("examples/my_custom_config.yaml", format="yaml")
    print("   Configuration saved to examples/my_custom_config.yaml")
    
    print("\n" + "=" * 70)
    print("Custom configuration example complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
