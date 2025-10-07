#!/usr/bin/env python3
"""
Quick Start Example

This example demonstrates the simplest way to get started with the Lexora
Agentic RAG SDK. It shows basic initialization and query processing.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lexora import RAGAgent


async def main():
    """Quick start example."""
    print("=" * 70)
    print("Lexora RAG Agent - Quick Start")
    print("=" * 70)
    
    # Step 1: Initialize the agent with default configuration
    print("\n1. Initializing RAGAgent with defaults...")
    agent = RAGAgent()
    print(f"   Agent initialized: {agent}")
    print(f"   Available tools: {len(agent.get_available_tools())}")
    
    # Step 2: Process a simple query
    print("\n2. Processing a query...")
    query = "What is machine learning?"
    print(f"   Query: {query}")
    
    response = await agent.query(query)
    
    print(f"\n3. Response:")
    print(f"   Answer: {response.answer}")
    print(f"   Confidence: {response.confidence:.2f}")
    print(f"   Execution time: {response.execution_time:.2f}s")
    print(f"   Sources: {len(response.sources)}")
    
    # Step 3: Check available tools
    print("\n4. Available tools:")
    for tool in agent.get_available_tools():
        print(f"   - {tool}")
    
    print("\n" + "=" * 70)
    print("Quick start complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
