#!/usr/bin/env python3
"""
Corpus Management Example

This example demonstrates how to create and manage document corpora,
add documents, and perform RAG queries.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lexora import RAGAgent


async def main():
    """Corpus management example."""
    print("=" * 70)
    print("Corpus Management Example")
    print("=" * 70)
    
    # Initialize agent
    print("\n1. Initializing RAGAgent...")
    agent = RAGAgent()
    
    # Create a corpus
    print("\n2. Creating a corpus...")
    corpus_name = "tech_docs"
    
    create_tool = agent.tool_registry.get_tool("create_corpus")
    result = await create_tool.run(
        corpus_name=corpus_name,
        description="Technical documentation corpus"
    )
    print(f"   Corpus created: {result.data.get('corpus_name')}")
    
    # Add documents to the corpus
    print("\n3. Adding documents to corpus...")
    documents = [
        {
            "content": "Python is a high-level programming language known for its simplicity and readability.",
            "metadata": {"topic": "python", "type": "intro"}
        },
        {
            "content": "JavaScript is a versatile programming language primarily used for web development.",
            "metadata": {"topic": "javascript", "type": "intro"}
        },
        {
            "content": "Docker is a platform for developing, shipping, and running applications in containers.",
            "metadata": {"topic": "docker", "type": "intro"}
        },
        {
            "content": "Kubernetes is an open-source container orchestration platform for automating deployment.",
            "metadata": {"topic": "kubernetes", "type": "intro"}
        }
    ]
    
    add_tool = agent.tool_registry.get_tool("add_data")
    add_result = await add_tool.run(
        corpus_name=corpus_name,
        documents=documents
    )
    print(f"   Documents added: {add_result.data.get('documents_added')}")
    
    # Get corpus information
    print("\n4. Getting corpus information...")
    info_tool = agent.tool_registry.get_tool("get_corpus_info")
    info_result = await info_tool.run(corpus_name=corpus_name)
    
    print(f"   Corpus: {info_result.data.get('corpus_name')}")
    print(f"   Documents: {info_result.data.get('document_count')}")
    print(f"   Description: {info_result.data.get('description')}")
    
    # Perform RAG query
    print("\n5. Performing RAG query...")
    query_tool = agent.tool_registry.get_tool("rag_query")
    query_result = await query_tool.run(
        corpus_name=corpus_name,
        query="Tell me about container technologies",
        top_k=2
    )
    
    print(f"   Query: 'Tell me about container technologies'")
    print(f"   Results found: {len(query_result.data.get('results', []))}")
    
    if query_result.data.get('results'):
        print("\n   Top results:")
        for i, result in enumerate(query_result.data['results'][:2], 1):
            print(f"   {i}. Score: {result['score']:.4f}")
            print(f"      Content: {result['content'][:80]}...")
    
    # List all corpora
    print("\n6. Listing all corpora...")
    list_tool = agent.tool_registry.get_tool("list_corpora")
    list_result = await list_tool.run()
    
    print(f"   Total corpora: {len(list_result.data.get('corpora', []))}")
    for corpus in list_result.data.get('corpora', [])[:5]:
        print(f"   - {corpus['name']}: {corpus['document_count']} documents")
    
    # Cleanup
    print("\n7. Cleaning up...")
    delete_tool = agent.tool_registry.get_tool("delete_corpus")
    await delete_tool.run(
        corpus_name=corpus_name,
        confirm_deletion=True
    )
    print(f"   Corpus '{corpus_name}' deleted")
    
    print("\n" + "=" * 70)
    print("Corpus management example complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
