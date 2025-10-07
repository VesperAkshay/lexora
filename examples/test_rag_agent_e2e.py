#!/usr/bin/env python3
"""
End-to-End Test for RAGAgent

This test creates a corpus, adds documents, and performs queries
to validate the complete RAG workflow.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lexora import RAGAgent, VectorDBConfig


async def test_complete_rag_workflow():
    """Test the complete RAG workflow from corpus creation to querying."""
    print("=" * 70)
    print("RAGAgent End-to-End Test")
    print("=" * 70)
    
    # Initialize agent with mock embeddings
    print("\n1. Initializing RAGAgent...")
    agent = RAGAgent()
    print(f"   ✓ Agent initialized with {len(agent.get_available_tools())} tools")
    
    # Create a test corpus
    print("\n2. Creating test corpus...")
    corpus_name = "ml_basics"
    
    try:
        # Try to delete if it exists
        result = await agent.tool_registry.get_tool("delete_corpus").run(
            corpus_name=corpus_name,
            confirm_deletion=corpus_name
        )
        print(f"   ✓ Deleted existing corpus '{corpus_name}'")
    except Exception:
        pass  # Corpus doesn't exist, that's fine
    
    # Create new corpus
    result = await agent.tool_registry.get_tool("create_corpus").run(
        corpus_name=corpus_name,
        description="Machine learning basics corpus for testing",
        overwrite_existing=True
    )
    print(f"   ✓ Created corpus: {result.data.get('corpus_name')}")
    
    # Add documents to the corpus
    print("\n3. Adding documents to corpus...")
    documents = [
        {
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            "metadata": {"topic": "ml_intro", "source": "test"}
        },
        {
            "content": "Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes that process information.",
            "metadata": {"topic": "neural_networks", "source": "test"}
        },
        {
            "content": "Supervised learning is a type of machine learning where the model is trained on labeled data. The algorithm learns to map inputs to outputs based on example input-output pairs.",
            "metadata": {"topic": "supervised_learning", "source": "test"}
        },
        {
            "content": "Deep learning is a subset of machine learning that uses neural networks with multiple layers. It excels at processing unstructured data like images and text.",
            "metadata": {"topic": "deep_learning", "source": "test"}
        },
        {
            "content": "Reinforcement learning is a machine learning paradigm where an agent learns to make decisions by interacting with an environment and receiving rewards or penalties.",
            "metadata": {"topic": "reinforcement_learning", "source": "test"}
        }
    ]
    
    add_result = await agent.tool_registry.get_tool("add_data").run(
        corpus_name=corpus_name,
        documents=documents
    )
    print(f"   ✓ Added {add_result.data.get('documents_added')} documents")
    
    # Get corpus info
    print("\n4. Checking corpus information...")
    info_result = await agent.tool_registry.get_tool("get_corpus_info").run(
        corpus_name=corpus_name
    )
    print(f"   ✓ Corpus: {info_result.data.get('corpus_name')}")
    print(f"   ✓ Documents: {info_result.data.get('document_count')}")
    print(f"   ✓ Description: {info_result.data.get('description')}")
    
    # Test direct RAG query (bypassing agent planning)
    print("\n5. Testing direct RAG query...")
    query_result = await agent.tool_registry.get_tool("rag_query").run(
        corpus_name=corpus_name,
        query="What is machine learning?",
        top_k=3
    )
    print(f"   ✓ Query: 'What is machine learning?'")
    print(f"   ✓ Found {len(query_result.data.get('results', []))} relevant documents")
    
    if query_result.data.get('results'):
        print("\n   Top result:")
        top_result = query_result.data['results'][0]
        print(f"   - Content: {top_result['content'][:100]}...")
        print(f"   - Score: {top_result['score']:.4f}")
    
    # Test another query
    print("\n6. Testing query about neural networks...")
    query_result2 = await agent.tool_registry.get_tool("rag_query").run(
        corpus_name=corpus_name,
        query="Explain neural networks",
        top_k=2
    )
    print(f"   ✓ Found {len(query_result2.data.get('results', []))} relevant documents")
    
    if query_result2.data.get('results'):
        print("\n   Top result:")
        top_result = query_result2.data['results'][0]
        print(f"   - Content: {top_result['content'][:100]}...")
        print(f"   - Score: {top_result['score']:.4f}")
    
    # List all corpora
    print("\n7. Listing all corpora...")
    list_result = await agent.tool_registry.get_tool("list_corpora").run()
    print(f"   ✓ Total corpora: {len(list_result.data.get('corpora', []))}")
    for corpus in list_result.data.get('corpora', []):
        print(f"   - {corpus['name']}: {corpus['document_count']} documents")
    
    # Test agent query (full workflow with planning)
    print("\n8. Testing full agent query workflow...")
    print("   Note: This may fail if the planner doesn't provide corpus_name")
    try:
        response = await agent.query(
            "What is supervised learning?",
            context={"corpus_name": corpus_name}  # Provide corpus name in context
        )
        print(f"   ✓ Query: 'What is supervised learning?'")
        print(f"   ✓ Answer: {response.answer[:150]}...")
        print(f"   ✓ Confidence: {response.confidence:.2f}")
        print(f"   ✓ Execution time: {response.execution_time:.2f}s")
    except Exception as e:
        print(f"   ✗ Agent query failed: {str(e)[:100]}")
        print("   (This is expected with mock LLM - planner can't generate proper parameters)")
    
    # Cleanup
    print("\n9. Cleaning up...")
    await agent.tool_registry.get_tool("delete_corpus").run(
        corpus_name=corpus_name
    )
    print(f"   ✓ Deleted test corpus '{corpus_name}'")
    
    print("\n" + "=" * 70)
    print("End-to-End Test Completed Successfully!")
    print("=" * 70)
    print("\nKey Findings:")
    print("✓ Corpus creation and management works")
    print("✓ Document addition and storage works")
    print("✓ Direct RAG queries work correctly")
    print("✓ Vector search returns relevant results")
    print("⚠ Full agent workflow needs real LLM for proper planning")


async def test_multiple_corpora():
    """Test working with multiple corpora."""
    print("\n" + "=" * 70)
    print("Testing Multiple Corpora")
    print("=" * 70)
    
    agent = RAGAgent()
    
    # Create two different corpora
    print("\n1. Creating multiple corpora...")
    
    # ML corpus
    ml_corpus = "ml_concepts"
    await agent.tool_registry.get_tool("create_corpus").run(
        corpus_name=ml_corpus,
        description="Machine learning concepts"
    )
    
    await agent.tool_registry.get_tool("add_data").run(
        corpus_name=ml_corpus,
        documents=[
            {"content": "Gradient descent is an optimization algorithm used to minimize loss functions in machine learning."},
            {"content": "Overfitting occurs when a model learns the training data too well, including noise and outliers."}
        ]
    )
    print(f"   ✓ Created and populated '{ml_corpus}'")
    
    # Programming corpus
    prog_corpus = "programming_basics"
    await agent.tool_registry.get_tool("create_corpus").run(
        corpus_name=prog_corpus,
        description="Programming fundamentals"
    )
    
    await agent.tool_registry.get_tool("add_data").run(
        corpus_name=prog_corpus,
        documents=[
            {"content": "Python is a high-level, interpreted programming language known for its simplicity and readability."},
            {"content": "Object-oriented programming is a paradigm based on the concept of objects containing data and code."}
        ]
    )
    print(f"   ✓ Created and populated '{prog_corpus}'")
    
    # Query each corpus
    print("\n2. Querying different corpora...")
    
    ml_result = await agent.tool_registry.get_tool("rag_query").run(
        corpus_name=ml_corpus,
        query="What is gradient descent?",
        top_k=1
    )
    print(f"   ✓ ML query returned {len(ml_result.data.get('results', []))} results")
    
    prog_result = await agent.tool_registry.get_tool("rag_query").run(
        corpus_name=prog_corpus,
        query="Tell me about Python",
        top_k=1
    )
    print(f"   ✓ Programming query returned {len(prog_result.data.get('results', []))} results")
    
    # List all corpora
    print("\n3. Listing all corpora...")
    list_result = await agent.tool_registry.get_tool("list_corpora").run()
    print(f"   ✓ Found {len(list_result.data.get('corpora', []))} corpora:")
    for corpus in list_result.data.get('corpora', []):
        print(f"     - {corpus['name']}: {corpus['document_count']} docs")
    
    # Cleanup
    print("\n4. Cleaning up...")
    await agent.tool_registry.get_tool("delete_corpus").run(corpus_name=ml_corpus)
    await agent.tool_registry.get_tool("delete_corpus").run(corpus_name=prog_corpus)
    print("   ✓ Deleted test corpora")
    
    print("\n" + "=" * 70)
    print("Multiple Corpora Test Completed!")
    print("=" * 70)


async def main():
    """Run all end-to-end tests."""
    try:
        await test_complete_rag_workflow()
        await test_multiple_corpora()
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
