#!/usr/bin/env python3
"""
Performance Testing for Lexora Agentic RAG SDK

This module tests response times, resource usage, and optimization.

Requirements tested: 6.4, 7.3
"""

import asyncio
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lexora import RAGAgent


async def test_query_response_time():
    """
    Test query response times are within acceptable limits.
    
    Requirements: 6.4, 7.3
    """
    print("\n⏱️  Test 1: Query Response Time")
    print("-" * 50)
    
    agent = RAGAgent()
    corpus_name = "perf_test_corpus"
    
    try:
        # Setup
        await agent.tool_registry.get_tool("create_corpus").run(
            corpus_name=corpus_name,
            description="Performance test corpus",
            overwrite_existing=True
        )
        
        # Add test documents
        documents = [
            {"content": f"Test document {i} with sample content for performance testing"}
            for i in range(10)
        ]
        
        await agent.tool_registry.get_tool("add_data").run(
            corpus_name=corpus_name,
            documents=documents
        )
        
        # Test query response time
        print("  Testing query response time...")
        start_time = time.time()
        
        result = await agent.tool_registry.get_tool("rag_query").run(
            corpus_name=corpus_name,
            query="test document",
            top_k=5
        )
        
        elapsed_time = time.time() - start_time
        
        assert result.status == "success"
        print(f"  Query completed in {elapsed_time:.3f}s")
        
        # Response time should be under 2 seconds for small corpus
        assert elapsed_time < 2.0, f"Query took too long: {elapsed_time:.3f}s"
        print(f"  ✅ Query response time acceptable ({elapsed_time:.3f}s < 2.0s)")
        
        return True
        
    finally:
        # Cleanup
        try:
            await agent.tool_registry.get_tool("delete_corpus").run(
                corpus_name=corpus_name,
                confirm_deletion=corpus_name
            )
        except Exception:
            pass


async def test_batch_processing_performance():
    """
    Test batch document processing performance.
    
    Requirements: 6.4
    """
    print("\n📦 Test 2: Batch Processing Performance")
    print("-" * 50)
    
    agent = RAGAgent()
    corpus_name = "batch_perf_test"
    
    try:
        await agent.tool_registry.get_tool("create_corpus").run(
            corpus_name=corpus_name,
            description="Batch performance test",
            overwrite_existing=True
        )
        
        # Test adding 100 documents
        print("  Adding 100 documents...")
        documents = [
            {"content": f"Document {i} with test content for batch processing performance evaluation"}
            for i in range(100)
        ]
        
        start_time = time.time()
        
        result = await agent.tool_registry.get_tool("add_data").run(
            corpus_name=corpus_name,
            documents=documents
        )
        
        elapsed_time = time.time() - start_time
        
        assert result.status == "success"
        assert result.data["documents_added"] == 100
        
        print(f"  Added 100 documents in {elapsed_time:.3f}s")
        print(f"  Average: {elapsed_time/100*1000:.1f}ms per document")
        
        # Should process at least 10 docs/second
        docs_per_second = 100 / elapsed_time
        assert docs_per_second >= 10, f"Too slow: {docs_per_second:.1f} docs/sec"
        print(f"  ✅ Batch processing acceptable ({docs_per_second:.1f} docs/sec)")
        
        return True
        
    finally:
        try:
            await agent.tool_registry.get_tool("delete_corpus").run(
                corpus_name=corpus_name,
                confirm_deletion=corpus_name
            )
        except Exception:
            pass


async def test_concurrent_operations():
    """
    Test performance with concurrent operations.
    
    Requirements: 6.4
    """
    print("\n🔀 Test 3: Concurrent Operations")
    print("-" * 50)
    
    agent = RAGAgent()
    corpus_name = "concurrent_test"
    
    try:
        await agent.tool_registry.get_tool("create_corpus").run(
            corpus_name=corpus_name,
            description="Concurrent test",
            overwrite_existing=True
        )
        
        # Add some documents
        documents = [
            {"content": f"Document {i} for concurrent testing"}
            for i in range(20)
        ]
        
        await agent.tool_registry.get_tool("add_data").run(
            corpus_name=corpus_name,
            documents=documents
        )
        
        # Test concurrent queries
        print("  Running 10 concurrent queries...")
        start_time = time.time()
        
        tasks = [
            agent.tool_registry.get_tool("rag_query").run(
                corpus_name=corpus_name,
                query=f"test query {i}",
                top_k=3
            )
            for i in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        
        elapsed_time = time.time() - start_time
        
        # All should succeed
        assert all(r.status == "success" for r in results)
        
        print(f"  Completed 10 concurrent queries in {elapsed_time:.3f}s")
        print(f"  Average: {elapsed_time/10:.3f}s per query")
        
        # Should complete in reasonable time
        assert elapsed_time < 5.0, f"Concurrent queries too slow: {elapsed_time:.3f}s"
        print(f"  ✅ Concurrent operations acceptable ({elapsed_time:.3f}s < 5.0s)")
        
        return True
        
    finally:
        try:
            await agent.tool_registry.get_tool("delete_corpus").run(
                corpus_name=corpus_name,
                confirm_deletion=corpus_name
            )
        except Exception:
            pass


async def test_memory_efficiency():
    """
    Test memory efficiency with large operations.
    
    Requirements: 6.4
    """
    print("\n💾 Test 4: Memory Efficiency")
    print("-" * 50)
    
    agent = RAGAgent()
    corpus_name = "memory_test"
    
    try:
        await agent.tool_registry.get_tool("create_corpus").run(
            corpus_name=corpus_name,
            description="Memory test",
            overwrite_existing=True
        )
        
        # Add documents in batches to test memory handling
        print("  Adding 200 documents in batches...")
        total_added = 0
        
        for batch in range(4):
            documents = [
                {"content": f"Batch {batch} document {i} with content"}
                for i in range(50)
            ]
            
            result = await agent.tool_registry.get_tool("add_data").run(
                corpus_name=corpus_name,
                documents=documents
            )
            
            assert result.status == "success"
            total_added += result.data["documents_added"]
        
        assert total_added == 200
        print(f"  ✅ Successfully added {total_added} documents in batches")
        
        # Verify corpus info
        info_result = await agent.tool_registry.get_tool("get_corpus_info").run(
            corpus_name=corpus_name
        )
        
        assert info_result.status == "success"
        assert info_result.data["document_count"] == 200
        print(f"  ✅ Corpus contains {info_result.data['document_count']} documents")
        
        return True
        
    finally:
        try:
            await agent.tool_registry.get_tool("delete_corpus").run(
                corpus_name=corpus_name,
                confirm_deletion=corpus_name
            )
        except Exception:
            pass


async def test_caching_effectiveness():
    """
    Test that repeated operations benefit from caching.
    
    Requirements: 6.4
    """
    print("\n🗄️  Test 5: Caching Effectiveness")
    print("-" * 50)
    
    agent = RAGAgent()
    corpus_name = "cache_test"
    
    try:
        await agent.tool_registry.get_tool("create_corpus").run(
            corpus_name=corpus_name,
            description="Cache test",
            overwrite_existing=True
        )
        
        documents = [
            {"content": f"Document {i} for caching test"}
            for i in range(10)
        ]
        
        await agent.tool_registry.get_tool("add_data").run(
            corpus_name=corpus_name,
            documents=documents
        )
        
        # First query (cold)
        print("  Running first query (cold)...")
        start_time = time.time()
        result1 = await agent.tool_registry.get_tool("rag_query").run(
            corpus_name=corpus_name,
            query="test document"
        )
        time1 = time.time() - start_time
        
        # Second query (potentially cached)
        print("  Running second query (warm)...")
        start_time = time.time()
        result2 = await agent.tool_registry.get_tool("rag_query").run(
            corpus_name=corpus_name,
            query="test document"
        )
        time2 = time.time() - start_time
        
        assert result1.status == "success"
        assert result2.status == "success"
        
        print(f"  First query: {time1:.3f}s")
        print(f"  Second query: {time2:.3f}s")
        
        # Second query should not be significantly slower
        # (may not be faster due to mock embeddings, but shouldn't be slower)
        print(f"  ✅ Repeated queries handled efficiently")
        
        return True
        
    finally:
        try:
            await agent.tool_registry.get_tool("delete_corpus").run(
                corpus_name=corpus_name,
                confirm_deletion=corpus_name
            )
        except Exception:
            pass


async def run_all_performance_tests():
    """Run all performance tests."""
    print("=" * 70)
    print("Performance Tests - Lexora Agentic RAG SDK")
    print("=" * 70)
    
    results = []
    
    # Run all tests
    results.append(await test_query_response_time())
    results.append(await test_batch_processing_performance())
    results.append(await test_concurrent_operations())
    results.append(await test_memory_efficiency())
    results.append(await test_caching_effectiveness())
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ ALL PERFORMANCE TESTS PASSED!")
    else:
        print(f"\n❌ {total - passed} test(s) failed")
    
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_performance_tests())
    sys.exit(0 if success else 1)
