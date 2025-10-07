#!/usr/bin/env python3
"""
Simplified End-to-End Integration Tests for Lexora Agentic RAG SDK

This module contains comprehensive integration tests that verify complete
workflows with real backends and multi-step reasoning scenarios.

Requirements tested: 2.1, 2.4, 6.1
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lexora import RAGAgent


async def test_complete_rag_pipeline():
    """
    Test the complete RAG pipeline:
    1. Create corpus
    2. Add documents
    3. Query with semantic search
    4. Verify results
    
    Requirements: 2.1, 2.4, 6.1
    """
    print("\n📋 Test 1: Complete RAG Pipeline")
    print("-" * 50)
    
    agent = RAGAgent()
    corpus_name = "e2e_test_corpus"
    
    try:
        # Step 1: Create corpus
        print("  Creating corpus...")
        create_result = await agent.tool_registry.get_tool("create_corpus").run(
            corpus_name=corpus_name,
            description="End-to-end test corpus",
            overwrite_existing=True
        )
        
        assert create_result.status == "success", f"Failed to create corpus: {create_result.error}"
        assert create_result.data["corpus_name"] == corpus_name
        print("  ✅ Corpus created successfully")
        
        # Step 2: Add documents
        print("  Adding documents...")
        documents = [
            {
                "content": "Python is a high-level programming language known for its simplicity and readability.",
                "metadata": {"topic": "python", "type": "intro"}
            },
            {
                "content": "Machine learning is a subset of AI that enables systems to learn from data.",
                "metadata": {"topic": "ml", "type": "definition"}
            },
            {
                "content": "Neural networks are inspired by biological neurons and consist of interconnected layers.",
                "metadata": {"topic": "ml", "type": "technical"}
            }
        ]
        
        add_result = await agent.tool_registry.get_tool("add_data").run(
            corpus_name=corpus_name,
            documents=documents
        )
        
        assert add_result.status == "success", f"Failed to add documents: {add_result.error}"
        assert add_result.data["documents_added"] == 3
        print(f"  ✅ Added {add_result.data['documents_added']} documents")
        
        # Step 3: Query the corpus
        print("  Querying corpus...")
        query_result = await agent.tool_registry.get_tool("rag_query").run(
            corpus_name=corpus_name,
            query="What is machine learning?",
            top_k=2
        )
        
        assert query_result.status == "success", f"Failed to query: {query_result.error}"
        assert len(query_result.data["results"]) > 0
        print(f"  ✅ Query returned {len(query_result.data['results'])} results")
        
        # Step 4: Verify results contain relevant information
        results_text = " ".join([r["content"] for r in query_result.data["results"]])
        assert "machine learning" in results_text.lower()
        print("  ✅ Results contain relevant information")
        
        print("\n✅ Complete RAG pipeline test PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n❌ Test FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Test ERROR: {e}")
        return False
    finally:
        # Cleanup
        try:
            await agent.tool_registry.get_tool("delete_corpus").run(
                corpus_name=corpus_name,
                confirm_deletion=corpus_name
            )
        except Exception:
            pass


async def test_multi_corpus_workflow():
    """
    Test workflow with multiple corpora.
    
    Requirements: 2.1, 6.1
    """
    print("\n📚 Test 2: Multi-Corpus Workflow")
    print("-" * 50)
    
    agent = RAGAgent()
    corpus1 = "e2e_corpus_1"
    corpus2 = "e2e_corpus_2"
    
    try:
        # Create first corpus
        print("  Creating first corpus...")
        await agent.tool_registry.get_tool("create_corpus").run(
            corpus_name=corpus1,
            description="First test corpus",
            overwrite_existing=True
        )
        print("  ✅ First corpus created")
        
        # Create second corpus
        print("  Creating second corpus...")
        await agent.tool_registry.get_tool("create_corpus").run(
            corpus_name=corpus2,
            description="Second test corpus",
            overwrite_existing=True
        )
        print("  ✅ Second corpus created")
        
        # Add data to first corpus
        print("  Adding data to first corpus...")
        await agent.tool_registry.get_tool("add_data").run(
            corpus_name=corpus1,
            documents=[{"content": "Python programming basics"}]
        )
        print("  ✅ Data added to first corpus")
        
        # Add data to second corpus
        print("  Adding data to second corpus...")
        await agent.tool_registry.get_tool("add_data").run(
            corpus_name=corpus2,
            documents=[{"content": "JavaScript web development"}]
        )
        print("  ✅ Data added to second corpus")
        
        # List all corpora
        print("  Listing all corpora...")
        list_result = await agent.tool_registry.get_tool("list_corpora").run()
        
        assert list_result.status == "success"
        corpus_names = [c["name"] for c in list_result.data["corpora"]]
        assert corpus1 in corpus_names
        assert corpus2 in corpus_names
        print(f"  ✅ Found {len(list_result.data['corpora'])} corpora")
        
        # Query specific corpus
        print("  Querying first corpus...")
        query_result = await agent.tool_registry.get_tool("rag_query").run(
            corpus_name=corpus1,
            query="programming"
        )
        
        assert query_result.status == "success"
        assert "Python" in query_result.data["results"][0]["content"]
        print("  ✅ Query returned correct corpus data")
        
        print("\n✅ Multi-corpus workflow test PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n❌ Test FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Test ERROR: {e}")
        return False
    finally:
        # Cleanup
        for corpus in [corpus1, corpus2]:
            try:
                await agent.tool_registry.get_tool("delete_corpus").run(
                    corpus_name=corpus,
                    confirm_deletion=corpus
                )
            except Exception:
                pass


async def test_error_handling():
    """
    Test error handling and recovery.
    
    Requirements: 6.1, 7.2
    """
    print("\n🛡️ Test 3: Error Handling")
    print("-" * 50)
    
    agent = RAGAgent()
    
    try:
        # Test querying nonexistent corpus
        print("  Testing query on nonexistent corpus...")
        result = await agent.tool_registry.get_tool("rag_query").run(
            corpus_name="nonexistent_corpus_12345",
            query="test query"
        )
        
        assert result.status == "error"
        assert "not found" in result.error.lower() or "does not exist" in result.error.lower()
        print("  ✅ Nonexistent corpus error handled correctly")
        
        # Test adding data to nonexistent corpus
        print("  Testing add data to nonexistent corpus...")
        result = await agent.tool_registry.get_tool("add_data").run(
            corpus_name="nonexistent_corpus_12345",
            documents=[{"content": "test"}]
        )
        
        assert result.status == "error"
        assert "not found" in result.error.lower() or "does not exist" in result.error.lower()
        print("  ✅ Add data error handled correctly")
        
        # Test duplicate corpus creation
        print("  Testing duplicate corpus creation...")
        corpus_name = "duplicate_test_corpus"
        
        # Create corpus first time
        result1 = await agent.tool_registry.get_tool("create_corpus").run(
            corpus_name=corpus_name,
            description="Test corpus",
            overwrite_existing=True
        )
        assert result1.status == "success"
        print("  ✅ First corpus creation successful")
        
        # Try to create again without overwrite
        result2 = await agent.tool_registry.get_tool("create_corpus").run(
            corpus_name=corpus_name,
            description="Test corpus"
        )
        
        assert result2.status == "error"
        assert "already exists" in result2.error.lower()
        print("  ✅ Duplicate corpus error handled correctly")
        
        # Cleanup
        await agent.tool_registry.get_tool("delete_corpus").run(
            corpus_name=corpus_name,
            confirm_deletion=corpus_name
        )
        
        print("\n✅ Error handling test PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n❌ Test FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Test ERROR: {e}")
        return False


async def test_faiss_persistence():
    """
    Test that data persists in FAISS backend.
    
    Requirements: 4.3, 6.1
    """
    print("\n💾 Test 4: FAISS Persistence")
    print("-" * 50)
    
    agent = RAGAgent()
    corpus_name = "faiss_integration_test"
    
    try:
        # Create corpus and add data
        print("  Creating corpus...")
        await agent.tool_registry.get_tool("create_corpus").run(
            corpus_name=corpus_name,
            description="FAISS integration test",
            overwrite_existing=True
        )
        print("  ✅ Corpus created")
        
        print("  Adding documents...")
        await agent.tool_registry.get_tool("add_data").run(
            corpus_name=corpus_name,
            documents=[
                {"content": "Test document for FAISS persistence"},
                {"content": "Another test document"}
            ]
        )
        print("  ✅ Documents added")
        
        # Get corpus info to verify data was added
        print("  Verifying corpus info...")
        info_result = await agent.tool_registry.get_tool("get_corpus_info").run(
            corpus_name=corpus_name
        )
        
        assert info_result.status == "success"
        assert info_result.data["document_count"] == 2
        print(f"  ✅ Corpus contains {info_result.data['document_count']} documents")
        
        # Query to verify data is retrievable
        print("  Querying corpus...")
        query_result = await agent.tool_registry.get_tool("rag_query").run(
            corpus_name=corpus_name,
            query="test document"
        )
        
        assert query_result.status == "success"
        assert len(query_result.data["results"]) > 0
        print(f"  ✅ Query returned {len(query_result.data['results'])} results")
        
        print("\n✅ FAISS persistence test PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n❌ Test FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Test ERROR: {e}")
        return False
    finally:
        # Cleanup
        try:
            await agent.tool_registry.get_tool("delete_corpus").run(
                corpus_name=corpus_name,
                confirm_deletion=corpus_name
            )
        except Exception:
            pass


async def test_large_document_batch():
    """
    Test adding a large batch of documents.
    
    Requirements: 5.1, 6.4
    """
    print("\n📦 Test 5: Large Document Batch")
    print("-" * 50)
    
    agent = RAGAgent()
    corpus_name = "large_batch_test"
    
    try:
        print("  Creating corpus...")
        await agent.tool_registry.get_tool("create_corpus").run(
            corpus_name=corpus_name,
            description="Large batch test",
            overwrite_existing=True
        )
        print("  ✅ Corpus created")
        
        # Create 50 documents
        print("  Creating 50 documents...")
        documents = [
            {"content": f"Document number {i} with test content about topic {i % 5}"}
            for i in range(50)
        ]
        
        print("  Adding documents...")
        result = await agent.tool_registry.get_tool("add_data").run(
            corpus_name=corpus_name,
            documents=documents
        )
        
        assert result.status == "success"
        assert result.data["documents_added"] == 50
        print(f"  ✅ Added {result.data['documents_added']} documents")
        
        # Verify all documents are queryable
        print("  Verifying corpus info...")
        info_result = await agent.tool_registry.get_tool("get_corpus_info").run(
            corpus_name=corpus_name
        )
        
        assert info_result.data["document_count"] == 50
        print(f"  ✅ Corpus contains {info_result.data['document_count']} documents")
        
        print("\n✅ Large document batch test PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n❌ Test FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Test ERROR: {e}")
        return False
    finally:
        # Cleanup
        try:
            await agent.tool_registry.get_tool("delete_corpus").run(
                corpus_name=corpus_name,
                confirm_deletion=corpus_name
            )
        except Exception:
            pass


async def run_all_tests():
    """Run all integration tests."""
    print("=" * 70)
    print("End-to-End Integration Tests - Lexora Agentic RAG SDK")
    print("=" * 70)
    
    results = []
    
    # Run all tests
    results.append(await test_complete_rag_pipeline())
    results.append(await test_multi_corpus_workflow())
    results.append(await test_error_handling())
    results.append(await test_faiss_persistence())
    results.append(await test_large_document_batch())
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n❌ {total - passed} test(s) failed")
    
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
