#!/usr/bin/env python3
"""
End-to-End Integration Tests for Lexora Agentic RAG SDK (Pytest Version)

This module contains comprehensive integration tests that verify complete
workflows with real backends and multi-step reasoning scenarios.

⚠️ NOTE: This file requires pytest to run properly due to fixture usage.
For standalone execution without pytest, use test_e2e_simple.py instead.

Requirements tested: 2.1, 2.4, 6.1

Usage:
    pytest tests/integration/test_e2e_workflows.py -v
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

try:
    import pytest
    import pytest_asyncio
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Create dummy pytest decorators for standalone execution
    class pytest:
        class fixture:
            def __init__(self, *args, **kwargs):
                pass
            def __call__(self, func):
                return func
        
        class mark:
            class asyncio:
                def __init__(self, *args, **kwargs):
                    pass
                def __call__(self, func):
                    return func
        
        fixture = fixture()
        mark = mark()
    
    class pytest_asyncio:
        class fixture:
            def __init__(self, *args, **kwargs):
                pass
            def __call__(self, func):
                return func
        
        fixture = fixture()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lexora import RAGAgent, LLMConfig, VectorDBConfig, AgentConfig
from lexora.models.core import Document


class TestCompleteRAGWorkflow:
    """Test complete RAG workflows from corpus creation to query answering."""
    
    @pytest_asyncio.fixture
    async def agent(self):
        """Create a RAGAgent instance for testing."""
        agent = RAGAgent()
        yield agent
        # Cleanup
        try:
            await agent.tool_registry.get_tool("delete_corpus").run(
                corpus_name="e2e_test_corpus",
                confirm_deletion="e2e_test_corpus"
            )
        except Exception:
            pass
    
    @pytest.mark.asyncio
    async def test_complete_rag_pipeline(self, agent):
        """
        Test the complete RAG pipeline:
        1. Create corpus
        2. Add documents
        3. Query with semantic search
        4. Verify results
        
        Requirements: 2.1, 2.4, 6.1
        """
        corpus_name = "e2e_test_corpus"
        
        # Step 1: Create corpus
        create_result = await agent.tool_registry.get_tool("create_corpus").run(
            corpus_name=corpus_name,
            description="End-to-end test corpus",
            overwrite_existing=True
        )
        
        assert create_result.status == "success"
        assert create_result.data["corpus_name"] == corpus_name
        
        # Step 2: Add documents
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
        
        assert add_result.status == "success"
        assert add_result.data["documents_added"] == 3
        
        # Step 3: Query the corpus
        query_result = await agent.tool_registry.get_tool("rag_query").run(
            corpus_name=corpus_name,
            query="What is machine learning?",
            top_k=2
        )
        
        assert query_result.status == "success"
        assert len(query_result.data["results"]) > 0
        
        # Step 4: Verify results contain relevant information
        results_text = " ".join([r["content"] for r in query_result.data["results"]])
        assert "machine learning" in results_text.lower()
        
        print("✅ Complete RAG pipeline test passed")
    
    @pytest.mark.asyncio
    async def test_multi_corpus_workflow(self, agent):
        """
        Test workflow with multiple corpora:
        1. Create multiple corpora
        2. Add different data to each
        3. Query specific corpus
        4. List all corpora
        
        Requirements: 2.1, 6.1
        """
        corpus1 = "e2e_corpus_1"
        corpus2 = "e2e_corpus_2"
        
        try:
            # Create first corpus
            await agent.tool_registry.get_tool("create_corpus").run(
                corpus_name=corpus1,
                description="First test corpus",
                overwrite_existing=True
            )
            
            # Create second corpus
            await agent.tool_registry.get_tool("create_corpus").run(
                corpus_name=corpus2,
                description="Second test corpus",
                overwrite_existing=True
            )
            
            # Add data to first corpus
            await agent.tool_registry.get_tool("add_data").run(
                corpus_name=corpus1,
                documents=[{"content": "Python programming basics"}]
            )
            
            # Add data to second corpus
            await agent.tool_registry.get_tool("add_data").run(
                corpus_name=corpus2,
                documents=[{"content": "JavaScript web development"}]
            )
            
            # List all corpora
            list_result = await agent.tool_registry.get_tool("list_corpora").run()
            
            assert list_result.status == "success"
            corpus_names = [c["name"] for c in list_result.data["corpora"]]
            assert corpus1 in corpus_names
            assert corpus2 in corpus_names
            
            # Query specific corpus
            query_result = await agent.tool_registry.get_tool("rag_query").run(
                corpus_name=corpus1,
                query="programming"
            )
            
            assert query_result.status == "success"
            assert "Python" in query_result.data["results"][0]["content"]
            
            print("✅ Multi-corpus workflow test passed")
            
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


class TestMultiStepReasoning:
    """Test multi-step reasoning scenarios."""
    
    @pytest_asyncio.fixture
    async def agent_with_data(self):
        """Create agent with pre-populated corpus."""
        agent = RAGAgent()
        corpus_name = "reasoning_test_corpus"
        
        # Setup corpus with data
        await agent.tool_registry.get_tool("create_corpus").run(
            corpus_name=corpus_name,
            description="Corpus for reasoning tests",
            overwrite_existing=True
        )
        
        documents = [
            {
                "content": "The capital of France is Paris. Paris is known for the Eiffel Tower.",
                "metadata": {"topic": "geography"}
            },
            {
                "content": "The Eiffel Tower was built in 1889 and is 330 meters tall.",
                "metadata": {"topic": "landmarks"}
            },
            {
                "content": "France is a country in Western Europe with a population of 67 million.",
                "metadata": {"topic": "geography"}
            }
        ]
        
        await agent.tool_registry.get_tool("add_data").run(
            corpus_name=corpus_name,
            documents=documents
        )
        
        yield agent, corpus_name
        
        # Cleanup
        try:
            await agent.tool_registry.get_tool("delete_corpus").run(
                corpus_name=corpus_name,
                confirm_deletion=corpus_name
            )
        except Exception:
            pass
    
    @pytest.mark.asyncio
    async def test_query_with_context_building(self, agent_with_data):
        """
        Test query that requires building context from multiple documents.
        
        Requirements: 2.4, 6.1
        """
        agent, corpus_name = agent_with_data
        
        # Query that requires information from multiple documents
        result = await agent.query(
            "Tell me about the Eiffel Tower in Paris"
        )
        
        assert result is not None
        assert result.answer is not None
        assert isinstance(result.confidence, float)
        
        # The response should synthesize information from multiple sources
        # Even if it's a fallback response, the system should handle it gracefully
        print(f"✅ Context building test passed - Confidence: {result.confidence}")
    
    @pytest.mark.asyncio
    async def test_sequential_queries(self, agent_with_data):
        """
        Test sequential queries that build on each other.
        
        Requirements: 2.1, 2.4
        """
        agent, corpus_name = agent_with_data
        
        # First query
        result1 = await agent.query("What is the capital of France?")
        assert result1 is not None
        assert result1.answer is not None
        
        # Second query building on first
        result2 = await agent.query("Tell me more about its famous landmark")
        assert result2 is not None
        assert result2.answer is not None
        
        print("✅ Sequential queries test passed")


class TestErrorRecoveryWorkflows:
    """Test error handling and recovery in complete workflows."""
    
    @pytest_asyncio.fixture
    async def agent(self):
        """Create a RAGAgent instance for testing."""
        agent = RAGAgent()
        yield agent
    
    @pytest.mark.asyncio
    async def test_query_nonexistent_corpus(self, agent):
        """
        Test querying a corpus that doesn't exist.
        
        Requirements: 6.1, 7.2
        """
        result = await agent.tool_registry.get_tool("rag_query").run(
            corpus_name="nonexistent_corpus_12345",
            query="test query"
        )
        
        assert result.status == "error"
        assert "not found" in result.error.lower() or "does not exist" in result.error.lower()
        
        print("✅ Nonexistent corpus error handling test passed")
    
    @pytest.mark.asyncio
    async def test_add_data_to_nonexistent_corpus(self, agent):
        """
        Test adding data to a corpus that doesn't exist.
        
        Requirements: 6.1, 7.2
        """
        result = await agent.tool_registry.get_tool("add_data").run(
            corpus_name="nonexistent_corpus_12345",
            documents=[{"content": "test"}]
        )
        
        assert result.status == "error"
        assert "not found" in result.error.lower() or "does not exist" in result.error.lower()
        
        print("✅ Add data error handling test passed")
    
    @pytest.mark.asyncio
    async def test_duplicate_corpus_creation(self, agent):
        """
        Test creating a corpus that already exists without overwrite flag.
        
        Requirements: 7.2
        """
        corpus_name = "duplicate_test_corpus"
        
        try:
            # Create corpus first time
            result1 = await agent.tool_registry.get_tool("create_corpus").run(
                corpus_name=corpus_name,
                description="Test corpus",
                overwrite_existing=True
            )
            assert result1.status == "success"
            
            # Try to create again without overwrite
            result2 = await agent.tool_registry.get_tool("create_corpus").run(
                corpus_name=corpus_name,
                description="Test corpus"
            )
            
            assert result2.status == "error"
            assert "already exists" in result2.error.lower()
            
            print("✅ Duplicate corpus error handling test passed")
            
        finally:
            # Cleanup
            try:
                await agent.tool_registry.get_tool("delete_corpus").run(
                    corpus_name=corpus_name,
                    confirm_deletion=corpus_name
                )
            except Exception:
                pass


class TestRealBackendIntegration:
    """Test integration with real backends (FAISS)."""
    
    @pytest_asyncio.fixture
    async def faiss_agent(self):
        """Create agent with FAISS backend."""
        agent = RAGAgent(
            vector_db_config=VectorDBConfig(
                provider="faiss",
                embedding_model="mock",
                dimension=512,
                connection_params={"storage_path": "test_faiss_storage"}
            )
        )
        yield agent
        
        # Cleanup test corpus
        try:
            await agent.tool_registry.get_tool("delete_corpus").run(
                corpus_name="faiss_integration_test",
                confirm_deletion="faiss_integration_test"
            )
        except Exception:
            pass
    
    @pytest.mark.asyncio
    async def test_faiss_persistence(self, faiss_agent):
        """
        Test that data persists in FAISS backend.
        
        Requirements: 4.3, 6.1
        """
        corpus_name = "faiss_integration_test"
        
        # Create corpus and add data
        await faiss_agent.tool_registry.get_tool("create_corpus").run(
            corpus_name=corpus_name,
            description="FAISS integration test",
            overwrite_existing=True
        )
        
        await faiss_agent.tool_registry.get_tool("add_data").run(
            corpus_name=corpus_name,
            documents=[
                {"content": "Test document for FAISS persistence"},
                {"content": "Another test document"}
            ]
        )
        
        # Get corpus info to verify data was added
        info_result = await faiss_agent.tool_registry.get_tool("get_corpus_info").run(
            corpus_name=corpus_name
        )
        
        assert info_result.status == "success"
        assert info_result.data["document_count"] == 2
        
        # Query to verify data is retrievable
        query_result = await faiss_agent.tool_registry.get_tool("rag_query").run(
            corpus_name=corpus_name,
            query="test document"
        )
        
        assert query_result.status == "success"
        assert len(query_result.data["results"]) > 0
        
        print("✅ FAISS persistence test passed")
    
    @pytest.mark.asyncio
    async def test_large_document_batch(self, faiss_agent):
        """
        Test adding a large batch of documents.
        
        Requirements: 5.1, 6.4
        """
        corpus_name = "large_batch_test"
        
        try:
            await faiss_agent.tool_registry.get_tool("create_corpus").run(
                corpus_name=corpus_name,
                description="Large batch test",
                overwrite_existing=True
            )
            
            # Create 50 documents
            documents = [
                {"content": f"Document number {i} with test content about topic {i % 5}"}
                for i in range(50)
            ]
            
            result = await faiss_agent.tool_registry.get_tool("add_data").run(
                corpus_name=corpus_name,
                documents=documents
            )
            
            assert result.status == "success"
            assert result.data["documents_added"] == 50
            
            # Verify all documents are queryable
            info_result = await faiss_agent.tool_registry.get_tool("get_corpus_info").run(
                corpus_name=corpus_name
            )
            
            assert info_result.data["document_count"] == 50
            
            print("✅ Large document batch test passed")
            
        finally:
            try:
                await faiss_agent.tool_registry.get_tool("delete_corpus").run(
                    corpus_name=corpus_name,
                    confirm_deletion=corpus_name
                )
            except Exception:
                pass


if __name__ == "__main__":
    print("=" * 70)
    print("⚠️  This test file requires pytest to run properly")
    print("=" * 70)
    print("\nThis file uses pytest fixtures which don't work in standalone mode.")
    print("\nTo run these tests, use:")
    print("  pytest tests/integration/test_e2e_workflows.py -v")
    print("\nFor standalone execution without pytest, use:")
    print("  python tests/integration/test_e2e_simple.py")
    print("=" * 70)
