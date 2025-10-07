#!/usr/bin/env python3

"""
Test script to verify the enhanced context management functionality.
"""

import asyncio
import sys
from typing import Dict, Any

# Add the current directory to Python path for imports
sys.path.insert(0, '.')

from lexora.rag_agent.planner import ExecutionPlan, PlanStep, PlanStepType, PlanStatus
from lexora.rag_agent.executor import AgentExecutor, ExecutionContext, ExecutionResult, ContextManager
from lexora.tools import ToolRegistry
from lexora.tools import CreateCorpusTool, AddDataTool, RAGQueryTool, ListCorporaTool
from lexora.llm.base_llm import MockLLMProvider
from lexora.vector_db.base_vector_db import MockVectorDB
from lexora.utils.embeddings import create_mock_embedding_manager
from lexora.utils.chunking import TextChunker


async def setup_test_environment():
    """Set up test environment with tools and dependencies."""
    print("📋 Setting up test environment...")
    
    # Create dependencies
    vector_db = MockVectorDB()
    await vector_db.connect()
    
    embedding_manager = create_mock_embedding_manager(dimension=384)
    text_chunker = TextChunker()
    
    # Create tool registry and register tools
    tool_registry = ToolRegistry()
    
    # Register RAG tools
    create_corpus_tool = CreateCorpusTool(vector_db=vector_db)
    add_data_tool = AddDataTool(vector_db=vector_db, embedding_manager=embedding_manager, text_chunker=text_chunker)
    rag_query_tool = RAGQueryTool(vector_db=vector_db, embedding_manager=embedding_manager)
    list_corpora_tool = ListCorporaTool(vector_db=vector_db)
    
    tool_registry.register_tool(create_corpus_tool, category="corpus_management")
    tool_registry.register_tool(add_data_tool, category="data_management")
    tool_registry.register_tool(rag_query_tool, category="search")
    tool_registry.register_tool(list_corpora_tool, category="corpus_management")
    
    print("✅ Test environment setup complete")
    return tool_registry, vector_db


async def test_context_manager():
    """Test the ContextManager functionality."""
    print("🧪 Testing ContextManager...")
    
    # Create context manager
    context_manager = ContextManager(max_size=1000, strategy="sliding_window")
    
    # Create execution context
    context = ExecutionContext(
        plan_id="test_plan",
        max_context_size=1000,
        context_truncation_strategy="sliding_window",
        preserve_recent_steps=3
    )
    
    # Add some step results to fill up context
    for i in range(10):
        large_result = {"data": "x" * 100, "step_number": i}  # Each result is ~100 chars
        context.add_step_result(f"step_{i}", large_result)
    
    initial_size = context.get_context_size()
    print(f"Initial context size: {initial_size}")
    
    # Test context management
    managed_context = context_manager.manage_context(context)
    final_size = managed_context.get_context_size()
    print(f"Final context size: {final_size}")
    
    assert final_size <= initial_size, "Context size should be reduced or stay the same"
    assert len(managed_context.step_results) <= context.preserve_recent_steps + 2, "Should preserve recent steps"
    
    # Test context summary
    summary = context_manager.get_context_summary(managed_context)
    assert "total_size" in summary
    assert "strategy" in summary
    print("✅ ContextManager tests passed")


async def test_execution_context():
    """Test the ExecutionContext functionality."""
    print("🧪 Testing ExecutionContext...")
    
    # Create execution context
    context = ExecutionContext(
        plan_id="test_plan",
        user_id="test_user",
        session_id="test_session"
    )
    
    # Test basic properties
    assert context.plan_id == "test_plan"
    assert context.user_id == "test_user"
    assert context.session_id == "test_session"
    print("✅ Basic properties working")
    
    # Test adding step results
    context.add_step_result("step_1", {"result": "test_result_1"})
    context.add_step_result("step_2", {"result": "test_result_2"})
    
    assert "step_1" in context.step_results
    assert "step_2" in context.step_results
    assert len(context.context_history) == 2
    print("✅ Step result tracking working")
    
    # Test shared context
    context.update_shared_context("corpus_name", "test_corpus")
    context.update_shared_context("user_query", "test query")
    
    assert context.shared_context["corpus_name"] == "test_corpus"
    assert context.shared_context["user_query"] == "test query"
    print("✅ Shared context working")
    
    # Test relevant context extraction
    relevant_context = context.get_relevant_context(["step_1"])
    assert "shared_context" in relevant_context
    assert "step_results" in relevant_context
    assert "step_1" in relevant_context["step_results"]
    assert "step_2" not in relevant_context["step_results"]  # Not a dependency
    print("✅ Relevant context extraction working")
    
    # Test context size calculation
    size = context.get_context_size()
    assert size > 0
    print(f"✅ Context size calculation working: {size} characters")
    
    print("✅ ExecutionContext tests passed")


async def test_enhanced_executor():
    """Test the enhanced executor with context management."""
    print("🧪 Testing Enhanced AgentExecutor...")
    
    tool_registry, vector_db = await setup_test_environment()
    
    # Create executor with context management settings
    executor = AgentExecutor(
        tool_registry=tool_registry,
        max_context_size=5000,
        context_truncation_strategy="sliding_window",
        enable_step_retry=True,
        max_retry_attempts=2
    )
    
    # Test basic properties
    assert hasattr(executor, 'context_manager')
    assert hasattr(executor, 'execution_contexts')
    print("✅ Enhanced executor initialization working")
    
    # Create a test plan with parameter substitution
    test_plan = ExecutionPlan(
        id="context_test_plan",
        query="Test context management",
        steps=[
            PlanStep(
                id="step_1",
                type=PlanStepType.INFORMATION_GATHERING,
                description="Gather initial information",
                expected_output="Initial context data"
            ),
            PlanStep(
                id="step_2",
                type=PlanStepType.ANALYSIS,
                description="Analyze gathered information",
                dependencies=["step_1"],
                expected_output="Analysis results"
            ),
            PlanStep(
                id="step_3",
                type=PlanStepType.SYNTHESIS,
                description="Synthesize final results",
                dependencies=["step_1", "step_2"],
                expected_output="Final synthesis"
            )
        ]
    )
    
    # Test plan execution with context management
    context = ExecutionContext(
        plan_id=test_plan.id,
        max_context_size=5000,
        preserve_recent_steps=2
    )
    
    result = await executor.execute_plan(test_plan, context)
    
    assert isinstance(result, ExecutionResult)
    print("✅ Enhanced plan execution working")
    
    # Test context information retrieval
    context_info = executor.get_context_info(test_plan.id)
    if context_info:  # Context might be cleaned up after execution
        assert "context_size" in context_info
        assert "step_results_count" in context_info
        print("✅ Context information retrieval working")
    
    # Test parameter substitution
    parameters = {
        "corpus_name": "${shared.corpus_name}",
        "query": "${step_1.result}",
        "top_k": 10
    }
    
    test_context = {
        "shared_context": {"corpus_name": "test_corpus"},
        "step_results": {"step_1": {"result": "test query"}}
    }
    
    substituted = executor._substitute_parameters_from_context(parameters, test_context)
    
    assert substituted["corpus_name"] == "test_corpus"
    assert substituted["query"] == "test query"
    assert substituted["top_k"] == 10
    print("✅ Parameter substitution working")
    
    await vector_db.disconnect()
    print("✅ Enhanced AgentExecutor tests passed")


async def test_retry_logic():
    """Test the retry logic functionality."""
    print("🧪 Testing Retry Logic...")
    
    tool_registry, vector_db = await setup_test_environment()
    
    # Create executor with retry enabled
    executor = AgentExecutor(
        tool_registry=tool_registry,
        enable_step_retry=True,
        max_retry_attempts=3,
        retry_delay=0.1,  # Short delay for testing
        retry_backoff_factor=1.5
    )
    
    # Test retry decision logic
    step = PlanStep(
        id="test_step",
        type=PlanStepType.TOOL_EXECUTION,
        description="Test step"
    )
    
    # Should retry for general failures
    should_retry = executor._should_retry_step(step, None, 0)
    assert should_retry == True
    print("✅ Retry decision for general failures working")
    
    # Should not retry for parameter errors
    param_error_result = ExecutionResult(
        success=False,
        error="Missing required parameters: ['test_param']"
    )
    should_retry = executor._should_retry_step(step, param_error_result, 0)
    assert should_retry == False
    print("✅ Retry decision for parameter errors working")
    
    # Should not retry after max attempts
    should_retry = executor._should_retry_step(step, None, 3)
    assert should_retry == False
    print("✅ Retry decision for max attempts working")
    
    await vector_db.disconnect()
    print("✅ Retry Logic tests passed")


async def main():
    """Run all context management tests."""
    print("🚀 Starting Context Management Tests\n")
    
    try:
        await test_context_manager()
        await test_execution_context()
        await test_enhanced_executor()
        await test_retry_logic()
        
        print("\n🎉 All context management tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())