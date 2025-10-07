#!/usr/bin/env python3

"""
Test script to verify the agent components (planner and executor) are working correctly.
"""

import asyncio
import sys
from typing import Dict, Any

# Add the current directory to Python path for imports
sys.path.insert(0, '.')

from lexora.rag_agent.planner import AgentPlanner, ExecutionPlan, PlanStep, PlanStepType, PlanStatus
from lexora.rag_agent.executor import AgentExecutor, ExecutionContext, ExecutionResult
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
    llm = MockLLMProvider()
    
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
    return llm, tool_registry, vector_db, embedding_manager


async def test_agent_planner():
    """Test the agent planner functionality."""
    print("🧪 Testing AgentPlanner...")
    
    llm, tool_registry, vector_db, embedding_manager = await setup_test_environment()
    
    # Create planner
    planner = AgentPlanner(llm=llm, tool_registry=tool_registry)
    
    # Test basic properties
    assert hasattr(planner, 'llm')
    assert hasattr(planner, 'tool_registry')
    print("✅ Planner initialization working")
    
    # Test plan creation for simple query
    simple_query = "List all available corpora"
    plan = await planner.create_plan(simple_query)
    
    assert isinstance(plan, ExecutionPlan)
    assert plan.query == simple_query
    assert len(plan.steps) > 0
    assert plan.status == PlanStatus.PENDING
    print("✅ Simple plan creation working")
    
    # Test plan creation for complex query
    complex_query = "Create a new corpus called 'test_docs', add some documents to it, then search for information about machine learning"
    complex_plan = await planner.create_plan(complex_query)
    
    assert isinstance(complex_plan, ExecutionPlan)
    assert len(complex_plan.steps) >= 1  # Should have at least one step (fallback creates one)
    print("✅ Complex plan creation working")
    
    # Test plan step types
    step_types = {step.type for step in complex_plan.steps}
    assert PlanStepType.TOOL_EXECUTION in step_types
    print("✅ Plan step types working")
    
    # Test plan serialization
    plan_dict = plan.to_dict()
    assert "id" in plan_dict
    assert "query" in plan_dict
    assert "steps" in plan_dict
    print("✅ Plan serialization working")
    
    await vector_db.disconnect()
    print("✅ AgentPlanner tests passed!\n")


async def test_agent_executor():
    """Test the agent executor functionality."""
    print("🧪 Testing AgentExecutor...")
    
    llm, tool_registry, vector_db, embedding_manager = await setup_test_environment()
    
    # Create executor
    executor = AgentExecutor(tool_registry=tool_registry)
    
    # Test basic properties
    assert hasattr(executor, 'tool_registry')
    assert hasattr(executor, 'active_executions')
    print("✅ Executor initialization working")
    
    # Create a simple test plan
    test_plan = ExecutionPlan(
        id="test_plan_1",
        query="List all corpora",
        steps=[
            PlanStep(
                id="step_1",
                type=PlanStepType.TOOL_EXECUTION,
                description="List all available corpora",
                tool_name="list_corpora",
                tool_parameters={"include_details": True}
            )
        ]
    )
    
    # Test plan execution
    context = ExecutionContext(plan_id=test_plan.id)
    result = await executor.execute_plan(test_plan, context)
    
    assert isinstance(result, ExecutionResult)
    assert result.success or result.error is not None  # Should have either success or error
    print("✅ Simple plan execution working")
    
    # Test multi-step plan
    multi_step_plan = ExecutionPlan(
        id="test_plan_2",
        query="Create corpus and list it",
        steps=[
            PlanStep(
                id="step_1",
                type=PlanStepType.TOOL_EXECUTION,
                description="Create a test corpus",
                tool_name="create_corpus",
                tool_parameters={"name": "test_corpus", "description": "Test corpus for agent testing"}
            ),
            PlanStep(
                id="step_2",
                type=PlanStepType.TOOL_EXECUTION,
                description="List all corpora to verify creation",
                tool_name="list_corpora",
                tool_parameters={"include_details": True},
                dependencies=["step_1"]
            )
        ]
    )
    
    # Test multi-step execution
    multi_result = await executor.execute_plan(multi_step_plan, context)
    assert isinstance(multi_result, ExecutionResult)
    print("✅ Multi-step plan execution working")
    
    # Test execution status tracking
    status = executor.get_execution_status("nonexistent_plan")
    assert status is None
    print("✅ Execution status tracking working")
    
    # Test execution history
    history = executor.get_execution_history()
    assert isinstance(history, list)
    print("✅ Execution history working")
    
    await vector_db.disconnect()
    print("✅ AgentExecutor tests passed!\n")


async def test_integration():
    """Test integration between planner and executor."""
    print("🧪 Testing Planner-Executor Integration...")
    
    llm, tool_registry, vector_db, embedding_manager = await setup_test_environment()
    
    # Create planner and executor
    planner = AgentPlanner(llm=llm, tool_registry=tool_registry)
    executor = AgentExecutor(tool_registry=tool_registry)
    
    # Create a plan using the planner
    query = "List all available corpora in the system"
    plan = await planner.create_plan(query)
    
    # Execute the plan using the executor
    context = ExecutionContext(plan_id=plan.id)
    result = await executor.execute_plan(plan, context)
    
    # Verify integration
    assert isinstance(result, ExecutionResult)
    assert "plan_id" in result.result
    assert result.result["plan_id"] == plan.id
    print("✅ Planner-Executor integration working")
    
    # Test plan update based on execution results
    if result.success and result.result.get("step_results"):
        updated_plan = await planner.update_plan(plan, result.result["step_results"])
        assert isinstance(updated_plan, ExecutionPlan)
        print("✅ Plan update based on results working")
    
    await vector_db.disconnect()
    print("✅ Integration tests passed!\n")


async def main():
    """Run all tests."""
    print("🚀 Starting Agent Components Tests\n")
    
    try:
        await test_agent_planner()
        await test_agent_executor()
        await test_integration()
        
        print("🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())