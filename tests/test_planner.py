"""
Unit tests for the AgentPlanner component.

Tests cover:
- Query analysis and tool selection
- Execution plan generation for complex queries
- Plan optimization and adaptive planning
"""

import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from rag_agent.planner import (
    AgentPlanner,
    ExecutionPlan,
    PlanStep,
    PlanStepType,
    PlanStatus,
    create_agent_planner
)
from tools import ToolRegistry
from tools.base_tool import BaseTool, ToolResult, ToolStatus
from llm.base_llm import MockLLMProvider


class MockTool(BaseTool):
    """Mock tool for testing."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.version = "1.0.0"
    
    async def run(self, **kwargs) -> ToolResult:
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={"result": f"Mock result from {self.name}"}
        )
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "param1": {"type": "string"},
                "param2": {"type": "integer"}
            },
            "required": ["param1"]
        }


@pytest.fixture
def tool_registry():
    """Create a tool registry with mock tools."""
    registry = ToolRegistry()
    
    # Register mock tools
    registry.register_tool(MockTool("search_tool", "Search for documents"), category="search")
    registry.register_tool(MockTool("create_tool", "Create a corpus"), category="management")
    registry.register_tool(MockTool("update_tool", "Update documents"), category="management")
    registry.register_tool(MockTool("delete_tool", "Delete documents"), category="management")
    
    return registry


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def planner(mock_llm, tool_registry):
    """Create an AgentPlanner instance."""
    return AgentPlanner(llm=mock_llm, tool_registry=tool_registry)


class TestAgentPlannerInitialization:
    """Test AgentPlanner initialization."""
    
    def test_planner_initialization(self, planner, mock_llm, tool_registry):
        """Test that planner initializes correctly."""
        assert planner.llm == mock_llm
        assert planner.tool_registry == tool_registry
        assert planner.max_plan_steps == 20
        assert planner.enable_step_optimization is True
    
    def test_planner_custom_config(self, mock_llm, tool_registry):
        """Test planner with custom configuration."""
        planner = AgentPlanner(
            llm=mock_llm,
            tool_registry=tool_registry,
            max_plan_steps=10,
            enable_step_optimization=False
        )
        
        assert planner.max_plan_steps == 10
        assert planner.enable_step_optimization is False
    
    def test_create_agent_planner_function(self, mock_llm, tool_registry):
        """Test the create_agent_planner convenience function."""
        planner = create_agent_planner(
            llm=mock_llm,
            tool_registry=tool_registry,
            max_plan_steps=15
        )
        
        assert isinstance(planner, AgentPlanner)
        assert planner.max_plan_steps == 15


class TestQueryAnalysis:
    """Test query analysis functionality."""
    
    @pytest.mark.asyncio
    async def test_analyze_simple_query(self, planner):
        """Test analysis of a simple query."""
        query = "Search for documents about machine learning"
        analysis = await planner._analyze_query(query, None)
        
        assert isinstance(analysis, dict)
        assert "intent" in analysis
        assert "operations" in analysis
        assert "complexity" in analysis
        assert "confidence" in analysis
    
    @pytest.mark.asyncio
    async def test_analyze_complex_query(self, planner):
        """Test analysis of a complex query."""
        query = "Create a new corpus, add documents, then search for relevant information"
        analysis = await planner._analyze_query(query, None)
        
        assert isinstance(analysis, dict)
        assert analysis["complexity"] in ["simple", "moderate", "complex"]
    
    @pytest.mark.asyncio
    async def test_analyze_query_with_context(self, planner):
        """Test query analysis with additional context."""
        query = "Search for documents"
        context = {"corpus_name": "test_corpus", "user_id": "user123"}
        
        analysis = await planner._analyze_query(query, context)
        
        assert isinstance(analysis, dict)
        assert "intent" in analysis


class TestToolSelection:
    """Test tool selection functionality."""
    
    @pytest.mark.asyncio
    async def test_get_available_tools_info(self, planner):
        """Test getting information about available tools."""
        tools_info = await planner._get_available_tools_info()
        
        assert isinstance(tools_info, list)
        assert len(tools_info) > 0
        
        # Check tool info structure
        for tool_info in tools_info:
            assert "name" in tool_info
            assert "description" in tool_info
            assert "parameters" in tool_info
            assert "version" in tool_info
    
    @pytest.mark.asyncio
    async def test_tool_info_contains_schema(self, planner):
        """Test that tool info includes parameter schema."""
        tools_info = await planner._get_available_tools_info()
        
        # Find a specific tool
        search_tool = next((t for t in tools_info if t["name"] == "search_tool"), None)
        assert search_tool is not None
        assert "parameters" in search_tool
        assert isinstance(search_tool["parameters"], dict)


class TestPlanGeneration:
    """Test execution plan generation."""
    
    @pytest.mark.asyncio
    async def test_create_simple_plan(self, planner):
        """Test creating a simple execution plan."""
        query = "Search for documents"
        plan = await planner.create_plan(query)
        
        assert isinstance(plan, ExecutionPlan)
        assert plan.query == query
        assert len(plan.steps) > 0
        assert plan.status == PlanStatus.PENDING
        assert plan.id.startswith("plan_")
    
    @pytest.mark.asyncio
    async def test_create_complex_plan(self, planner):
        """Test creating a complex execution plan."""
        query = "Create a corpus, add documents, and search for information"
        plan = await planner.create_plan(query)
        
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.steps) >= 1  # Fallback creates at least one step
        
        # Check plan metadata
        assert "query_analysis" in plan.metadata
        assert "available_tools" in plan.metadata
    
    @pytest.mark.asyncio
    async def test_plan_with_context(self, planner):
        """Test creating a plan with additional context."""
        query = "Search for documents"
        context = {"corpus_name": "test_corpus"}
        
        plan = await planner.create_plan(query, context)
        
        assert isinstance(plan, ExecutionPlan)
        assert plan.metadata["context"] == context
    
    @pytest.mark.asyncio
    async def test_plan_steps_have_correct_structure(self, planner):
        """Test that generated plan steps have correct structure."""
        query = "Search for documents"
        plan = await planner.create_plan(query)
        
        for step in plan.steps:
            assert isinstance(step, PlanStep)
            assert step.id is not None
            assert step.type in PlanStepType
            assert step.description is not None
            assert step.status == PlanStatus.PENDING
            assert isinstance(step.dependencies, list)


class TestFallbackPlanning:
    """Test fallback planning when LLM fails."""
    
    @pytest.mark.asyncio
    async def test_fallback_for_information_retrieval(self, planner):
        """Test fallback plan for information retrieval queries."""
        analysis = {
            "intent": "information_retrieval",
            "operations": ["search"],
            "complexity": "simple"
        }
        
        steps = planner._create_fallback_plan("test query", analysis)
        
        assert len(steps) > 0
        assert steps[0].type == PlanStepType.TOOL_EXECUTION
        assert steps[0].tool_name == "rag_query"
    
    @pytest.mark.asyncio
    async def test_fallback_for_document_management(self, planner):
        """Test fallback plan for document management queries."""
        analysis = {
            "intent": "document_management",
            "operations": ["create"],
            "complexity": "simple"
        }
        
        steps = planner._create_fallback_plan("create corpus", analysis)
        
        assert len(steps) > 0
        assert steps[0].type == PlanStepType.TOOL_EXECUTION
        assert steps[0].tool_name == "create_corpus"
    
    @pytest.mark.asyncio
    async def test_fallback_for_unknown_intent(self, planner):
        """Test fallback plan for unknown intent."""
        analysis = {
            "intent": "unknown",
            "operations": [],
            "complexity": "simple"
        }
        
        steps = planner._create_fallback_plan("unknown query", analysis)
        
        assert len(steps) > 0
        assert steps[0].type == PlanStepType.INFORMATION_GATHERING


class TestPlanOptimization:
    """Test plan optimization functionality."""
    
    @pytest.mark.asyncio
    async def test_optimize_removes_duplicates(self, planner):
        """Test that optimization removes duplicate steps."""
        steps = [
            PlanStep(
                id="step_1",
                type=PlanStepType.TOOL_EXECUTION,
                description="Search",
                tool_name="search_tool",
                tool_parameters={"query": "test"}
            ),
            PlanStep(
                id="step_2",
                type=PlanStepType.TOOL_EXECUTION,
                description="Search again",
                tool_name="search_tool",
                tool_parameters={"query": "test"}  # Same parameters
            ),
            PlanStep(
                id="step_3",
                type=PlanStepType.TOOL_EXECUTION,
                description="Different search",
                tool_name="search_tool",
                tool_parameters={"query": "different"}
            )
        ]
        
        optimized = await planner._optimize_plan_steps(steps)
        
        # Should remove step_2 as it's duplicate of step_1
        assert len(optimized) == 2
        assert optimized[0].id == "step_1"
        assert optimized[1].id == "step_3"
    
    @pytest.mark.asyncio
    async def test_optimize_preserves_unique_steps(self, planner):
        """Test that optimization preserves unique steps."""
        steps = [
            PlanStep(
                id="step_1",
                type=PlanStepType.TOOL_EXECUTION,
                description="Search",
                tool_name="search_tool",
                tool_parameters={"query": "test1"}
            ),
            PlanStep(
                id="step_2",
                type=PlanStepType.TOOL_EXECUTION,
                description="Create",
                tool_name="create_tool",
                tool_parameters={"name": "corpus"}
            )
        ]
        
        optimized = await planner._optimize_plan_steps(steps)
        
        assert len(optimized) == 2


class TestPlanUpdate:
    """Test plan update and adaptive planning."""
    
    @pytest.mark.asyncio
    async def test_update_plan_with_results(self, planner):
        """Test updating a plan with step results."""
        query = "Search for documents"
        plan = await planner.create_plan(query)
        
        # Simulate step results
        step_results = {
            plan.steps[0].id: {"result": "test result"}
        }
        
        updated_plan = await planner.update_plan(plan, step_results)
        
        assert isinstance(updated_plan, ExecutionPlan)
        assert updated_plan.steps[0].result == {"result": "test result"}
        assert updated_plan.steps[0].status == PlanStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_check_replanning_not_needed(self, planner):
        """Test that replanning is not needed for successful results."""
        query = "Search for documents"
        plan = await planner.create_plan(query)
        
        step_results = {
            plan.steps[0].id: {"result": "success"}
        }
        
        needs_replanning = await planner._check_replanning_needed(plan, step_results)
        
        assert needs_replanning is False
    
    @pytest.mark.asyncio
    async def test_generate_adaptive_steps_for_no_results(self, planner):
        """Test generating adaptive steps when search returns no results."""
        query = "Search for documents"
        plan = await planner.create_plan(query)
        
        # Simulate a search that returned no results
        step_results = {
            plan.steps[0].id: {"total_count": 0, "results": []}
        }
        
        # Update the step to use rag_query tool
        plan.steps[0].tool_name = "rag_query"
        
        additional_steps = await planner._generate_adaptive_steps(plan, step_results)
        
        # Should generate a retry step with broader search
        assert len(additional_steps) > 0
        assert additional_steps[0].tool_name == "rag_query"
        assert additional_steps[0].dependencies == [plan.steps[0].id]


class TestExecutionPlanMethods:
    """Test ExecutionPlan class methods."""
    
    def test_plan_to_dict(self):
        """Test converting plan to dictionary."""
        plan = ExecutionPlan(
            id="test_plan",
            query="test query",
            steps=[
                PlanStep(
                    id="step_1",
                    type=PlanStepType.TOOL_EXECUTION,
                    description="Test step"
                )
            ]
        )
        
        plan_dict = plan.to_dict()
        
        assert isinstance(plan_dict, dict)
        assert plan_dict["id"] == "test_plan"
        assert plan_dict["query"] == "test query"
        assert len(plan_dict["steps"]) == 1
        assert plan_dict["status"] == "pending"
    
    def test_get_step_by_id(self):
        """Test getting a step by ID."""
        step1 = PlanStep(id="step_1", type=PlanStepType.TOOL_EXECUTION, description="Step 1")
        step2 = PlanStep(id="step_2", type=PlanStepType.ANALYSIS, description="Step 2")
        
        plan = ExecutionPlan(
            id="test_plan",
            query="test",
            steps=[step1, step2]
        )
        
        found_step = plan.get_step("step_2")
        
        assert found_step is not None
        assert found_step.id == "step_2"
        assert found_step.description == "Step 2"
    
    def test_get_ready_steps(self):
        """Test getting steps that are ready to execute."""
        step1 = PlanStep(id="step_1", type=PlanStepType.TOOL_EXECUTION, description="Step 1")
        step1.status = PlanStatus.COMPLETED
        
        step2 = PlanStep(
            id="step_2",
            type=PlanStepType.ANALYSIS,
            description="Step 2",
            dependencies=["step_1"]
        )
        
        step3 = PlanStep(
            id="step_3",
            type=PlanStepType.SYNTHESIS,
            description="Step 3",
            dependencies=["step_2"]
        )
        
        plan = ExecutionPlan(
            id="test_plan",
            query="test",
            steps=[step1, step2, step3]
        )
        
        ready_steps = plan.get_ready_steps()
        
        # Only step_2 should be ready (step_1 is complete, step_3 depends on step_2)
        assert len(ready_steps) == 1
        assert ready_steps[0].id == "step_2"
    
    def test_is_complete(self):
        """Test checking if plan is complete."""
        step1 = PlanStep(id="step_1", type=PlanStepType.TOOL_EXECUTION, description="Step 1")
        step1.status = PlanStatus.COMPLETED
        
        step2 = PlanStep(id="step_2", type=PlanStepType.ANALYSIS, description="Step 2")
        step2.status = PlanStatus.COMPLETED
        
        plan = ExecutionPlan(
            id="test_plan",
            query="test",
            steps=[step1, step2]
        )
        
        assert plan.is_complete() is True
    
    def test_has_failed_steps(self):
        """Test checking if plan has failed steps."""
        step1 = PlanStep(id="step_1", type=PlanStepType.TOOL_EXECUTION, description="Step 1")
        step1.status = PlanStatus.COMPLETED
        
        step2 = PlanStep(id="step_2", type=PlanStepType.ANALYSIS, description="Step 2")
        step2.status = PlanStatus.FAILED
        
        plan = ExecutionPlan(
            id="test_plan",
            query="test",
            steps=[step1, step2]
        )
        
        assert plan.has_failed_steps() is True


class TestPlanStepMethods:
    """Test PlanStep class methods."""
    
    def test_step_to_dict(self):
        """Test converting step to dictionary."""
        step = PlanStep(
            id="step_1",
            type=PlanStepType.TOOL_EXECUTION,
            description="Test step",
            tool_name="test_tool",
            tool_parameters={"param": "value"},
            dependencies=["step_0"]
        )
        
        step_dict = step.to_dict()
        
        assert isinstance(step_dict, dict)
        assert step_dict["id"] == "step_1"
        assert step_dict["type"] == "tool_execution"
        assert step_dict["description"] == "Test step"
        assert step_dict["tool_name"] == "test_tool"
        assert step_dict["tool_parameters"] == {"param": "value"}
        assert step_dict["dependencies"] == ["step_0"]


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_create_plan_with_empty_query(self, planner):
        """Test creating a plan with an empty query."""
        plan = await planner.create_plan("")
        
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.steps) > 0  # Should still create a fallback plan
    
    @pytest.mark.asyncio
    async def test_create_plan_with_very_long_query(self, planner):
        """Test creating a plan with a very long query."""
        long_query = "search " * 1000  # Very long query
        plan = await planner.create_plan(long_query)
        
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.steps) <= planner.max_plan_steps
    
    @pytest.mark.asyncio
    async def test_update_plan_with_empty_results(self, planner):
        """Test updating a plan with empty results."""
        query = "Search for documents"
        plan = await planner.create_plan(query)
        
        updated_plan = await planner.update_plan(plan, {})
        
        assert isinstance(updated_plan, ExecutionPlan)
    
    def test_get_nonexistent_step(self):
        """Test getting a step that doesn't exist."""
        plan = ExecutionPlan(
            id="test_plan",
            query="test",
            steps=[]
        )
        
        step = plan.get_step("nonexistent")
        
        assert step is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
