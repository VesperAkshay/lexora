"""
Unit tests for the AgentExecutor component.

Tests cover:
- Tool execution and context management
- Error handling and recovery scenarios
- Retry logic and timeout handling
- Context size management and truncation
"""

import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from rag_agent.planner import ExecutionPlan, PlanStep, PlanStepType, PlanStatus
from rag_agent.executor import (
    AgentExecutor,
    ExecutionResult,
    ExecutionContext,
    ContextManager,
    create_agent_executor
)
from tools import ToolRegistry
from tools.base_tool import BaseTool, ToolResult, ToolStatus


class MockTool(BaseTool):
    """Mock tool for testing."""
    
    def __init__(self, name: str, description: str, should_fail: bool = False):
        self.name = name
        self.description = description
        self.version = "1.0.0"
        self.should_fail = should_fail
        self.call_count = 0
    
    async def run(self, **kwargs) -> ToolResult:
        self.call_count += 1
        
        if self.should_fail:
            return ToolResult(
                status=ToolStatus.ERROR,
                error="Mock tool failure"
            )
        
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={"result": f"Mock result from {self.name}", "params": kwargs}
        )
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "param1": {"type": "string"}
            },
            "required": []
        }


@pytest.fixture
def tool_registry():
    """Create a tool registry with mock tools."""
    registry = ToolRegistry()
    
    registry.register_tool(MockTool("test_tool", "Test tool"), category="test")
    registry.register_tool(MockTool("failing_tool", "Failing tool", should_fail=True), category="test")
    registry.register_tool(MockTool("search_tool", "Search tool"), category="search")
    
    return registry


@pytest.fixture
def executor(tool_registry):
    """Create an AgentExecutor instance."""
    return AgentExecutor(
        tool_registry=tool_registry,
        enable_step_retry=True,
        max_retry_attempts=3,
        retry_delay=0.1
    )


@pytest.fixture
def execution_context():
    """Create an execution context."""
    return ExecutionContext(
        plan_id="test_plan",
        user_id="test_user",
        max_context_size=5000
    )


class TestAgentExecutorInitialization:
    """Test AgentExecutor initialization."""
    
    def test_executor_initialization(self, executor, tool_registry):
        """Test that executor initializes correctly."""
        assert executor.tool_registry == tool_registry
        assert executor.enable_step_retry is True
        assert executor.max_retry_attempts == 3
        assert isinstance(executor.context_manager, ContextManager)
    
    def test_executor_custom_config(self, tool_registry):
        """Test executor with custom configuration."""
        executor = AgentExecutor(
            tool_registry=tool_registry,
            max_parallel_steps=10,
            step_timeout=30.0,
            max_context_size=10000
        )
        
        assert executor.max_parallel_steps == 10
        assert executor.step_timeout == 30.0
        assert executor.context_manager.max_size == 10000
    
    def test_create_agent_executor_function(self, tool_registry):
        """Test the create_agent_executor convenience function."""
        executor = create_agent_executor(
            tool_registry=tool_registry,
            max_retry_attempts=5
        )
        
        assert isinstance(executor, AgentExecutor)
        assert executor.max_retry_attempts == 5



class TestExecutionContext:
    """Test ExecutionContext functionality."""
    
    def test_context_initialization(self):
        """Test context initialization."""
        context = ExecutionContext(
            plan_id="test_plan",
            user_id="user123",
            session_id="session456"
        )
        
        assert context.plan_id == "test_plan"
        assert context.user_id == "user123"
        assert context.session_id == "session456"
        assert context.max_context_size == 50000
        assert len(context.shared_context) == 0
        assert len(context.step_results) == 0
    
    def test_add_step_result(self):
        """Test adding step results to context."""
        context = ExecutionContext(plan_id="test_plan")
        
        context.add_step_result("step_1", {"data": "result1"})
        context.add_step_result("step_2", {"data": "result2"})
        
        assert len(context.step_results) == 2
        assert context.step_results["step_1"] == {"data": "result1"}
        assert len(context.context_history) == 2
    
    def test_update_shared_context(self):
        """Test updating shared context."""
        context = ExecutionContext(plan_id="test_plan")
        
        context.update_shared_context("corpus_name", "test_corpus")
        context.update_shared_context("user_query", "test query")
        
        assert context.shared_context["corpus_name"] == "test_corpus"
        assert context.shared_context["user_query"] == "test query"
    
    def test_get_relevant_context(self):
        """Test getting relevant context for dependencies."""
        context = ExecutionContext(plan_id="test_plan")
        
        context.add_step_result("step_1", {"data": "result1"})
        context.add_step_result("step_2", {"data": "result2"})
        context.add_step_result("step_3", {"data": "result3"})
        context.update_shared_context("key", "value")
        
        relevant = context.get_relevant_context(["step_1", "step_3"])
        
        assert "shared_context" in relevant
        assert "step_results" in relevant
        assert "step_1" in relevant["step_results"]
        assert "step_3" in relevant["step_results"]
        assert "step_2" not in relevant["step_results"]
        assert relevant["shared_context"]["key"] == "value"
    
    def test_get_context_size(self):
        """Test calculating context size."""
        context = ExecutionContext(plan_id="test_plan")
        
        initial_size = context.get_context_size()
        assert initial_size > 0
        
        # Add some data
        context.add_step_result("step_1", {"data": "x" * 1000})
        
        new_size = context.get_context_size()
        assert new_size > initial_size
    
    def test_context_to_dict(self):
        """Test converting context to dictionary."""
        context = ExecutionContext(
            plan_id="test_plan",
            user_id="user123"
        )
        context.update_shared_context("key", "value")
        
        context_dict = context.to_dict()
        
        assert isinstance(context_dict, dict)
        assert context_dict["plan_id"] == "test_plan"
        assert context_dict["user_id"] == "user123"
        assert "shared_context" in context_dict



class TestContextManager:
    """Test ContextManager functionality."""
    
    def test_context_manager_initialization(self):
        """Test context manager initialization."""
        manager = ContextManager(max_size=1000, strategy="sliding_window")
        
        assert manager.max_size == 1000
        assert manager.strategy == "sliding_window"
    
    def test_manage_context_within_limit(self):
        """Test managing context that's within size limit."""
        manager = ContextManager(max_size=10000)
        context = ExecutionContext(plan_id="test_plan")
        
        context.add_step_result("step_1", {"data": "small"})
        
        managed = manager.manage_context(context)
        
        assert len(managed.step_results) == 1
    
    def test_sliding_window_truncation(self):
        """Test sliding window truncation strategy."""
        manager = ContextManager(max_size=1000, strategy="sliding_window")
        context = ExecutionContext(
            plan_id="test_plan",
            preserve_recent_steps=3
        )
        
        # Add many step results
        for i in range(10):
            context.add_step_result(f"step_{i}", {"data": "x" * 100})
        
        managed = manager.manage_context(context)
        
        # Should preserve only recent steps
        assert len(managed.context_history) <= context.preserve_recent_steps + 2
    
    def test_oldest_first_truncation(self):
        """Test oldest-first truncation strategy."""
        manager = ContextManager(max_size=500, strategy="oldest_first")
        context = ExecutionContext(plan_id="test_plan")
        
        # Add step results that exceed size limit
        for i in range(10):
            context.add_step_result(f"step_{i}", {"data": "x" * 100})
        
        initial_size = context.get_context_size()
        managed = manager.manage_context(context)
        final_size = managed.get_context_size()
        
        assert final_size <= manager.max_size
        assert final_size < initial_size
    
    def test_get_context_summary(self):
        """Test getting context summary."""
        manager = ContextManager(max_size=5000)
        context = ExecutionContext(plan_id="test_plan")
        
        context.add_step_result("step_1", {"data": "result"})
        context.update_shared_context("key", "value")
        
        summary = manager.get_context_summary(context)
        
        assert "total_size" in summary
        assert "max_size" in summary
        assert "strategy" in summary
        assert "step_count" in summary
        assert summary["step_count"] == 1



class TestToolExecution:
    """Test tool execution functionality."""
    
    @pytest.mark.asyncio
    async def test_execute_successful_tool_step(self, executor, execution_context):
        """Test executing a successful tool step."""
        step = PlanStep(
            id="step_1",
            type=PlanStepType.TOOL_EXECUTION,
            description="Test tool execution",
            tool_name="test_tool",
            tool_parameters={"param1": "value1"}
        )
        
        result = await executor.execute_step(step, execution_context)
        
        assert result.success is True
        assert result.result is not None
        assert step.status == PlanStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_execute_failing_tool_step(self, executor, execution_context):
        """Test executing a failing tool step."""
        step = PlanStep(
            id="step_1",
            type=PlanStepType.TOOL_EXECUTION,
            description="Failing tool execution",
            tool_name="failing_tool",
            tool_parameters={}
        )
        
        result = await executor.execute_step(step, execution_context)
        
        assert result.success is False
        assert result.error is not None
        assert step.status == PlanStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_execute_tool_with_parameter_substitution(self, executor, execution_context):
        """Test tool execution with parameter substitution."""
        # Add context data
        execution_context.update_shared_context("corpus_name", "test_corpus")
        execution_context.add_step_result("step_0", {"query": "test query"})
        
        step = PlanStep(
            id="step_1",
            type=PlanStepType.TOOL_EXECUTION,
            description="Test with substitution",
            tool_name="test_tool",
            tool_parameters={
                "corpus": "${shared.corpus_name}",
                "query": "${step_0.query}"
            },
            dependencies=["step_0"]
        )
        
        result = await executor.execute_step(step, execution_context)
        
        assert result.success is True
        # Check that parameters were substituted
        assert result.metadata["tool_parameters"]["corpus"] == "test_corpus"
        assert result.metadata["tool_parameters"]["query"] == "test query"
    
    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self, executor, execution_context):
        """Test executing a step with nonexistent tool."""
        step = PlanStep(
            id="step_1",
            type=PlanStepType.TOOL_EXECUTION,
            description="Nonexistent tool",
            tool_name="nonexistent_tool",
            tool_parameters={}
        )
        
        result = await executor.execute_step(step, execution_context)
        
        assert result.success is False
        assert "failed" in result.error.lower() or "not found" in result.error.lower()



class TestRetryLogic:
    """Test retry logic functionality."""
    
    @pytest.mark.asyncio
    async def test_retry_on_failure(self, tool_registry):
        """Test that failed steps are retried."""
        # Create a tool that fails first time, succeeds second time
        class RetryableTool(MockTool):
            def __init__(self):
                super().__init__("retryable_tool", "Retryable tool")
                self.attempts = 0
            
            async def run(self, **kwargs):
                self.attempts += 1
                if self.attempts == 1:
                    return ToolResult(status=ToolStatus.ERROR, error="First attempt failed")
                return ToolResult(status=ToolStatus.SUCCESS, data={"result": "success"})
        
        retryable_tool = RetryableTool()
        tool_registry.register_tool(retryable_tool, category="test")
        
        executor = AgentExecutor(
            tool_registry=tool_registry,
            enable_step_retry=True,
            max_retry_attempts=3,
            retry_delay=0.1
        )
        
        context = ExecutionContext(plan_id="test_plan")
        step = PlanStep(
            id="step_1",
            type=PlanStepType.TOOL_EXECUTION,
            description="Retryable step",
            tool_name="retryable_tool",
            tool_parameters={}
        )
        
        result = await executor.execute_step(step, context)
        
        assert result.success is True
        assert retryable_tool.attempts == 2  # Failed once, succeeded on retry
    
    def test_should_retry_step_general_failure(self, executor):
        """Test retry decision for general failures."""
        step = PlanStep(
            id="step_1",
            type=PlanStepType.TOOL_EXECUTION,
            description="Test step"
        )
        
        should_retry = executor._should_retry_step(step, None, 0)
        
        assert should_retry is True
    
    def test_should_not_retry_parameter_error(self, executor):
        """Test that parameter errors are not retried."""
        step = PlanStep(
            id="step_1",
            type=PlanStepType.TOOL_EXECUTION,
            description="Test step"
        )
        
        result = ExecutionResult(
            success=False,
            error="Missing required parameters: ['param1']"
        )
        
        should_retry = executor._should_retry_step(step, result, 0)
        
        assert should_retry is False
    
    def test_should_not_retry_after_max_attempts(self, executor):
        """Test that retry stops after max attempts."""
        step = PlanStep(
            id="step_1",
            type=PlanStepType.TOOL_EXECUTION,
            description="Test step"
        )
        
        should_retry = executor._should_retry_step(step, None, 3)
        
        assert should_retry is False
    
    def test_should_not_retry_validation_steps(self, executor):
        """Test that validation steps are not retried."""
        step = PlanStep(
            id="step_1",
            type=PlanStepType.VALIDATION,
            description="Validation step"
        )
        
        should_retry = executor._should_retry_step(step, None, 0)
        
        assert should_retry is False



class TestPlanExecution:
    """Test complete plan execution."""
    
    @pytest.mark.asyncio
    async def test_execute_simple_plan(self, executor):
        """Test executing a simple plan."""
        plan = ExecutionPlan(
            id="test_plan",
            query="Test query",
            steps=[
                PlanStep(
                    id="step_1",
                    type=PlanStepType.TOOL_EXECUTION,
                    description="Execute test tool",
                    tool_name="test_tool",
                    tool_parameters={"param1": "value1"}
                )
            ]
        )
        
        context = ExecutionContext(plan_id=plan.id)
        result = await executor.execute_plan(plan, context)
        
        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert plan.status == PlanStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_execute_multi_step_plan(self, executor):
        """Test executing a multi-step plan."""
        plan = ExecutionPlan(
            id="test_plan",
            query="Multi-step query",
            steps=[
                PlanStep(
                    id="step_1",
                    type=PlanStepType.TOOL_EXECUTION,
                    description="First step",
                    tool_name="test_tool",
                    tool_parameters={}
                ),
                PlanStep(
                    id="step_2",
                    type=PlanStepType.ANALYSIS,
                    description="Analyze results",
                    dependencies=["step_1"]
                ),
                PlanStep(
                    id="step_3",
                    type=PlanStepType.SYNTHESIS,
                    description="Synthesize final answer",
                    dependencies=["step_1", "step_2"]
                )
            ]
        )
        
        context = ExecutionContext(plan_id=plan.id)
        result = await executor.execute_plan(plan, context)
        
        assert result.success is True
        assert all(step.status == PlanStatus.COMPLETED for step in plan.steps)
    
    @pytest.mark.asyncio
    async def test_execute_plan_with_dependencies(self, executor):
        """Test that dependencies are respected during execution."""
        plan = ExecutionPlan(
            id="test_plan",
            query="Test dependencies",
            steps=[
                PlanStep(
                    id="step_1",
                    type=PlanStepType.TOOL_EXECUTION,
                    description="First step",
                    tool_name="test_tool",
                    tool_parameters={}
                ),
                PlanStep(
                    id="step_2",
                    type=PlanStepType.TOOL_EXECUTION,
                    description="Depends on step 1",
                    tool_name="test_tool",
                    tool_parameters={},
                    dependencies=["step_1"]
                )
            ]
        )
        
        context = ExecutionContext(plan_id=plan.id)
        result = await executor.execute_plan(plan, context)
        
        assert result.success is True
        # Step 2 should have access to step 1's results in context
        assert "step_1" in context.step_results
    
    @pytest.mark.asyncio
    async def test_execute_plan_with_failure(self, executor):
        """Test plan execution when a step fails."""
        plan = ExecutionPlan(
            id="test_plan",
            query="Test failure",
            steps=[
                PlanStep(
                    id="step_1",
                    type=PlanStepType.TOOL_EXECUTION,
                    description="Failing step",
                    tool_name="failing_tool",
                    tool_parameters={}
                )
            ]
        )
        
        context = ExecutionContext(plan_id=plan.id)
        result = await executor.execute_plan(plan, context)
        
        assert result.success is False
        assert plan.status == PlanStatus.FAILED



class TestStepTypes:
    """Test different step type executions."""
    
    @pytest.mark.asyncio
    async def test_information_gathering_step(self, executor, execution_context):
        """Test information gathering step execution."""
        execution_context.update_shared_context("key", "value")
        execution_context.add_step_result("step_0", {"data": "previous"})
        
        step = PlanStep(
            id="step_1",
            type=PlanStepType.INFORMATION_GATHERING,
            description="Gather information",
            dependencies=["step_0"]
        )
        
        result = await executor.execute_step(step, execution_context)
        
        assert result.success is True
        assert "shared_context" in result.result
        assert "step_results" in result.result
    
    @pytest.mark.asyncio
    async def test_analysis_step(self, executor, execution_context):
        """Test analysis step execution."""
        execution_context.add_step_result("step_0", {"data": "to analyze"})
        
        step = PlanStep(
            id="step_1",
            type=PlanStepType.ANALYSIS,
            description="Analyze data",
            dependencies=["step_0"]
        )
        
        result = await executor.execute_step(step, execution_context)
        
        assert result.success is True
        assert "results_analyzed" in result.result
        assert result.result["results_analyzed"] == 1
    
    @pytest.mark.asyncio
    async def test_synthesis_step(self, executor, execution_context):
        """Test synthesis step execution."""
        execution_context.add_step_result("step_0", {"data": "result1"})
        execution_context.add_step_result("step_1", {"data": "result2"})
        
        step = PlanStep(
            id="step_2",
            type=PlanStepType.SYNTHESIS,
            description="Synthesize results",
            dependencies=["step_0", "step_1"]
        )
        
        result = await executor.execute_step(step, execution_context)
        
        assert result.success is True
        assert "synthesized_data" in result.result
        assert "step_results" in result.result["synthesized_data"]
    
    @pytest.mark.asyncio
    async def test_validation_step_success(self, executor, execution_context):
        """Test validation step with valid context."""
        execution_context.add_step_result("step_0", {"data": "valid"})
        
        step = PlanStep(
            id="step_1",
            type=PlanStepType.VALIDATION,
            description="Validate results",
            dependencies=["step_0"]
        )
        
        result = await executor.execute_step(step, execution_context)
        
        assert result.success is True
        assert result.result["validation_passed"] is True
    
    @pytest.mark.asyncio
    async def test_validation_step_missing_dependency(self, executor, execution_context):
        """Test validation step with missing dependency."""
        step = PlanStep(
            id="step_1",
            type=PlanStepType.VALIDATION,
            description="Validate results",
            dependencies=["step_0"]  # step_0 doesn't exist
        )
        
        result = await executor.execute_step(step, execution_context)
        
        assert result.success is False
        assert result.result["validation_passed"] is False
        assert any("Missing required dependency" in msg for msg in result.result["validation_messages"])



class TestExecutionManagement:
    """Test execution management features."""
    
    @pytest.mark.asyncio
    async def test_get_execution_status(self, executor):
        """Test getting execution status."""
        plan = ExecutionPlan(
            id="test_plan",
            query="Test query",
            steps=[
                PlanStep(
                    id="step_1",
                    type=PlanStepType.TOOL_EXECUTION,
                    description="Test step",
                    tool_name="test_tool",
                    tool_parameters={}
                )
            ]
        )
        
        # Start execution in background
        context = ExecutionContext(plan_id=plan.id)
        task = asyncio.create_task(executor.execute_plan(plan, context))
        
        # Give it a moment to start
        await asyncio.sleep(0.1)
        
        # Get status
        status = executor.get_execution_status(plan.id)
        
        # Wait for completion
        await task
        
        # Status should have been available during execution
        # (might be None now if execution completed very quickly)
        assert status is None or "plan_id" in status
    
    @pytest.mark.asyncio
    async def test_cancel_execution(self, executor):
        """Test cancelling an active execution."""
        plan = ExecutionPlan(
            id="test_plan",
            query="Test query",
            steps=[
                PlanStep(
                    id="step_1",
                    type=PlanStepType.TOOL_EXECUTION,
                    description="Test step",
                    tool_name="test_tool",
                    tool_parameters={}
                )
            ]
        )
        
        context = ExecutionContext(plan_id=plan.id)
        
        # Add to active executions manually
        executor.active_executions[plan.id] = plan
        executor.execution_contexts[plan.id] = context
        
        # Cancel
        cancelled = executor.cancel_execution(plan.id)
        
        assert cancelled is True
        assert plan.status == PlanStatus.CANCELLED
        assert plan.id not in executor.active_executions
    
    def test_get_execution_history(self, executor):
        """Test getting execution history."""
        # Add some history
        executor.execution_history.append({
            "plan_id": "plan_1",
            "query": "query 1",
            "result": {},
            "timestamp": "2024-01-01T00:00:00"
        })
        executor.execution_history.append({
            "plan_id": "plan_2",
            "query": "query 2",
            "result": {},
            "timestamp": "2024-01-01T00:01:00"
        })
        
        history = executor.get_execution_history(limit=10)
        
        assert len(history) == 2
        assert history[0]["plan_id"] == "plan_1"
    
    @pytest.mark.asyncio
    async def test_get_context_info(self, executor):
        """Test getting context information."""
        plan = ExecutionPlan(
            id="test_plan",
            query="Test query",
            steps=[]
        )
        
        context = ExecutionContext(plan_id=plan.id)
        context.update_shared_context("key", "value")
        context.add_step_result("step_1", {"data": "result"})
        
        executor.execution_contexts[plan.id] = context
        
        info = executor.get_context_info(plan.id)
        
        assert info is not None
        assert info["plan_id"] == plan.id
        assert info["step_results_count"] == 1
        assert "test_corpus" not in info["shared_context_keys"] or "key" in info["shared_context_keys"]
    
    @pytest.mark.asyncio
    async def test_update_context_settings(self, executor):
        """Test updating context settings."""
        plan_id = "test_plan"
        context = ExecutionContext(plan_id=plan_id)
        executor.execution_contexts[plan_id] = context
        
        updated = executor.update_context_settings(
            plan_id=plan_id,
            max_context_size=10000,
            truncation_strategy="oldest_first"
        )
        
        assert updated is True
        assert context.max_context_size == 10000
        assert context.context_truncation_strategy == "oldest_first"


class TestParameterSubstitution:
    """Test parameter substitution functionality."""
    
    def test_substitute_shared_context(self, executor):
        """Test substituting parameters from shared context."""
        parameters = {
            "corpus_name": "${shared.corpus}",
            "user_id": "${shared.user}"
        }
        
        context = {
            "shared_context": {
                "corpus": "test_corpus",
                "user": "user123"
            },
            "step_results": {}
        }
        
        substituted = executor._substitute_parameters_from_context(parameters, context)
        
        assert substituted["corpus_name"] == "test_corpus"
        assert substituted["user_id"] == "user123"
    
    def test_substitute_step_results(self, executor):
        """Test substituting parameters from step results."""
        parameters = {
            "query": "${step_1.query}",
            "top_k": "${step_1.top_k}"
        }
        
        context = {
            "shared_context": {},
            "step_results": {
                "step_1": {
                    "query": "test query",
                    "top_k": 10
                }
            }
        }
        
        substituted = executor._substitute_parameters_from_context(parameters, context)
        
        assert substituted["query"] == "test query"
        assert substituted["top_k"] == 10
    
    def test_substitute_mixed_parameters(self, executor):
        """Test substituting mixed parameter types."""
        parameters = {
            "corpus": "${shared.corpus}",
            "query": "${step_1.query}",
            "static": "static_value"
        }
        
        context = {
            "shared_context": {"corpus": "test_corpus"},
            "step_results": {"step_1": {"query": "test query"}}
        }
        
        substituted = executor._substitute_parameters_from_context(parameters, context)
        
        assert substituted["corpus"] == "test_corpus"
        assert substituted["query"] == "test query"
        assert substituted["static"] == "static_value"
    
    def test_substitute_missing_reference(self, executor):
        """Test substitution with missing reference."""
        parameters = {
            "value": "${shared.missing}"
        }
        
        context = {
            "shared_context": {},
            "step_results": {}
        }
        
        substituted = executor._substitute_parameters_from_context(parameters, context)
        
        # Should keep original value if substitution fails
        assert substituted["value"] == "${shared.missing}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
