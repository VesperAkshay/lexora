"""Agent Executor for the Lexora Agentic RAG SDK.

This module implements the execution component of the agentic RAG system,
responsible for executing plans created by the planner, managing tool
invocations, and handling execution flow control.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, field

from .planner import ExecutionPlan, PlanStep, PlanStatus, PlanStepType
from ..tools.base_tool import BaseTool, ToolResult, ToolStatus
from ..tools import ToolRegistry
from ..exceptions import LexoraError, ErrorCode, create_tool_error
from ..utils.logging import get_logger


@dataclass
class ExecutionResult:
    """Result of executing a plan or step."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert execution result to dictionary."""
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ExecutionContext:
    """Context for plan execution."""
    plan_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    timeout: Optional[float] = None
    max_retries: int = 3
    parallel_execution: bool = True
    environment: Dict[str, Any] = field(default_factory=dict)
    callbacks: Dict[str, Callable] = field(default_factory=dict)
    
    # Context management settings
    max_context_size: int = 50000  # Maximum context size in characters
    context_truncation_strategy: str = "sliding_window"  # "sliding_window", "summarize", "oldest_first"
    preserve_recent_steps: int = 5  # Number of recent steps to always preserve
    
    # Execution context data
    shared_context: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)
    context_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert execution context to dictionary."""
        return {
            "plan_id": self.plan_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "parallel_execution": self.parallel_execution,
            "environment": self.environment,
            "max_context_size": self.max_context_size,
            "context_truncation_strategy": self.context_truncation_strategy,
            "preserve_recent_steps": self.preserve_recent_steps,
            "shared_context": self.shared_context,
            "step_results": self.step_results
        }
    
    def get_context_size(self) -> int:
        """Calculate current context size in characters."""
        import json
        context_str = json.dumps({
            "shared_context": self.shared_context,
            "step_results": self.step_results,
            "context_history": self.context_history
        })
        return len(context_str)
    
    def add_step_result(self, step_id: str, result: Any) -> None:
        """Add a step result to the context."""
        self.step_results[step_id] = result
        self.context_history.append({
            "step_id": step_id,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def update_shared_context(self, key: str, value: Any) -> None:
        """Update shared context with a key-value pair."""
        self.shared_context[key] = value
    
    def get_relevant_context(self, step_dependencies: List[str]) -> Dict[str, Any]:
        """Get relevant context for a step based on its dependencies."""
        relevant_context = {
            "shared_context": self.shared_context.copy(),
            "step_results": {}
        }
        
        # Include results from dependency steps
        for dep_id in step_dependencies:
            if dep_id in self.step_results:
                relevant_context["step_results"][dep_id] = self.step_results[dep_id]
        
        return relevant_context


class ContextManager:
    """Manages execution context size and truncation strategies."""
    
    def __init__(self, max_size: int = 50000, strategy: str = "sliding_window"):
        """
        Initialize context manager.
        
        Args:
            max_size: Maximum context size in characters
            strategy: Truncation strategy ("sliding_window", "summarize", "oldest_first")
        """
        self.max_size = max_size
        self.strategy = strategy
        self.logger = get_logger(self.__class__.__name__)
    
    def manage_context(self, context: ExecutionContext) -> ExecutionContext:
        """
        Manage context size and apply truncation if needed.
        
        Args:
            context: Execution context to manage
            
        Returns:
            Managed execution context
        """
        current_size = context.get_context_size()
        
        if current_size <= self.max_size:
            return context
        
        self.logger.info(f"Context size ({current_size}) exceeds limit ({self.max_size}), applying {self.strategy} strategy")
        
        if self.strategy == "sliding_window":
            return self._apply_sliding_window(context)
        elif self.strategy == "oldest_first":
            return self._apply_oldest_first(context)
        elif self.strategy == "summarize":
            return self._apply_summarization(context)
        else:
            self.logger.warning(f"Unknown strategy {self.strategy}, falling back to sliding_window")
            return self._apply_sliding_window(context)
    
    def _apply_sliding_window(self, context: ExecutionContext) -> ExecutionContext:
        """Apply sliding window truncation strategy."""
        # Keep the most recent steps up to preserve_recent_steps
        if len(context.context_history) > context.preserve_recent_steps:
            # Remove oldest entries while preserving recent ones
            entries_to_remove = len(context.context_history) - context.preserve_recent_steps
            removed_entries = context.context_history[:entries_to_remove]
            context.context_history = context.context_history[entries_to_remove:]
            
            # Remove corresponding step results
            for entry in removed_entries:
                step_id = entry.get("step_id")
                if step_id and step_id in context.step_results:
                    del context.step_results[step_id]
            
            self.logger.info(f"Removed {entries_to_remove} old context entries using sliding window")
        
        return context
    
    def _apply_oldest_first(self, context: ExecutionContext) -> ExecutionContext:
        """Apply oldest-first truncation strategy."""
        while context.get_context_size() > self.max_size and context.context_history:
            # Remove the oldest entry
            removed_entry = context.context_history.pop(0)
            step_id = removed_entry.get("step_id")
            if step_id and step_id in context.step_results:
                del context.step_results[step_id]
        
        self.logger.info("Applied oldest-first truncation strategy")
        return context
    
    def _apply_summarization(self, context: ExecutionContext) -> ExecutionContext:
        """Apply summarization strategy (placeholder for future LLM-based summarization)."""
        # For now, fall back to sliding window
        # In the future, this could use an LLM to summarize old context
        self.logger.info("Summarization strategy not yet implemented, falling back to sliding window")
        return self._apply_sliding_window(context)
    
    def get_context_summary(self, context: ExecutionContext) -> Dict[str, Any]:
        """Get a summary of the current context."""
        return {
            "total_size": context.get_context_size(),
            "max_size": self.max_size,
            "strategy": self.strategy,
            "step_count": len(context.step_results),
            "history_entries": len(context.context_history),
            "shared_context_keys": list(context.shared_context.keys())
        }


class AgentExecutor:
    """
    Agent executor responsible for executing plans and managing tool invocations.
    
    The executor takes execution plans from the planner and executes them step by step,
    handling dependencies, parallel execution, error recovery, and result aggregation.
    """
    
    def __init__(
        self,
        tool_registry: ToolRegistry,
        **kwargs
    ):
        """
        Initialize the agent executor.
        
        Args:
            tool_registry: Registry of available tools
            **kwargs: Additional configuration options
        """
        self.tool_registry = tool_registry
        self.logger = get_logger(self.__class__.__name__)
        
        # Configuration
        self.default_timeout = kwargs.get('default_timeout', 300.0)  # 5 minutes
        self.max_parallel_steps = kwargs.get('max_parallel_steps', 5)
        self.enable_step_retry = kwargs.get('enable_step_retry', True)
        self.enable_error_recovery = kwargs.get('enable_error_recovery', True)
        self.step_timeout = kwargs.get('step_timeout', 60.0)
        self.max_retry_attempts = kwargs.get('max_retry_attempts', 3)
        self.retry_delay = kwargs.get('retry_delay', 1.0)
        self.retry_backoff_factor = kwargs.get('retry_backoff_factor', 2.0)
        
        # Context management
        self.context_manager = ContextManager(
            max_size=kwargs.get('max_context_size', 50000),
            strategy=kwargs.get('context_truncation_strategy', 'sliding_window')
        )
        
        # Execution state
        self.active_executions: Dict[str, ExecutionPlan] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.execution_contexts: Dict[str, ExecutionContext] = {}
    
    async def execute_plan(
        self,
        plan: ExecutionPlan,
        context: Optional[ExecutionContext] = None
    ) -> ExecutionResult:
        """
        Execute a complete execution plan.
        
        Args:
            plan: Execution plan to execute
            context: Optional execution context
            
        Returns:
            ExecutionResult: Overall execution result
            
        Raises:
            LexoraError: If execution fails
        """
        if context is None:
            context = ExecutionContext(plan_id=plan.id)
        
        start_time = time.time()
        
        try:
            self.logger.info(
                f"Starting execution of plan '{plan.id}' with {len(plan.steps)} steps"
            )
            
            # Add to active executions
            self.active_executions[plan.id] = plan
            self.execution_contexts[plan.id] = context
            plan.status = PlanStatus.IN_PROGRESS
            
            # Initialize context with plan information
            context.update_shared_context("plan_id", plan.id)
            context.update_shared_context("query", plan.query)
            context.update_shared_context("total_steps", len(plan.steps))
            
            # Execute steps
            execution_results = await self._execute_plan_steps(plan, context)
            
            # Determine overall success
            overall_success = plan.is_complete() and not plan.has_failed_steps()
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Update plan status
            if overall_success:
                plan.status = PlanStatus.COMPLETED
            elif plan.has_failed_steps():
                plan.status = PlanStatus.FAILED
            else:
                plan.status = PlanStatus.CANCELLED
            
            # Create result
            result = ExecutionResult(
                success=overall_success,
                result={
                    "plan_id": plan.id,
                    "steps_executed": len([s for s in plan.steps if s.status == PlanStatus.COMPLETED]),
                    "steps_failed": len([s for s in plan.steps if s.status == PlanStatus.FAILED]),
                    "step_results": {step.id: step.result for step in plan.steps if step.result is not None},
                    "final_answer": self._synthesize_final_answer(plan, execution_results)
                },
                execution_time=execution_time,
                metadata={
                    "plan_metadata": plan.metadata,
                    "context": context.to_dict(),
                    "execution_summary": self._create_execution_summary(plan)
                }
            )
            
            # Add to history
            self.execution_history.append({
                "plan_id": plan.id,
                "query": plan.query,
                "result": result.to_dict(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Remove from active executions
            self.active_executions.pop(plan.id, None)
            self.execution_contexts.pop(plan.id, None)
            
            self.logger.info(
                f"Completed execution of plan '{plan.id}' in {execution_time:.2f}s "
                f"(success: {overall_success})"
            )
            
            return result
            
        except Exception as e:
            # Handle execution failure
            plan.status = PlanStatus.FAILED
            self.active_executions.pop(plan.id, None)
            self.execution_contexts.pop(plan.id, None)
            
            execution_time = time.time() - start_time
            
            error_result = ExecutionResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                metadata={"plan_id": plan.id, "error_type": type(e).__name__}
            )
            
            self.logger.error(f"Plan execution failed: {e}")
            
            return error_result
    
    async def _execute_plan_steps(
        self,
        plan: ExecutionPlan,
        context: ExecutionContext
    ) -> Dict[str, ExecutionResult]:
        """
        Execute all steps in a plan, respecting dependencies.
        
        Args:
            plan: Execution plan
            context: Execution context
            
        Returns:
            Dictionary mapping step IDs to execution results
        """
        execution_results = {}
        
        while not plan.is_complete() and not plan.has_failed_steps():
            # Get steps ready for execution
            ready_steps = plan.get_ready_steps()
            
            if not ready_steps:
                # No more steps can be executed
                break
            
            # Execute steps (parallel or sequential based on configuration)
            if context.parallel_execution and len(ready_steps) > 1:
                # Execute steps in parallel
                step_results = await self._execute_steps_parallel(
                    ready_steps[:self.max_parallel_steps],
                    context
                )
            else:
                # Execute steps sequentially
                step_results = await self._execute_steps_sequential(
                    ready_steps,
                    context
                )
            
            # Update results
            execution_results.update(step_results)
        
        return execution_results
    
    async def _execute_steps_parallel(
        self,
        steps: List[PlanStep],
        context: ExecutionContext
    ) -> Dict[str, ExecutionResult]:
        """
        Execute multiple steps in parallel.
        
        Args:
            steps: Steps to execute
            context: Execution context
            
        Returns:
            Dictionary mapping step IDs to execution results
        """
        self.logger.info(f"Executing {len(steps)} steps in parallel")
        
        # Create tasks for parallel execution
        tasks = {
            step.id: asyncio.create_task(
                self.execute_step(step, context)
            )
            for step in steps
        }
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        # Process results
        step_results = {}
        for step_id, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                step_results[step_id] = ExecutionResult(
                    success=False,
                    error=str(result),
                    metadata={"step_id": step_id, "error_type": type(result).__name__}
                )
            else:
                step_results[step_id] = result
        
        return step_results
    
    async def _execute_steps_sequential(
        self,
        steps: List[PlanStep],
        context: ExecutionContext
    ) -> Dict[str, ExecutionResult]:
        """
        Execute steps sequentially.
        
        Args:
            steps: Steps to execute
            context: Execution context
            
        Returns:
            Dictionary mapping step IDs to execution results
        """
        step_results = {}
        
        for step in steps:
            result = await self.execute_step(step, context)
            step_results[step.id] = result
            
            # Stop if step failed and error recovery is disabled
            if not result.success and not self.enable_error_recovery:
                break
        
        return step_results
    
    async def execute_step(
        self,
        step: PlanStep,
        context: ExecutionContext
    ) -> ExecutionResult:
        """
        Execute a single plan step with retry logic and context management.
        
        Args:
            step: Plan step to execute
            context: Execution context
            
        Returns:
            ExecutionResult: Step execution result
        """
        start_time = time.time()
        last_error = None
        
        # Manage context size before execution
        context = self.context_manager.manage_context(context)
        
        # Get relevant context for this step
        relevant_context = context.get_relevant_context(step.dependencies)
        
        for attempt in range(self.max_retry_attempts if self.enable_step_retry else 1):
            try:
                if attempt > 0:
                    delay = self.retry_delay * (self.retry_backoff_factor ** (attempt - 1))
                    self.logger.info(f"Retrying step '{step.id}' (attempt {attempt + 1}) after {delay:.1f}s delay")
                    await asyncio.sleep(delay)
                
                self.logger.info(f"Executing step '{step.id}': {step.description} (attempt {attempt + 1})")
                
                step.status = PlanStatus.IN_PROGRESS
                step.updated_at = datetime.now(timezone.utc)
                
                # Execute based on step type
                if step.type == PlanStepType.TOOL_EXECUTION:
                    result = await self._execute_tool_step_with_context(step, context, relevant_context)
                elif step.type == PlanStepType.INFORMATION_GATHERING:
                    result = await self._execute_information_gathering_step(step, context, relevant_context)
                elif step.type == PlanStepType.ANALYSIS:
                    result = await self._execute_analysis_step(step, context, relevant_context)
                elif step.type == PlanStepType.SYNTHESIS:
                    result = await self._execute_synthesis_step(step, context, relevant_context)
                elif step.type == PlanStepType.VALIDATION:
                    result = await self._execute_validation_step(step, context, relevant_context)
                else:
                    raise ValueError(f"Unknown step type: {step.type}")
                
                # If successful, update context and break retry loop
                if result.success:
                    execution_time = time.time() - start_time
                    step.execution_time = execution_time
                    step.result = result.result
                    step.status = PlanStatus.COMPLETED
                    step.updated_at = datetime.now(timezone.utc)
                    
                    # Add result to context
                    context.add_step_result(step.id, result.result)
                    
                    # Update shared context with any relevant information
                    if isinstance(result.result, dict):
                        for key, value in result.result.items():
                            if key.startswith("shared_"):
                                context.update_shared_context(key[7:], value)  # Remove "shared_" prefix
                    
                    self.logger.info(
                        f"Step '{step.id}' completed successfully in {execution_time:.2f}s"
                    )
                    
                    return result
                else:
                    # Step failed, but we might retry
                    last_error = result.error
                    if not self._should_retry_step(step, result, attempt):
                        break
                    
            except Exception as e:
                last_error = str(e)
                self.logger.error(f"Step '{step.id}' failed with exception (attempt {attempt + 1}): {e}")
                
                if not self._should_retry_step(step, None, attempt):
                    break
        
        # All retry attempts failed
        execution_time = time.time() - start_time
        step.status = PlanStatus.FAILED
        step.error = last_error
        step.execution_time = execution_time
        step.updated_at = datetime.now(timezone.utc)
        
        self.logger.error(f"Step '{step.id}' failed after {self.max_retry_attempts} attempts: {last_error}")
        
        return ExecutionResult(
            success=False,
            error=last_error or "Step execution failed",
            execution_time=execution_time,
            metadata={
                "step_id": step.id,
                "attempts": self.max_retry_attempts if self.enable_step_retry else 1,
                "final_error": last_error
            }
        )
    
    def _should_retry_step(self, step: PlanStep, result: Optional[ExecutionResult], attempt: int) -> bool:
        """
        Determine if a step should be retried.
        
        Args:
            step: The step that failed
            result: The execution result (if any)
            attempt: Current attempt number (0-based)
            
        Returns:
            True if the step should be retried
        """
        if not self.enable_step_retry:
            return False
        
        if attempt >= self.max_retry_attempts - 1:
            return False
        
        # Don't retry validation errors or parameter errors
        if result and result.error:
            if "parameter" in result.error.lower() or "validation" in result.error.lower():
                return False
        
        # Don't retry certain step types
        if step.type in [PlanStepType.VALIDATION]:
            return False
        
        return True
    
    async def _execute_tool_step_with_context(
        self,
        step: PlanStep,
        context: ExecutionContext,
        relevant_context: Dict[str, Any]
    ) -> ExecutionResult:
        """
        Execute a tool execution step.
        
        Args:
            step: Tool execution step
            context: Execution context
            relevant_context: Relevant context for this step
            
        Returns:
            ExecutionResult: Tool execution result
        """
        if not step.tool_name:
            return ExecutionResult(
                success=False,
                error="Tool name not specified for tool execution step"
            )
        
        try:
            # Get tool from registry
            tool = self.tool_registry.get_tool(step.tool_name)
            
            # Prepare tool parameters
            tool_parameters = step.tool_parameters.copy()
            
            # Substitute parameters from context and previous results
            tool_parameters = self._substitute_parameters_from_context(
                tool_parameters, relevant_context
            )
            
            # Execute tool with timeout
            if context.timeout:
                tool_result = await asyncio.wait_for(
                    tool.run(**tool_parameters),
                    timeout=min(context.timeout, self.step_timeout)
                )
            else:
                tool_result = await tool.run(**tool_parameters)
            
            # Convert tool result to execution result
            if tool_result.status == ToolStatus.SUCCESS:
                return ExecutionResult(
                    success=True,
                    result=tool_result.data,
                    metadata={
                        "tool_name": step.tool_name,
                        "tool_parameters": tool_parameters,
                        "tool_metadata": getattr(tool_result, 'metadata', {})
                    }
                )
            else:
                return ExecutionResult(
                    success=False,
                    error=tool_result.error or "Tool execution failed",
                    metadata={
                        "tool_name": step.tool_name,
                        "tool_parameters": tool_parameters
                    }
                )
                
        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False,
                error=f"Tool execution timed out after {self.step_timeout}s",
                metadata={"tool_name": step.tool_name}
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Tool execution failed: {str(e)}",
                metadata={
                    "tool_name": step.tool_name,
                    "error_type": type(e).__name__
                }
            )
    
    def _substitute_parameters_from_context(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Substitute parameter values from execution context.
        
        Args:
            parameters: Original parameters
            context: Execution context with shared_context and step_results
            
        Returns:
            Parameters with substituted values
        """
        substituted = parameters.copy()
        shared_context = context.get("shared_context", {})
        step_results = context.get("step_results", {})
        
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("${"):
                # Parameter substitution: ${step_id.field} or ${shared.field}
                try:
                    # Extract reference from ${reference}
                    ref = value[2:-1]  # Remove ${ and }
                    
                    if ref.startswith("shared."):
                        # Shared context reference: ${shared.field}
                        field = ref[7:]  # Remove "shared."
                        if field in shared_context:
                            substituted[key] = shared_context[field]
                    elif "." in ref:
                        # Step result reference: ${step_id.field}
                        step_id, field = ref.split(".", 1)
                        if step_id in step_results:
                            result_data = step_results[step_id]
                            if isinstance(result_data, dict) and field in result_data:
                                substituted[key] = result_data[field]
                    else:
                        # Direct step result reference: ${step_id}
                        if ref in step_results:
                            substituted[key] = step_results[ref]
                        elif ref in shared_context:
                            substituted[key] = shared_context[ref]
                            
                except Exception as e:
                    self.logger.warning(f"Parameter substitution failed for {key}: {e}")
        
        return substituted
    
    async def _execute_information_gathering_step(
        self,
        step: PlanStep,
        context: ExecutionContext,
        relevant_context: Dict[str, Any]
    ) -> ExecutionResult:
        """
        Execute an information gathering step.
        
        Args:
            step: Information gathering step
            context: Execution context
            relevant_context: Relevant context for this step
            
        Returns:
            ExecutionResult: Information gathering result
        """
        # Aggregate information from context
        gathered_info = {
            "step_description": step.description,
            "shared_context": relevant_context.get("shared_context", {}),
            "step_results": relevant_context.get("step_results", {}),
            "context_summary": self.context_manager.get_context_summary(context),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return ExecutionResult(
            success=True,
            result=gathered_info,
            metadata={"step_type": "information_gathering"}
        )
    
    async def _execute_analysis_step(
        self,
        step: PlanStep,
        context: ExecutionContext,
        relevant_context: Dict[str, Any]
    ) -> ExecutionResult:
        """
        Execute an analysis step.
        
        Args:
            step: Analysis step
            context: Execution context
            relevant_context: Relevant context for this step
            
        Returns:
            ExecutionResult: Analysis result
        """
        # Analyze context and results
        step_results = relevant_context.get("step_results", {})
        shared_context = relevant_context.get("shared_context", {})
        
        analysis = {
            "step_description": step.description,
            "results_analyzed": len(step_results),
            "shared_context_keys": list(shared_context.keys()),
            "analysis_summary": f"Analyzed {len(step_results)} step results and {len(shared_context)} shared context items",
            "context_size": context.get_context_size(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return ExecutionResult(
            success=True,
            result=analysis,
            metadata={"step_type": "analysis"}
        )
    
    async def _execute_synthesis_step(
        self,
        step: PlanStep,
        context: ExecutionContext,
        relevant_context: Dict[str, Any]
    ) -> ExecutionResult:
        """
        Execute a synthesis step.
        
        Args:
            step: Synthesis step
            context: Execution context
            relevant_context: Relevant context for this step
            
        Returns:
            ExecutionResult: Synthesis result
        """
        # Synthesize data from context
        step_results = relevant_context.get("step_results", {})
        shared_context = relevant_context.get("shared_context", {})
        
        # Create a synthesized view of all available data
        synthesized_data = {
            "step_results": step_results,
            "shared_context": shared_context,
            "synthesis_metadata": {
                "total_steps": len(step_results),
                "context_keys": list(shared_context.keys()),
                "synthesis_timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
        
        synthesis = {
            "step_description": step.description,
            "synthesized_data": synthesized_data,
            "synthesis_summary": f"Synthesized data from {len(step_results)} steps and {len(shared_context)} context items",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return ExecutionResult(
            success=True,
            result=synthesis,
            metadata={"step_type": "synthesis"}
        )
    
    async def _execute_validation_step(
        self,
        step: PlanStep,
        context: ExecutionContext,
        relevant_context: Dict[str, Any]
    ) -> ExecutionResult:
        """
        Execute a validation step.
        
        Args:
            step: Validation step
            context: Execution context
            relevant_context: Relevant context for this step
            
        Returns:
            ExecutionResult: Validation result
        """
        # Validate context and results
        step_results = relevant_context.get("step_results", {})
        shared_context = relevant_context.get("shared_context", {})
        validation_passed = True
        validation_messages = []
        
        # Check if we have required dependencies
        for dep_id in step.dependencies:
            if dep_id not in step_results:
                validation_passed = False
                validation_messages.append(f"Missing required dependency: {dep_id}")
        
        # Check context size
        context_size = context.get_context_size()
        if context_size > context.max_context_size:
            validation_messages.append(f"Context size ({context_size}) exceeds limit ({context.max_context_size})")
        
        # Check if we have any results to validate
        if not step_results and not shared_context:
            validation_passed = False
            validation_messages.append("No context or results to validate")
        
        validation = {
            "step_description": step.description,
            "validation_passed": validation_passed,
            "validation_messages": validation_messages,
            "context_size": context_size,
            "dependencies_checked": step.dependencies,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return ExecutionResult(
            success=validation_passed,
            result=validation,
            metadata={"step_type": "validation"}
        )
    
    def _synthesize_final_answer(
        self,
        plan: ExecutionPlan,
        execution_results: Dict[str, ExecutionResult]
    ) -> str:
        """
        Synthesize a final answer from execution results.
        
        Args:
            plan: Executed plan
            execution_results: Results from all steps
            
        Returns:
            Synthesized final answer
        """
        # Simple synthesis - this could be enhanced with LLM
        successful_results = [
            result.result for result in execution_results.values()
            if result.success and result.result is not None
        ]
        
        if not successful_results:
            return "No results were obtained from the execution."
        
        # Try to find the most relevant result
        for result in successful_results:
            if isinstance(result, dict):
                # Look for common result patterns
                if "results" in result and "message" in result:
                    return result["message"]
                elif "final_answer" in result:
                    return str(result["final_answer"])
                elif "message" in result:
                    return result["message"]
        
        return f"Execution completed successfully with {len(successful_results)} results."
    
    def _create_execution_summary(
        self,
        plan: ExecutionPlan
    ) -> Dict[str, Any]:
        """
        Create a summary of plan execution.
        
        Args:
            plan: Executed plan
            
        Returns:
            Execution summary
        """
        return {
            "total_steps": len(plan.steps),
            "completed_steps": len([s for s in plan.steps if s.status == PlanStatus.COMPLETED]),
            "failed_steps": len([s for s in plan.steps if s.status == PlanStatus.FAILED]),
            "skipped_steps": len([s for s in plan.steps if s.status == PlanStatus.SKIPPED]),
            "total_execution_time": sum(s.execution_time or 0 for s in plan.steps),
            "plan_status": plan.status.value,
            "step_details": [
                {
                    "id": step.id,
                    "description": step.description,
                    "status": step.status.value,
                    "execution_time": step.execution_time,
                    "error": step.error
                }
                for step in plan.steps
            ]
        }
    
    def get_execution_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current execution status of a plan.
        
        Args:
            plan_id: Plan ID to check
            
        Returns:
            Execution status information or None if not found
        """
        if plan_id in self.active_executions:
            plan = self.active_executions[plan_id]
            return {
                "plan_id": plan_id,
                "status": plan.status.value,
                "progress": {
                    "completed_steps": len([s for s in plan.steps if s.status == PlanStatus.COMPLETED]),
                    "total_steps": len(plan.steps),
                    "current_step": next(
                        (s.description for s in plan.steps if s.status == PlanStatus.IN_PROGRESS),
                        None
                    )
                },
                "updated_at": plan.updated_at.isoformat()
            }
        return None
    
    def cancel_execution(self, plan_id: str) -> bool:
        """
        Cancel an active execution.
        
        Args:
            plan_id: Plan ID to cancel
            
        Returns:
            True if cancellation was successful
        """
        if plan_id in self.active_executions:
            plan = self.active_executions[plan_id]
            plan.status = PlanStatus.CANCELLED
            
            # Cancel any pending steps
            for step in plan.steps:
                if step.status == PlanStatus.PENDING:
                    step.status = PlanStatus.CANCELLED
            
            self.active_executions.pop(plan_id, None)
            self.execution_contexts.pop(plan_id, None)
            self.logger.info(f"Cancelled execution of plan '{plan_id}'")
            return True
        
        return False
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent execution history.
        
        Args:
            limit: Maximum number of executions to return
            
        Returns:
            List of recent execution records
        """
        return self.execution_history[-limit:] if self.execution_history else []
    
    def get_context_info(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """
        Get context information for a specific plan.
        
        Args:
            plan_id: Plan ID to get context for
            
        Returns:
            Context information or None if not found
        """
        if plan_id in self.execution_contexts:
            context = self.execution_contexts[plan_id]
            return {
                "plan_id": plan_id,
                "context_size": context.get_context_size(),
                "max_context_size": context.max_context_size,
                "truncation_strategy": context.context_truncation_strategy,
                "step_results_count": len(context.step_results),
                "shared_context_keys": list(context.shared_context.keys()),
                "context_history_entries": len(context.context_history),
                "context_summary": self.context_manager.get_context_summary(context)
            }
        return None
    
    def update_context_settings(
        self,
        plan_id: str,
        max_context_size: Optional[int] = None,
        truncation_strategy: Optional[str] = None
    ) -> bool:
        """
        Update context management settings for a specific plan.
        
        Args:
            plan_id: Plan ID to update
            max_context_size: New maximum context size
            truncation_strategy: New truncation strategy
            
        Returns:
            True if settings were updated successfully
        """
        if plan_id in self.execution_contexts:
            context = self.execution_contexts[plan_id]
            
            if max_context_size is not None:
                context.max_context_size = max_context_size
            
            if truncation_strategy is not None:
                context.context_truncation_strategy = truncation_strategy
            
            self.logger.info(f"Updated context settings for plan '{plan_id}'")
            return True
        
        return False

# Convenience function for creating the executor
def create_agent_executor(
    tool_registry: ToolRegistry,
    **kwargs
) -> AgentExecutor:
    """
    Create an AgentExecutor instance.
    
    Args:
        tool_registry: Tool registry instance
        **kwargs: Additional configuration options
        
    Returns:
        Configured AgentExecutor instance
    """
    return AgentExecutor(
        tool_registry=tool_registry,
        **kwargs
    )