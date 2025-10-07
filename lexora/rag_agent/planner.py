"""Agent Planner for the Lexora Agentic RAG SDK.

This module implements the planning component of the agentic RAG system,
responsible for analyzing user queries, decomposing complex tasks, and
creating execution plans using available tools.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

from ..llm.base_llm import BaseLLM
from ..tools.base_tool import BaseTool
from ..tools import ToolRegistry
from ..exceptions import LexoraError, ErrorCode, create_tool_error
from ..utils.logging import get_logger


class PlanStepType(str, Enum):
    """Types of plan steps."""
    TOOL_EXECUTION = "tool_execution"
    INFORMATION_GATHERING = "information_gathering"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    DECISION = "decision"


class PlanStatus(str, Enum):
    """Status of a plan or plan step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class PlanStep:
    """Represents a single step in an execution plan."""
    id: str
    type: PlanStepType
    description: str
    tool_name: Optional[str] = None
    tool_parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    expected_output: Optional[str] = None
    status: PlanStatus = PlanStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    def to_dict(self) -> Dict[str, Any]:
        """Convert plan step to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "description": self.description,
            "tool_name": self.tool_name,
            "tool_parameters": self.tool_parameters,
            "dependencies": self.dependencies,
            "expected_output": self.expected_output,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class ExecutionPlan:
    """Represents a complete execution plan for a user query."""
    id: str
    query: str
    steps: List[PlanStep] = field(default_factory=list)
    status: PlanStatus = PlanStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert execution plan to dictionary."""
        return {
            "id": self.id,
            "query": self.query,
            "steps": [step.to_dict() for step in self.steps],
            "status": self.status.value,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    def get_step(self, step_id: str) -> Optional[PlanStep]:
        """Get a step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None
    
    def get_ready_steps(self) -> List[PlanStep]:
        """Get steps that are ready to execute (dependencies satisfied)."""
        ready_steps = []
        completed_step_ids = {step.id for step in self.steps if step.status == PlanStatus.COMPLETED}
        
        for step in self.steps:
            if step.status == PlanStatus.PENDING:
                # Check if all dependencies are completed
                if all(dep_id in completed_step_ids for dep_id in step.dependencies):
                    ready_steps.append(step)
        
        return ready_steps
    
    def is_complete(self) -> bool:
        """Check if all steps are completed."""
        return all(step.status in [PlanStatus.COMPLETED, PlanStatus.SKIPPED] for step in self.steps)
    
    def has_failed_steps(self) -> bool:
        """Check if any steps have failed."""
        return any(step.status == PlanStatus.FAILED for step in self.steps)


class AgentPlanner:
    """
    Agent planner responsible for analyzing queries and creating execution plans.
    
    The planner uses an LLM to understand user queries, decompose complex tasks,
    and create structured execution plans using available tools.
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        tool_registry: ToolRegistry,
        **kwargs
    ):
        """
        Initialize the agent planner.
        
        Args:
            llm: Language model for query analysis and planning
            tool_registry: Registry of available tools
            **kwargs: Additional configuration options
        """
        self.llm = llm
        self.tool_registry = tool_registry
        self.logger = get_logger(self.__class__.__name__)
        
        # Configuration
        self.max_plan_steps = kwargs.get('max_plan_steps', 20)
        self.planning_timeout = kwargs.get('planning_timeout', 60.0)
        self.enable_step_optimization = kwargs.get('enable_step_optimization', True)
        self.enable_parallel_planning = kwargs.get('enable_parallel_planning', True)
        
        # Planning templates and prompts
        self.planning_system_prompt = self._get_planning_system_prompt()
        self.step_analysis_prompt = self._get_step_analysis_prompt()
    
    async def create_plan(self, query: str, context: Optional[Dict[str, Any]] = None) -> ExecutionPlan:
        """
        Create an execution plan for the given query.
        
        Args:
            query: User query to create a plan for
            context: Optional context information
            
        Returns:
            ExecutionPlan: Generated execution plan
            
        Raises:
            LexoraError: If plan creation fails
        """
        try:
            self.logger.info(f"Creating execution plan for query: '{query[:100]}...'")
            
            # Generate unique plan ID
            plan_id = f"plan_{int(datetime.utcnow().timestamp())}_{hash(query) % 10000}"
            
            # Analyze query and available tools
            query_analysis = await self._analyze_query(query, context)
            available_tools = await self._get_available_tools_info()
            
            # Generate plan steps using LLM
            plan_steps = await self._generate_plan_steps(
                query, query_analysis, available_tools, context
            )
            
            # Optimize plan if enabled
            if self.enable_step_optimization:
                plan_steps = await self._optimize_plan_steps(plan_steps)
            
            # Create execution plan
            plan = ExecutionPlan(
                id=plan_id,
                query=query,
                steps=plan_steps,
                metadata={
                    "query_analysis": query_analysis,
                    "available_tools": len(available_tools),
                    "optimization_enabled": self.enable_step_optimization,
                    "context": context or {}
                }
            )
            
            self.logger.info(
                f"Created execution plan '{plan_id}' with {len(plan_steps)} steps"
            )
            
            return plan
            
        except Exception as e:
            raise create_tool_error(
                f"Failed to create execution plan: {str(e)}",
                "agent_planner",
                {"query": query, "error_type": type(e).__name__},
                ErrorCode.PLANNING_FAILED,
                e
            )
    
    async def update_plan(self, plan: ExecutionPlan, step_results: Dict[str, Any]) -> ExecutionPlan:
        """
        Update an execution plan based on step results.
        
        Args:
            plan: Current execution plan
            step_results: Results from executed steps
            
        Returns:
            ExecutionPlan: Updated execution plan
        """
        try:
            self.logger.info(f"Updating execution plan '{plan.id}' with step results")
            
            # Update step statuses and results
            for step_id, result in step_results.items():
                step = plan.get_step(step_id)
                if step:
                    step.result = result
                    step.status = PlanStatus.COMPLETED
                    step.updated_at = datetime.utcnow()
            
            # Check if plan needs modification based on results
            needs_replanning = await self._check_replanning_needed(plan, step_results)
            
            if needs_replanning:
                # Generate additional steps or modify existing ones
                additional_steps = await self._generate_adaptive_steps(plan, step_results)
                plan.steps.extend(additional_steps)
                
                self.logger.info(
                    f"Added {len(additional_steps)} adaptive steps to plan '{plan.id}'"
                )
            
            # Update overall plan status based on step states
            if plan.is_complete():
                plan.status = PlanStatus.COMPLETED
            elif plan.has_failed_steps():
                plan.status = PlanStatus.FAILED
            elif any(step.status == PlanStatus.IN_PROGRESS for step in plan.steps):
                plan.status = PlanStatus.IN_PROGRESS
            
            plan.updated_at = datetime.utcnow()
            return plan
            
        except Exception as e:
            self.logger.error(f"Failed to update plan: {e}")
            raise    
    async def _analyze_query(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the user query to understand intent and requirements.
        
        Args:
            query: User query to analyze
            context: Optional context information
            
        Returns:
            Dictionary containing query analysis results
        """
        analysis_prompt = f"""
        Analyze the following user query and provide structured analysis:
        
        Query: "{query}"
        Context: {context or 'None'}
        
        Please analyze:
        1. Query intent (information_retrieval, document_management, analysis, etc.)
        2. Required operations (search, create, update, delete, etc.)
        3. Entities mentioned (corpus names, document IDs, etc.)
        4. Complexity level (simple, moderate, complex)
        5. Expected output type (documents, statistics, confirmation, etc.)
        
        Respond in JSON format with these fields:
        {{
            "intent": "string",
            "operations": ["list of operations"],
            "entities": {{"type": "value"}},
            "complexity": "simple|moderate|complex",
            "output_type": "string",
            "confidence": 0.0-1.0
        }}
        """
        
        try:
            response = await self.llm.generate(
                prompt=analysis_prompt,
                max_tokens=500,
                temperature=0.1
            )
            
            # Parse JSON response
            import json
            analysis = json.loads(response)
            
            return analysis
            
        except Exception as e:
            self.logger.warning(f"Query analysis failed, using fallback: {e}")
            # Fallback analysis
            return {
                "intent": "information_retrieval",
                "operations": ["search"],
                "entities": {},
                "complexity": "moderate",
                "output_type": "documents",
                "confidence": 0.5
            }
    
    async def _get_available_tools_info(self) -> List[Dict[str, Any]]:
        """
        Get information about available tools.
        
        Returns:
            List of tool information dictionaries
        """
        tools_info = []
        
        for tool_name in self.tool_registry.list_tools():
            try:
                tool = self.tool_registry.get_tool(tool_name)
                schema = tool.get_schema()
                
                tools_info.append({
                    "name": tool_name,
                    "description": tool.description,
                    "parameters": schema.get("properties", {}),
                    "required": schema.get("required", []),
                    "version": tool.version
                })
            except Exception as e:
                self.logger.warning(f"Failed to get info for tool {tool_name}: {e}")
        
        return tools_info
    
    async def _generate_plan_steps(
        self,
        query: str,
        analysis: Dict[str, Any],
        available_tools: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]]
    ) -> List[PlanStep]:
        """
        Generate plan steps using LLM.
        
        Args:
            query: Original user query
            analysis: Query analysis results
            available_tools: Information about available tools
            context: Optional context
            
        Returns:
            List of generated plan steps
        """
        # Create planning prompt
        tools_description = "\n".join([
            f"- {tool['name']}: {tool['description']}"
            for tool in available_tools
        ])
        
        planning_prompt = f"""
        Create an execution plan for the following query using the available tools.
        
        Query: "{query}"
        Analysis: {analysis}
        Context: {context or 'None'}
        
        Available Tools:
        {tools_description}
        
        Create a step-by-step plan with the following format for each step:
        {{
            "id": "step_1",
            "type": "tool_execution|information_gathering|analysis|synthesis",
            "description": "What this step does",
            "tool_name": "tool_to_use" or null,
            "tool_parameters": {{"param": "value"}},
            "dependencies": ["step_ids_this_depends_on"],
            "expected_output": "What this step should produce"
        }}
        
        Guidelines:
        1. Break complex queries into logical steps
        2. Use appropriate tools for each operation
        3. Consider dependencies between steps
        4. Aim for 3-10 steps for most queries
        5. Include validation steps when appropriate
        
        Respond with a JSON array of steps.
        """
        
        try:
            response = await self.llm.generate(
                prompt=f"{self.planning_system_prompt}\n\n{planning_prompt}",
                max_tokens=2000,
                temperature=0.2
            )
            
            # Parse JSON response
            import json
            steps_data = json.loads(response)
            
            # Convert to PlanStep objects
            plan_steps = []
            for i, step_data in enumerate(steps_data):
                step = PlanStep(
                    id=step_data.get("id", f"step_{i+1}"),
                    type=PlanStepType(step_data.get("type", "tool_execution")),
                    description=step_data.get("description", ""),
                    tool_name=step_data.get("tool_name"),
                    tool_parameters=step_data.get("tool_parameters", {}),
                    dependencies=step_data.get("dependencies", []),
                    expected_output=step_data.get("expected_output")
                )
                plan_steps.append(step)
            
            return plan_steps[:self.max_plan_steps]  # Limit number of steps
            
        except Exception as e:
            self.logger.warning(f"LLM plan generation failed, using fallback: {e}")
            # Fallback: create simple plan based on analysis
            return self._create_fallback_plan(query, analysis)
    
    def _create_fallback_plan(self, query: str, analysis: Dict[str, Any]) -> List[PlanStep]:
        """
        Create a fallback plan when LLM planning fails.
        
        Args:
            query: Original user query
            analysis: Query analysis results
            
        Returns:
            List of fallback plan steps
        """
        steps = []
        
        # Simple fallback based on query intent
        intent = analysis.get("intent", "information_retrieval")
        
        if intent == "information_retrieval":
            # Simple search plan
            steps.append(PlanStep(
                id="step_1",
                type=PlanStepType.TOOL_EXECUTION,
                description="Search for relevant documents",
                tool_name="rag_query",
                tool_parameters={"query": query, "top_k": 10},
                expected_output="List of relevant documents"
            ))
        elif intent == "document_management":
            # Check what operation is needed
            operations = analysis.get("operations", [])
            if "create" in operations:
                steps.append(PlanStep(
                    id="step_1",
                    type=PlanStepType.TOOL_EXECUTION,
                    description="Create new corpus or add documents",
                    tool_name="create_corpus",
                    expected_output="Confirmation of creation"
                ))
        else:
            # Generic information gathering
            steps.append(PlanStep(
                id="step_1",
                type=PlanStepType.INFORMATION_GATHERING,
                description="Gather information to answer the query",
                expected_output="Relevant information"
            ))
        
        return steps
    
    async def _optimize_plan_steps(self, steps: List[PlanStep]) -> List[PlanStep]:
        """
        Optimize plan steps for better execution.
        
        Args:
            steps: Original plan steps
            
        Returns:
            Optimized plan steps
        """
        # Simple optimization: remove redundant steps and optimize dependencies
        optimized_steps = []
        seen_operations = set()
        
        for step in steps:
            # Create a signature for the step
            step_signature = f"{step.tool_name}_{hash(str(step.tool_parameters))}"
            
            # Skip if we've seen this exact operation
            if step_signature not in seen_operations:
                optimized_steps.append(step)
                seen_operations.add(step_signature)
            else:
                self.logger.info(f"Skipping redundant step: {step.description}")
        
        return optimized_steps
    
    async def _check_replanning_needed(self, plan: ExecutionPlan, step_results: Dict[str, Any]) -> bool:
        """
        Check if the plan needs to be modified based on step results.
        
        Args:
            plan: Current execution plan
            step_results: Results from executed steps
            
        Returns:
            True if replanning is needed
        """
        # Simple heuristic: replan if any step failed or returned unexpected results
        for step_id, result in step_results.items():
            step = plan.get_step(step_id)
            if step and step.status == PlanStatus.FAILED:
                return True
            
            # Check if result indicates need for additional steps
            if isinstance(result, dict) and result.get("requires_additional_steps"):
                return True
        
        return False
    
    async def _generate_adaptive_steps(self, plan: ExecutionPlan, step_results: Dict[str, Any]) -> List[PlanStep]:
        """
        Generate additional steps based on current results.
        
        Args:
            plan: Current execution plan
            step_results: Results from executed steps
            
        Returns:
            List of additional plan steps
        """
        # Simple adaptive planning - this could be enhanced with LLM
        additional_steps = []
        
        # Example: if search returned no results, try different search terms
        for step_id, result in step_results.items():
            step = plan.get_step(step_id)
            if step and step.tool_name == "rag_query":
                if isinstance(result, dict) and result.get("total_count", 0) == 0:
                    # No results found, try broader search
                    additional_steps.append(PlanStep(
                        id=f"adaptive_{len(plan.steps) + len(additional_steps) + 1}",
                        type=PlanStepType.TOOL_EXECUTION,
                        description="Retry search with broader terms",
                        tool_name="rag_query",
                        tool_parameters={
                            "query": plan.query,
                            "top_k": 20,
                            "min_score": 0.1  # Lower threshold
                        },
                        dependencies=[step_id],
                        expected_output="Broader search results"
                    ))
        
        return additional_steps
    
    def _get_planning_system_prompt(self) -> str:
        """Get the system prompt for planning."""
        return """
        You are an AI planning assistant for a RAG (Retrieval-Augmented Generation) system.
        Your role is to analyze user queries and create step-by-step execution plans using available tools.
        
        Key principles:
        1. Break complex queries into logical, sequential steps
        2. Use the most appropriate tools for each operation
        3. Consider dependencies between steps
        4. Include validation and error handling where appropriate
        5. Optimize for efficiency and accuracy
        
        Always respond with valid JSON and ensure all required fields are included.
        """
    
    def _get_step_analysis_prompt(self) -> str:
        """Get the prompt template for step analysis."""
        return """
        Analyze the following execution step and determine if it can be executed:
        
        Step: {step}
        Available Tools: {tools}
        Previous Results: {results}
        
        Determine:
        1. Is the step ready to execute?
        2. Are all dependencies satisfied?
        3. Are the tool parameters valid?
        4. Any potential issues or optimizations?
        """


# Convenience function for creating the planner
def create_agent_planner(
    llm: BaseLLM,
    tool_registry: ToolRegistry,
    **kwargs
) -> AgentPlanner:
    """
    Create an AgentPlanner instance.
    
    Args:
        llm: Language model instance
        tool_registry: Tool registry instance
        **kwargs: Additional configuration options
        
    Returns:
        Configured AgentPlanner instance
    """
    return AgentPlanner(
        llm=llm,
        tool_registry=tool_registry,
        **kwargs
    )