"""Reasoning Engine for the Lexora Agentic RAG SDK.

This module implements the reasoning component of the agentic RAG system,
responsible for synthesizing outputs from multiple tool executions, generating
coherent responses, and providing source attribution.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from .planner import ExecutionPlan, PlanStep, PlanStatus
from .executor import ExecutionResult, ExecutionContext
from ..llm.base_llm import BaseLLM
from ..exceptions import LexoraError, ErrorCode, create_tool_error
from ..utils.logging import get_logger


class ReasoningStrategy(str, Enum):
    """Strategies for reasoning about tool outputs."""
    DIRECT = "direct"  # Use the most relevant result directly
    SYNTHESIS = "synthesis"  # Synthesize multiple results
    CHAIN_OF_THOUGHT = "chain_of_thought"  # Step-by-step reasoning
    SUMMARIZATION = "summarization"  # Summarize all results


class ConfidenceLevel(str, Enum):
    """Confidence levels for generated responses."""
    VERY_LOW = "very_low"  # < 0.3
    LOW = "low"  # 0.3 - 0.5
    MEDIUM = "medium"  # 0.5 - 0.7
    HIGH = "high"  # 0.7 - 0.9
    VERY_HIGH = "very_high"  # > 0.9


@dataclass
class SourceAttribution:
    """Attribution information for a source used in reasoning."""
    source_type: str  # "tool_result", "context", "inference"
    source_id: str  # Step ID or context key
    content: str  # Relevant content from the source
    relevance_score: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))    
    def to_dict(self) -> Dict[str, Any]:
        """Convert source attribution to dictionary."""
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "content": self.content,
            "relevance_score": self.relevance_score,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ReasoningResult:
    """Result of reasoning process."""
    answer: str
    confidence: float
    confidence_level: ConfidenceLevel
    sources: List[SourceAttribution] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert reasoning result to dictionary."""
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "sources": [source.to_dict() for source in self.sources],
            "reasoning_chain": self.reasoning_chain,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class ReasoningEngine:
    """
    Reasoning engine responsible for synthesizing outputs and generating responses.
    
    The reasoning engine takes execution results from the executor and synthesizes
    them into coherent responses with source attribution and confidence scoring.
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        **kwargs
    ):
        """
        Initialize the reasoning engine.
        
        Args:
            llm: Language model for response generation
            **kwargs: Additional configuration options
        """
        self.llm = llm
        self.logger = get_logger(self.__class__.__name__)
        
        # Configuration
        self.default_strategy = kwargs.get('default_strategy', ReasoningStrategy.SYNTHESIS)
        self.min_confidence_threshold = kwargs.get('min_confidence_threshold', 0.3)
        self.max_sources = kwargs.get('max_sources', 10)
        self.enable_chain_of_thought = kwargs.get('enable_chain_of_thought', True)
        self.attribution_enabled = kwargs.get('attribution_enabled', True)
        
        # Reasoning templates
        self.synthesis_prompt_template = self._get_synthesis_prompt_template()
        self.chain_of_thought_template = self._get_chain_of_thought_template()
    
    async def generate_response(
        self,
        query: str,
        plan: ExecutionPlan,
        execution_result: ExecutionResult,
        context: Optional[ExecutionContext] = None,
        strategy: Optional[ReasoningStrategy] = None
    ) -> ReasoningResult:
        """
        Generate a response from execution results.
        
        Args:
            query: Original user query
            plan: Execution plan that was executed
            execution_result: Result from plan execution
            context: Optional execution context
            strategy: Reasoning strategy to use
            
        Returns:
            ReasoningResult: Generated response with attribution
            
        Raises:
            LexoraError: If response generation fails
        """
        try:
            self.logger.info(f"Generating response for query: '{query[:100]}...'")
            
            strategy = strategy or self.default_strategy
            
            # Extract relevant information from execution results
            step_results = execution_result.result.get("step_results", {})
            
            # Generate response based on strategy
            if strategy == ReasoningStrategy.DIRECT:
                reasoning_result = await self._generate_direct_response(
                    query, step_results, context
                )
            elif strategy == ReasoningStrategy.SYNTHESIS:
                reasoning_result = await self._generate_synthesis_response(
                    query, step_results, context
                )
            elif strategy == ReasoningStrategy.CHAIN_OF_THOUGHT:
                reasoning_result = await self._generate_chain_of_thought_response(
                    query, plan, step_results, context
                )
            elif strategy == ReasoningStrategy.SUMMARIZATION:
                reasoning_result = await self._generate_summarization_response(
                    query, step_results, context
                )
            else:
                raise ValueError(f"Unknown reasoning strategy: {strategy}")
            
            # Add metadata
            reasoning_result.metadata.update({
                "query": query,
                "strategy": strategy.value,
                "plan_id": plan.id,
                "execution_success": execution_result.success,
                "execution_time": execution_result.execution_time
            })
            
            self.logger.info(
                f"Generated response with confidence {reasoning_result.confidence:.2f} "
                f"({reasoning_result.confidence_level.value})"
            )
            
            return reasoning_result
            
        except Exception as e:
            raise create_tool_error(
                f"Failed to generate response: {str(e)}",
                "reasoning_engine",
                {"query": query, "error_type": type(e).__name__},
                ErrorCode.REASONING_FAILED,
                e
            )
    
    async def synthesize_multi_step_results(
        self,
        query: str,
        step_results: Dict[str, Any],
        reasoning_chain: Optional[List[str]] = None
    ) -> ReasoningResult:
        """
        Synthesize results from multiple execution steps.
        
        Args:
            query: Original user query
            step_results: Results from all executed steps
            reasoning_chain: Optional existing reasoning chain
            
        Returns:
            ReasoningResult: Synthesized response
        """
        try:
            self.logger.info(f"Synthesizing {len(step_results)} step results")
            
            # Build reasoning chain if not provided
            if reasoning_chain is None:
                reasoning_chain = self._build_reasoning_chain(step_results)
            
            # Extract sources
            sources = self._extract_sources(step_results)
            
            # Generate synthesis prompt
            synthesis_prompt = self._create_synthesis_prompt(
                query, step_results, reasoning_chain
            )
            
            # Generate response using LLM
            response = await self.llm.generate(
                prompt=synthesis_prompt,
                max_tokens=1000,
                temperature=0.3
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(step_results, sources)
            confidence_level = self._get_confidence_level(confidence)
            
            return ReasoningResult(
                answer=response,
                confidence=confidence,
                confidence_level=confidence_level,
                sources=sources[:self.max_sources],
                reasoning_chain=reasoning_chain,
                metadata={
                    "step_count": len(step_results),
                    "source_count": len(sources)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to synthesize results: {e}")
            raise
    
    async def _generate_direct_response(
        self,
        query: str,
        step_results: Dict[str, Any],
        context: Optional[ExecutionContext]
    ) -> ReasoningResult:
        """Generate a direct response from the most relevant result."""
        # Find the most relevant result
        most_relevant = self._find_most_relevant_result(step_results)
        
        if not most_relevant:
            return ReasoningResult(
                answer="No relevant results found.",
                confidence=0.0,
                confidence_level=ConfidenceLevel.VERY_LOW,
                metadata={"strategy": "direct", "results_found": False}
            )
        
        step_id, result = most_relevant
        
        # Extract answer from result
        answer = self._extract_answer_from_result(result)
        
        # Create source attribution
        source = SourceAttribution(
            source_type="tool_result",
            source_id=step_id,
            content=str(result)[:500],
            relevance_score=1.0
        )
        
        return ReasoningResult(
            answer=answer,
            confidence=0.8,
            confidence_level=ConfidenceLevel.HIGH,
            sources=[source],
            reasoning_chain=[f"Used direct result from {step_id}"],
            metadata={"strategy": "direct"}
        )
    
    async def _generate_synthesis_response(
        self,
        query: str,
        step_results: Dict[str, Any],
        context: Optional[ExecutionContext]
    ) -> ReasoningResult:
        """Generate a synthesized response from multiple results."""
        if not step_results:
            return ReasoningResult(
                answer="No results available to synthesize.",
                confidence=0.0,
                confidence_level=ConfidenceLevel.VERY_LOW,
                metadata={"strategy": "synthesis", "results_found": False}
            )
        
        # Build reasoning chain
        reasoning_chain = self._build_reasoning_chain(step_results)
        
        # Extract sources
        sources = self._extract_sources(step_results)
        
        # Create synthesis prompt
        synthesis_prompt = self._create_synthesis_prompt(
            query, step_results, reasoning_chain
        )
        
        # Generate response
        try:
            response = await self.llm.generate(
                prompt=synthesis_prompt,
                max_tokens=1000,
                temperature=0.3
            )
        except Exception as e:
            self.logger.warning(f"LLM synthesis failed, using fallback: {e}")
            response = self._create_fallback_synthesis(step_results)
        
        # Calculate confidence
        confidence = self._calculate_confidence(step_results, sources)
        confidence_level = self._get_confidence_level(confidence)
        
        return ReasoningResult(
            answer=response,
            confidence=confidence,
            confidence_level=confidence_level,
            sources=sources[:self.max_sources],
            reasoning_chain=reasoning_chain,
            metadata={"strategy": "synthesis", "step_count": len(step_results)}
        )
    
    async def _generate_chain_of_thought_response(
        self,
        query: str,
        plan: ExecutionPlan,
        step_results: Dict[str, Any],
        context: Optional[ExecutionContext]
    ) -> ReasoningResult:
        """Generate a response using chain-of-thought reasoning."""
        if not self.enable_chain_of_thought:
            # Fall back to synthesis
            return await self._generate_synthesis_response(query, step_results, context)
        
        # Build detailed reasoning chain from plan steps
        reasoning_chain = []
        for step in plan.steps:
            if step.status == PlanStatus.COMPLETED and step.id in step_results:
                reasoning_chain.append(
                    f"Step {step.id}: {step.description} -> {self._summarize_result(step_results[step.id])}"
                )
        
        # Create chain-of-thought prompt
        cot_prompt = self._create_chain_of_thought_prompt(
            query, reasoning_chain, step_results
        )
        
        # Generate response
        try:
            response = await self.llm.generate(
                prompt=cot_prompt,
                max_tokens=1500,
                temperature=0.2
            )
        except Exception as e:
            self.logger.warning(f"Chain-of-thought generation failed, using fallback: {e}")
            response = self._create_fallback_synthesis(step_results)
        
        # Extract sources
        sources = self._extract_sources(step_results)
        
        # Calculate confidence
        confidence = self._calculate_confidence(step_results, sources)
        confidence_level = self._get_confidence_level(confidence)
        
        return ReasoningResult(
            answer=response,
            confidence=confidence,
            confidence_level=confidence_level,
            sources=sources[:self.max_sources],
            reasoning_chain=reasoning_chain,
            metadata={"strategy": "chain_of_thought", "step_count": len(reasoning_chain)}
        )
    
    async def _generate_summarization_response(
        self,
        query: str,
        step_results: Dict[str, Any],
        context: Optional[ExecutionContext]
    ) -> ReasoningResult:
        """Generate a summarized response from all results."""
        if not step_results:
            return ReasoningResult(
                answer="No results available to summarize.",
                confidence=0.0,
                confidence_level=ConfidenceLevel.VERY_LOW,
                metadata={"strategy": "summarization", "results_found": False}
            )
        
        # Collect all result content
        all_content = []
        for step_id, result in step_results.items():
            content = self._extract_content_from_result(result)
            if content:
                all_content.append(f"From {step_id}: {content}")
        
        # Create summarization prompt
        summary_prompt = f"""
        Summarize the following information to answer the query: "{query}"
        
        Information:
        {chr(10).join(all_content)}
        
        Provide a concise, accurate summary that directly answers the query.
        """
        
        # Generate summary
        try:
            response = await self.llm.generate(
                prompt=summary_prompt,
                max_tokens=800,
                temperature=0.3
            )
        except Exception as e:
            self.logger.warning(f"Summarization failed, using fallback: {e}")
            response = f"Summary of {len(step_results)} results: " + "; ".join(all_content[:3])
        
        # Extract sources
        sources = self._extract_sources(step_results)
        
        # Calculate confidence
        confidence = self._calculate_confidence(step_results, sources)
        confidence_level = self._get_confidence_level(confidence)
        
        return ReasoningResult(
            answer=response,
            confidence=confidence,
            confidence_level=confidence_level,
            sources=sources[:self.max_sources],
            reasoning_chain=[f"Summarized {len(step_results)} results"],
            metadata={"strategy": "summarization", "content_items": len(all_content)}
        )

    
    def _find_most_relevant_result(self, step_results: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
        """Find the most relevant result from step results."""
        if not step_results:
            return None
        
        # Simple heuristic: prefer results with more content
        best_result = None
        best_score = 0
        
        for step_id, result in step_results.items():
            score = len(str(result))
            if score > best_score:
                best_score = score
                best_result = (step_id, result)
        
        return best_result
    
    def _extract_answer_from_result(self, result: Any) -> str:
        """Extract an answer from a result."""
        if isinstance(result, dict):
            # Look for common answer fields
            for key in ["answer", "result", "message", "content", "data"]:
                if key in result:
                    return str(result[key])
            # Return string representation
            return str(result)
        return str(result)
    
    def _extract_content_from_result(self, result: Any) -> str:
        """Extract content from a result for summarization."""
        if isinstance(result, dict):
            # Extract relevant content fields
            content_parts = []
            for key in ["content", "data", "result", "message", "answer"]:
                if key in result and result[key]:
                    content_parts.append(str(result[key]))
            return " ".join(content_parts) if content_parts else str(result)
        return str(result)
    
    def _build_reasoning_chain(self, step_results: Dict[str, Any]) -> List[str]:
        """Build a reasoning chain from step results."""
        chain = []
        
        for step_id, result in step_results.items():
            summary = self._summarize_result(result)
            chain.append(f"{step_id}: {summary}")
        
        return chain
    
    def _summarize_result(self, result: Any) -> str:
        """Create a brief summary of a result."""
        result_str = str(result)
        if len(result_str) > 100:
            return result_str[:97] + "..."
        return result_str
    
    def _extract_sources(self, step_results: Dict[str, Any]) -> List[SourceAttribution]:
        """Extract source attributions from step results."""
        sources = []
        
        for step_id, result in step_results.items():
            content = self._extract_content_from_result(result)
            
            # Calculate relevance score (simple heuristic based on content length)
            relevance_score = min(1.0, len(content) / 1000.0)
            
            source = SourceAttribution(
                source_type="tool_result",
                source_id=step_id,
                content=content[:500],  # Limit content length
                relevance_score=relevance_score
            )
            sources.append(source)
        
        # Sort by relevance score
        sources.sort(key=lambda s: s.relevance_score, reverse=True)
        
        return sources
    
    def _calculate_confidence(
        self,
        step_results: Dict[str, Any],
        sources: List[SourceAttribution]
    ) -> float:
        """Calculate confidence score for the response."""
        if not step_results:
            return 0.0
        
        # Base confidence on number of results and source quality
        result_count_factor = min(1.0, len(step_results) / 5.0)  # More results = higher confidence
        
        # Average source relevance
        if sources:
            avg_relevance = sum(s.relevance_score for s in sources) / len(sources)
        else:
            avg_relevance = 0.5
        
        # Combine factors
        confidence = (result_count_factor * 0.4) + (avg_relevance * 0.6)
        
        return min(1.0, max(0.0, confidence))
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to confidence level."""
        if confidence < 0.3:
            return ConfidenceLevel.VERY_LOW
        elif confidence < 0.5:
            return ConfidenceLevel.LOW
        elif confidence < 0.7:
            return ConfidenceLevel.MEDIUM
        elif confidence < 0.9:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH
    
    def _create_synthesis_prompt(
        self,
        query: str,
        step_results: Dict[str, Any],
        reasoning_chain: List[str]
    ) -> str:
        """Create a prompt for synthesizing results."""
        results_summary = "\n".join([
            f"- {step_id}: {self._summarize_result(result)}"
            for step_id, result in step_results.items()
        ])
        
        chain_summary = "\n".join([f"{i+1}. {step}" for i, step in enumerate(reasoning_chain)])
        
        return self.synthesis_prompt_template.format(
            query=query,
            results=results_summary,
            reasoning_chain=chain_summary
        )
    
    def _create_chain_of_thought_prompt(
        self,
        query: str,
        reasoning_chain: List[str],
        step_results: Dict[str, Any]
    ) -> str:
        """Create a chain-of-thought reasoning prompt."""
        chain_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(reasoning_chain)])
        
        return self.chain_of_thought_template.format(
            query=query,
            reasoning_chain=chain_text,
            step_count=len(reasoning_chain)
        )
    
    def _create_fallback_synthesis(self, step_results: Dict[str, Any]) -> str:
        """Create a fallback synthesis when LLM fails."""
        if not step_results:
            return "No results available."
        
        # Simple concatenation of results
        parts = []
        for step_id, result in step_results.items():
            content = self._extract_content_from_result(result)
            if content:
                parts.append(content)
        
        if parts:
            return " ".join(parts[:3])  # Limit to first 3 results
        return "Results were obtained but could not be synthesized."
    
    def _get_synthesis_prompt_template(self) -> str:
        """Get the synthesis prompt template."""
        return """
        You are an AI assistant helping to answer a user's query by synthesizing information from multiple sources.
        
        User Query: "{query}"
        
        Available Information:
        {results}
        
        Reasoning Steps:
        {reasoning_chain}
        
        Based on the above information, provide a clear, accurate, and concise answer to the user's query.
        Synthesize the information from all sources to create a comprehensive response.
        If the information is insufficient or contradictory, acknowledge this in your response.
        """
    
    def _get_chain_of_thought_template(self) -> str:
        """Get the chain-of-thought prompt template."""
        return """
        You are an AI assistant using step-by-step reasoning to answer a user's query.
        
        User Query: "{query}"
        
        Reasoning Chain ({step_count} steps):
        {reasoning_chain}
        
        Based on this step-by-step reasoning, provide a final answer to the user's query.
        Explain your reasoning clearly and show how each step contributes to the final answer.
        """


# Convenience function for creating the reasoning engine
def create_reasoning_engine(
    llm: BaseLLM,
    **kwargs
) -> ReasoningEngine:
    """
    Create a ReasoningEngine instance.
    
    Args:
        llm: Language model instance
        **kwargs: Additional configuration options
        
    Returns:
        Configured ReasoningEngine instance
    """
    return ReasoningEngine(
        llm=llm,
        **kwargs
    )
