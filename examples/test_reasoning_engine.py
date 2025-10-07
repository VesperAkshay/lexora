#!/usr/bin/env python3

"""
Test script to verify the reasoning engine functionality.
"""

import asyncio
import sys
from typing import Dict, Any

# Add the current directory to Python path for imports
sys.path.insert(0, '.')

from lexora.rag_agent.reasoning import (
    ReasoningEngine,
    ReasoningResult,
    ReasoningStrategy,
    ConfidenceLevel,
    SourceAttribution,
    create_reasoning_engine
)
from lexora.rag_agent.planner import ExecutionPlan, PlanStep, PlanStepType, PlanStatus
from lexora.rag_agent.executor import ExecutionResult, ExecutionContext
from lexora.llm.base_llm import MockLLMProvider


async def test_reasoning_engine_initialization():
    """Test reasoning engine initialization."""
    print("🧪 Testing ReasoningEngine Initialization...")
    
    llm = MockLLMProvider()
    engine = ReasoningEngine(llm=llm)
    
    assert engine.llm == llm
    assert engine.default_strategy == ReasoningStrategy.SYNTHESIS
    assert engine.min_confidence_threshold == 0.3
    print("✅ Reasoning engine initialization working")
    
    # Test with custom config
    engine2 = ReasoningEngine(
        llm=llm,
        default_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
        min_confidence_threshold=0.5
    )
    
    assert engine2.default_strategy == ReasoningStrategy.CHAIN_OF_THOUGHT
    assert engine2.min_confidence_threshold == 0.5
    print("✅ Custom configuration working")
    
    # Test convenience function
    engine3 = create_reasoning_engine(llm=llm, max_sources=5)
    assert isinstance(engine3, ReasoningEngine)
    assert engine3.max_sources == 5
    print("✅ Convenience function working")


async def test_source_attribution():
    """Test source attribution functionality."""
    print("\n🧪 Testing Source Attribution...")
    
    source = SourceAttribution(
        source_type="tool_result",
        source_id="step_1",
        content="Test content from tool",
        relevance_score=0.85
    )
    
    assert source.source_type == "tool_result"
    assert source.source_id == "step_1"
    assert source.relevance_score == 0.85
    print("✅ Source attribution creation working")
    
    # Test serialization
    source_dict = source.to_dict()
    assert "source_type" in source_dict
    assert "source_id" in source_dict
    assert "relevance_score" in source_dict
    print("✅ Source attribution serialization working")


async def test_reasoning_result():
    """Test reasoning result functionality."""
    print("\n🧪 Testing Reasoning Result...")
    
    result = ReasoningResult(
        answer="This is a test answer",
        confidence=0.75,
        confidence_level=ConfidenceLevel.HIGH,
        sources=[
            SourceAttribution(
                source_type="tool_result",
                source_id="step_1",
                content="Source content"
            )
        ],
        reasoning_chain=["Step 1: Analyzed query", "Step 2: Generated response"]
    )
    
    assert result.answer == "This is a test answer"
    assert result.confidence == 0.75
    assert result.confidence_level == ConfidenceLevel.HIGH
    assert len(result.sources) == 1
    assert len(result.reasoning_chain) == 2
    print("✅ Reasoning result creation working")
    
    # Test serialization
    result_dict = result.to_dict()
    assert "answer" in result_dict
    assert "confidence" in result_dict
    assert "confidence_level" in result_dict
    assert "sources" in result_dict
    assert "reasoning_chain" in result_dict
    print("✅ Reasoning result serialization working")


async def test_confidence_levels():
    """Test confidence level calculation."""
    print("\n🧪 Testing Confidence Levels...")
    
    llm = MockLLMProvider()
    engine = ReasoningEngine(llm=llm)
    
    # Test different confidence scores
    assert engine._get_confidence_level(0.2) == ConfidenceLevel.VERY_LOW
    assert engine._get_confidence_level(0.4) == ConfidenceLevel.LOW
    assert engine._get_confidence_level(0.6) == ConfidenceLevel.MEDIUM
    assert engine._get_confidence_level(0.8) == ConfidenceLevel.HIGH
    assert engine._get_confidence_level(0.95) == ConfidenceLevel.VERY_HIGH
    print("✅ Confidence level calculation working")


async def test_direct_response_generation():
    """Test direct response generation strategy."""
    print("\n🧪 Testing Direct Response Generation...")
    
    llm = MockLLMProvider()
    engine = ReasoningEngine(llm=llm)
    
    # Create test data
    query = "What is machine learning?"
    step_results = {
        "step_1": {
            "answer": "Machine learning is a subset of AI",
            "confidence": 0.9
        }
    }
    
    result = await engine._generate_direct_response(query, step_results, None)
    
    assert isinstance(result, ReasoningResult)
    assert result.answer is not None
    assert result.confidence > 0
    assert len(result.sources) > 0
    print("✅ Direct response generation working")


async def test_synthesis_response_generation():
    """Test synthesis response generation strategy."""
    print("\n🧪 Testing Synthesis Response Generation...")
    
    llm = MockLLMProvider()
    engine = ReasoningEngine(llm=llm)
    
    # Create test data with multiple results
    query = "What are the benefits of machine learning?"
    step_results = {
        "step_1": {
            "data": "ML can automate tasks"
        },
        "step_2": {
            "data": "ML improves accuracy over time"
        },
        "step_3": {
            "data": "ML can handle large datasets"
        }
    }
    
    result = await engine._generate_synthesis_response(query, step_results, None)
    
    assert isinstance(result, ReasoningResult)
    assert result.answer is not None
    assert result.confidence > 0
    assert len(result.sources) == 3
    assert len(result.reasoning_chain) > 0
    print("✅ Synthesis response generation working")


async def test_multi_step_synthesis():
    """Test multi-step result synthesis."""
    print("\n🧪 Testing Multi-Step Synthesis...")
    
    llm = MockLLMProvider()
    engine = ReasoningEngine(llm=llm)
    
    query = "Explain neural networks"
    step_results = {
        "step_1": {"content": "Neural networks are inspired by biological neurons"},
        "step_2": {"content": "They consist of layers of interconnected nodes"},
        "step_3": {"content": "They learn through backpropagation"}
    }
    
    result = await engine.synthesize_multi_step_results(query, step_results)
    
    assert isinstance(result, ReasoningResult)
    assert result.answer is not None
    assert len(result.sources) == 3
    assert result.metadata["step_count"] == 3
    print("✅ Multi-step synthesis working")


async def test_generate_response_with_plan():
    """Test generating response with execution plan."""
    print("\n🧪 Testing Response Generation with Plan...")
    
    llm = MockLLMProvider()
    engine = ReasoningEngine(llm=llm)
    
    # Create a test plan
    plan = ExecutionPlan(
        id="test_plan",
        query="What is deep learning?",
        steps=[
            PlanStep(
                id="step_1",
                type=PlanStepType.TOOL_EXECUTION,
                description="Search for deep learning information",
                status=PlanStatus.COMPLETED
            ),
            PlanStep(
                id="step_2",
                type=PlanStepType.ANALYSIS,
                description="Analyze search results",
                status=PlanStatus.COMPLETED
            )
        ]
    )
    
    # Create execution result
    execution_result = ExecutionResult(
        success=True,
        result={
            "step_results": {
                "step_1": {"data": "Deep learning uses neural networks"},
                "step_2": {"analysis": "Deep learning is effective for complex patterns"}
            }
        },
        execution_time=1.5
    )
    
    # Generate response
    result = await engine.generate_response(
        query="What is deep learning?",
        plan=plan,
        execution_result=execution_result,
        strategy=ReasoningStrategy.SYNTHESIS
    )
    
    assert isinstance(result, ReasoningResult)
    assert result.answer is not None
    assert result.confidence > 0
    assert "plan_id" in result.metadata
    assert result.metadata["plan_id"] == "test_plan"
    print("✅ Response generation with plan working")


async def test_chain_of_thought_reasoning():
    """Test chain-of-thought reasoning strategy."""
    print("\n🧪 Testing Chain-of-Thought Reasoning...")
    
    llm = MockLLMProvider()
    engine = ReasoningEngine(llm=llm, enable_chain_of_thought=True)
    
    # Create a plan with multiple steps
    plan = ExecutionPlan(
        id="test_plan",
        query="How does gradient descent work?",
        steps=[
            PlanStep(
                id="step_1",
                type=PlanStepType.INFORMATION_GATHERING,
                description="Gather information about gradient descent",
                status=PlanStatus.COMPLETED
            ),
            PlanStep(
                id="step_2",
                type=PlanStepType.ANALYSIS,
                description="Analyze the optimization process",
                status=PlanStatus.COMPLETED
            ),
            PlanStep(
                id="step_3",
                type=PlanStepType.SYNTHESIS,
                description="Synthesize explanation",
                status=PlanStatus.COMPLETED
            )
        ]
    )
    
    step_results = {
        "step_1": {"info": "Gradient descent is an optimization algorithm"},
        "step_2": {"analysis": "It iteratively adjusts parameters"},
        "step_3": {"synthesis": "It minimizes the loss function"}
    }
    
    result = await engine._generate_chain_of_thought_response(
        query="How does gradient descent work?",
        plan=plan,
        step_results=step_results,
        context=None
    )
    
    assert isinstance(result, ReasoningResult)
    assert result.answer is not None
    assert len(result.reasoning_chain) == 3
    assert result.metadata["strategy"] == "chain_of_thought"
    print("✅ Chain-of-thought reasoning working")


async def test_empty_results_handling():
    """Test handling of empty results."""
    print("\n🧪 Testing Empty Results Handling...")
    
    llm = MockLLMProvider()
    engine = ReasoningEngine(llm=llm)
    
    query = "Test query"
    step_results = {}
    
    result = await engine._generate_synthesis_response(query, step_results, None)
    
    assert isinstance(result, ReasoningResult)
    assert result.confidence == 0.0
    assert result.confidence_level == ConfidenceLevel.VERY_LOW
    assert "results_found" in result.metadata
    assert result.metadata["results_found"] is False
    print("✅ Empty results handling working")


async def test_source_extraction():
    """Test source extraction from results."""
    print("\n🧪 Testing Source Extraction...")
    
    llm = MockLLMProvider()
    engine = ReasoningEngine(llm=llm)
    
    step_results = {
        "step_1": {"content": "First piece of information"},
        "step_2": {"content": "Second piece of information"},
        "step_3": {"content": "Third piece of information"}
    }
    
    sources = engine._extract_sources(step_results)
    
    assert len(sources) == 3
    assert all(isinstance(s, SourceAttribution) for s in sources)
    assert all(s.source_type == "tool_result" for s in sources)
    # Sources should be sorted by relevance
    assert sources[0].relevance_score >= sources[-1].relevance_score
    print("✅ Source extraction working")


async def test_confidence_calculation():
    """Test confidence score calculation."""
    print("\n🧪 Testing Confidence Calculation...")
    
    llm = MockLLMProvider()
    engine = ReasoningEngine(llm=llm)
    
    # Test with multiple results
    step_results = {
        "step_1": {"data": "result 1"},
        "step_2": {"data": "result 2"},
        "step_3": {"data": "result 3"}
    }
    
    sources = engine._extract_sources(step_results)
    confidence = engine._calculate_confidence(step_results, sources)
    
    assert 0.0 <= confidence <= 1.0
    assert confidence > 0  # Should have some confidence with results
    print("✅ Confidence calculation working")
    
    # Test with no results
    empty_confidence = engine._calculate_confidence({}, [])
    assert empty_confidence == 0.0
    print("✅ Empty results confidence calculation working")


async def main():
    """Run all reasoning engine tests."""
    print("🚀 Starting Reasoning Engine Tests\n")
    
    try:
        await test_reasoning_engine_initialization()
        await test_source_attribution()
        await test_reasoning_result()
        await test_confidence_levels()
        await test_direct_response_generation()
        await test_synthesis_response_generation()
        await test_multi_step_synthesis()
        await test_generate_response_with_plan()
        await test_chain_of_thought_reasoning()
        await test_empty_results_handling()
        await test_source_extraction()
        await test_confidence_calculation()
        
        print("\n🎉 All reasoning engine tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
