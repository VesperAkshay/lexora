#!/usr/bin/env python3
"""Simple test for RAGAgent without unicode characters."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lexora import RAGAgent, create_rag_agent, LLMConfig, VectorDBConfig, AgentConfig


async def test_basic_initialization():
    """Test basic RAGAgent initialization."""
    print("Test 1: Basic initialization...")
    try:
        agent = RAGAgent()
        assert agent.is_initialized is True
        assert agent.llm is not None
        assert agent.vector_db is not None
        print("PASS: Basic initialization")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False


async def test_tool_management():
    """Test tool management."""
    print("\nTest 2: Tool management...")
    try:
        agent = RAGAgent()
        tools = agent.get_available_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0
        print(f"PASS: Found {len(tools)} tools")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False


async def test_query_processing():
    """Test query processing."""
    print("\nTest 3: Query processing...")
    try:
        agent = RAGAgent()
        response = await agent.query("What is machine learning?")
        assert response is not None
        assert response.answer is not None
        assert isinstance(response.confidence, float)
        print(f"PASS: Query processed (confidence: {response.confidence:.2f})")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False


async def test_configuration():
    """Test custom configuration."""
    print("\nTest 4: Custom configuration...")
    try:
        llm_config = LLMConfig(
            provider="mock",
            model="test-model",
            temperature=0.5
        )
        agent = create_rag_agent(llm_config=llm_config)
        assert agent.llm_config.model == "test-model"
        print("PASS: Custom configuration")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False


async def test_health_check():
    """Test health check."""
    print("\nTest 5: Health check...")
    try:
        agent = RAGAgent()
        health = await agent.health_check()
        assert "agent" in health
        assert "components" in health
        print(f"PASS: Health check (status: {health['agent']})")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False


async def main():
    """Run all tests."""
    print("=" * 50)
    print("RAGAgent Simple Test Suite")
    print("=" * 50)
    
    results = []
    results.append(await test_basic_initialization())
    results.append(await test_tool_management())
    results.append(await test_query_processing())
    results.append(await test_configuration())
    results.append(await test_health_check())
    
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("\nAll tests PASSED!")
        sys.exit(0)
    else:
        print(f"\n{total - passed} test(s) FAILED")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
