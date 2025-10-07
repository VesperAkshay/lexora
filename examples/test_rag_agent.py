#!/usr/bin/env python3

"""
Test script to verify the RAGAgent functionality.
"""

import asyncio
import sys
from typing import Dict, Any

# Add the current directory to Python path for imports
sys.path.insert(0, '.')

from lexora import RAGAgent, AgentResponse, create_rag_agent, LLMConfig, VectorDBConfig, AgentConfig
from lexora.tools.base_tool import BaseTool, ToolResult, ToolStatus


class CustomTestTool(BaseTool):
    """Custom test tool for testing tool registration."""
    
    def __init__(self):
        self.name = "custom_test_tool"
        self.description = "A custom tool for testing"
        self.version = "1.0.0"
    
    async def run(self, **kwargs) -> ToolResult:
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={"message": "Custom tool executed successfully"}
        )
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "test_param": {"type": "string"}
            },
            "required": []
        }


async def test_agent_initialization():
    """Test RAGAgent initialization."""
    print("Testing RAGAgent Initialization...")
    
    # Test with default configuration
    agent = RAGAgent()
    
    assert agent.is_initialized is True
    assert agent.llm is not None
    assert agent.vector_db is not None
    assert agent.planner is not None
    assert agent.executor is not None
    assert agent.reasoning_engine is not None
    print("✅ Default initialization working")
    
    # Test with custom configuration
    llm_config = LLMConfig(
        provider="mock",
        model="test-model",
        temperature=0.5
    )
    
    vector_db_config = VectorDBConfig(
        provider="mock",
        embedding_model="test-embedding",
        dimension=512
    )
    
    agent_config = AgentConfig(
        max_context_length=10000,
        max_tool_calls=10,
        log_level="DEBUG"
    )
    
    agent2 = RAGAgent(
        llm_config=llm_config,
        vector_db_config=vector_db_config,
        agent_config=agent_config
    )
    
    assert agent2.llm_config.model == "test-model"
    assert agent2.vector_db_config.dimension == 512
    assert agent2.agent_config.max_context_length == 10000
    print("✅ Custom configuration working")
    
    # Test convenience function
    agent3 = create_rag_agent(llm_config=llm_config)
    assert isinstance(agent3, RAGAgent)
    print("✅ Convenience function working")


async def test_tool_management():
    """Test tool management functionality."""
    print("\n🧪 Testing Tool Management...")
    
    agent = RAGAgent()
    
    # Test getting available tools
    tools = agent.get_available_tools()
    assert isinstance(tools, list)
    assert len(tools) > 0
    print(f"✅ Found {len(tools)} default tools")
    
    # Test adding custom tool
    custom_tool = CustomTestTool()
    agent.add_tool(custom_tool, category="testing")
    
    updated_tools = agent.get_available_tools()
    assert "custom_test_tool" in updated_tools
    assert len(updated_tools) == len(tools) + 1
    print("✅ Custom tool registration working")
    
    # Test getting tool info
    tool_info = agent.get_tool_info("custom_test_tool")
    assert tool_info["name"] == "custom_test_tool"
    assert "description" in tool_info
    assert "parameters" in tool_info
    print("✅ Tool info retrieval working")


async def test_query_processing():
    """Test query processing through the full pipeline."""
    print("\n🧪 Testing Query Processing...")
    
    agent = RAGAgent()
    
    # Test simple query
    query = "What is machine learning?"
    response = await agent.query(query)
    
    assert isinstance(response, AgentResponse)
    assert response.answer is not None
    assert isinstance(response.confidence, float)
    assert 0.0 <= response.confidence <= 1.0
    assert response.execution_time > 0
    print("✅ Simple query processing working")
    
    # Test query with context
    context = {"user_id": "test_user", "session_id": "test_session"}
    response2 = await agent.query("Tell me about neural networks", context=context)
    
    assert isinstance(response2, AgentResponse)
    assert response2.answer is not None
    print("✅ Query with context working")
    
    # Test response structure
    response_dict = response.to_dict()
    assert "answer" in response_dict
    assert "confidence" in response_dict
    assert "sources" in response_dict
    assert "reasoning_chain" in response_dict
    assert "execution_time" in response_dict
    assert "metadata" in response_dict
    print("✅ Response structure correct")


async def test_query_history():
    """Test query history management."""
    print("\n🧪 Testing Query History...")
    
    agent = RAGAgent()
    
    # Process some queries
    await agent.query("First query")
    await agent.query("Second query")
    await agent.query("Third query")
    
    # Get history
    history = agent.get_query_history(limit=10)
    
    assert len(history) == 3
    assert history[0]["query"] == "First query"
    assert history[2]["query"] == "Third query"
    print("✅ Query history tracking working")
    
    # Test history limit
    limited_history = agent.get_query_history(limit=2)
    assert len(limited_history) == 2
    assert limited_history[0]["query"] == "Second query"
    print("✅ History limit working")
    
    # Test clear history
    agent.clear_history()
    empty_history = agent.get_query_history()
    assert len(empty_history) == 0
    print("✅ History clearing working")


async def test_health_check():
    """Test health check functionality."""
    print("\n🧪 Testing Health Check...")
    
    agent = RAGAgent()
    
    health_status = await agent.health_check()
    
    assert isinstance(health_status, dict)
    assert "agent" in health_status
    assert "components" in health_status
    assert "timestamp" in health_status
    
    # Check component statuses
    components = health_status["components"]
    assert "llm" in components
    assert "vector_db" in components
    assert "tool_registry" in components
    assert "planner" in components
    assert "executor" in components
    assert "reasoning_engine" in components
    
    print(f"✅ Health check working (status: {health_status['agent']})")


async def test_error_handling():
    """Test error handling in query processing."""
    print("\n🧪 Testing Error Handling...")
    
    agent = RAGAgent()
    
    # Test empty query
    response = await agent.query("")
    assert isinstance(response, AgentResponse)
    assert response.confidence == 0.0
    assert "error" in response.metadata
    print("✅ Empty query handling working")
    
    # Test whitespace-only query
    response2 = await agent.query("   ")
    assert isinstance(response2, AgentResponse)
    assert response2.confidence == 0.0
    print("✅ Whitespace query handling working")


async def test_agent_representation():
    """Test agent string representation."""
    print("\n🧪 Testing Agent Representation...")
    
    agent = RAGAgent()
    
    repr_str = repr(agent)
    assert "RAGAgent" in repr_str
    assert "llm=" in repr_str
    assert "vector_db=" in repr_str
    assert "tools=" in repr_str
    print(f"✅ Agent representation working: {repr_str}")


async def test_configuration_management():
    """Test configuration management."""
    print("\n🧪 Testing Configuration Management...")
    
    # Test default configurations
    agent = RAGAgent()
    
    assert agent.llm_config is not None
    assert agent.vector_db_config is not None
    assert agent.agent_config is not None
    print("✅ Default configurations loaded")
    
    # Test custom configurations
    custom_llm_config = LLMConfig(
        provider="mock",
        model="custom-model",
        temperature=0.3,
        max_tokens=1500
    )
    
    custom_vector_config = VectorDBConfig(
        provider="mock",
        embedding_model="custom-embedding",
        dimension=768
    )
    
    custom_agent_config = AgentConfig(
        max_context_length=20000,
        max_tool_calls=15,
        log_level="WARNING"
    )
    
    agent2 = RAGAgent(
        llm_config=custom_llm_config,
        vector_db_config=custom_vector_config,
        agent_config=custom_agent_config
    )
    
    assert agent2.llm_config.model == "custom-model"
    assert agent2.llm_config.temperature == 0.3
    assert agent2.vector_db_config.dimension == 768
    assert agent2.agent_config.max_tool_calls == 15
    print("✅ Custom configurations applied correctly")


async def test_tool_registration_with_initialization():
    """Test registering tools during initialization."""
    print("\n🧪 Testing Tool Registration During Initialization...")
    
    custom_tools = [
        CustomTestTool()
    ]
    
    agent = RAGAgent(tools=custom_tools)
    
    tools = agent.get_available_tools()
    assert "custom_test_tool" in tools
    print("✅ Tools registered during initialization")


async def test_multiple_queries():
    """Test processing multiple queries in sequence."""
    print("\n🧪 Testing Multiple Queries...")
    
    agent = RAGAgent()
    
    queries = [
        "What is artificial intelligence?",
        "Explain machine learning",
        "What are neural networks?"
    ]
    
    responses = []
    for query in queries:
        response = await agent.query(query)
        responses.append(response)
        assert isinstance(response, AgentResponse)
        assert response.answer is not None
    
    assert len(responses) == 3
    assert all(r.execution_time > 0 for r in responses)
    print("✅ Multiple queries processed successfully")


async def main():
    """Run all RAGAgent tests."""
    print("Starting RAGAgent Tests\n")
    
    try:
        await test_agent_initialization()
        await test_tool_management()
        await test_query_processing()
        await test_query_history()
        await test_health_check()
        await test_error_handling()
        await test_agent_representation()
        await test_configuration_management()
        await test_tool_registration_with_initialization()
        await test_multiple_queries()
        
        print("\nAll RAGAgent tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
