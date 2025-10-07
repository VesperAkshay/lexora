"""
Test script for Lexora SDK with Gemini
"""
import asyncio
import os
from lexora import RAGAgent
from lexora.models.config import LLMConfig, VectorDBConfig, AgentConfig

async def test_lexora_sdk():
    """Test the Lexora SDK with Gemini"""
    
    print("=" * 60)
    print("Testing Lexora SDK with Gemini")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n❌ Error: GEMINI_API_KEY environment variable not set")
        print("Please set it with: $env:GEMINI_API_KEY='your-api-key'")
        return
    
    try:
        # Configure LLM (Gemini)
        print("\n1. Configuring Gemini LLM...")
        llm_config = LLMConfig(
            provider="litellm",
            model="gemini/gemini-1.5-flash",
            api_key=api_key,
            temperature=0.7,
            max_tokens=1000
        )
        print("✓ LLM configured")
        
        # Configure Vector DB (FAISS)
        print("\n2. Configuring FAISS Vector DB...")
        vector_db_config = VectorDBConfig(
            provider="faiss",
            dimension=384,  # all-MiniLM-L6-v2 dimension
            connection_params={
                "index_type": "Flat",
                "persist_directory": "./test_faiss_db"
            }
        )
        print("✓ Vector DB configured")
        
        # Configure Agent
        print("\n3. Configuring RAG Agent...")
        agent_config = AgentConfig(
            max_iterations=3,
            enable_reasoning=True,
            enable_planning=True
        )
        print("✓ Agent configured")
        
        # Initialize RAG Agent
        print("\n4. Initializing RAG Agent...")
        agent = RAGAgent(
            llm_config=llm_config,
            vector_db_config=vector_db_config,
            agent_config=agent_config
        )
        await agent.initialize()
        print("✓ Agent initialized successfully")
        
        # Create a test corpus
        print("\n5. Creating test corpus...")
        corpus_result = await agent.tool_registry.execute_tool(
            "create_corpus",
            corpus_name="test_corpus",
            description="Test corpus for SDK validation"
        )
        print(f"✓ Corpus created: {corpus_result.status}")
        
        # Add test documents
        print("\n6. Adding test documents...")
        doc1_result = await agent.tool_registry.execute_tool(
            "add_data",
            corpus_name="test_corpus",
            content="Python is a high-level programming language known for its simplicity and readability.",
            metadata={"topic": "programming", "language": "python"}
        )
        print(f"✓ Document 1 added: {doc1_result.data.get('document_id', 'N/A')}")
        
        doc2_result = await agent.tool_registry.execute_tool(
            "add_data",
            corpus_name="test_corpus",
            content="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            metadata={"topic": "ai", "subtopic": "machine learning"}
        )
        print(f"✓ Document 2 added: {doc2_result.data.get('document_id', 'N/A')}")
        
        # List corpora
        print("\n7. Listing corpora...")
        list_result = await agent.tool_registry.execute_tool("list_corpora")
        print(f"✓ Found {len(list_result.data.get('corpora', []))} corpus(es)")
        
        # Query the agent
        print("\n8. Testing RAG query...")
        query = "What is Python?"
        print(f"Query: {query}")
        
        response = await agent.query(query)
        print(f"\n✓ Response received:")
        print(f"  Answer: {response.answer}")
        print(f"  Confidence: {response.confidence}")
        print(f"  Sources: {len(response.sources)}")
        print(f"  Execution time: {response.execution_time:.2f}s")
        
        # Get corpus info
        print("\n9. Getting corpus info...")
        info_result = await agent.tool_registry.execute_tool(
            "get_corpus_info",
            corpus_name="test_corpus"
        )
        print(f"✓ Corpus info: {info_result.data.get('document_count', 0)} documents")
        
        # Cleanup
        print("\n10. Cleaning up...")
        await agent.tool_registry.execute_tool("delete_corpus", corpus_name="test_corpus")
        print("✓ Test corpus deleted")
        
        await agent.cleanup()
        print("✓ Agent cleaned up")
        
        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_lexora_sdk())
    exit(0 if success else 1)
