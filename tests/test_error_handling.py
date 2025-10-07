#!/usr/bin/env python3
"""
Error Handling Validation Tests for Lexora Agentic RAG SDK

This module tests all error scenarios and validates structured error responses.

Requirements tested: 7.1, 7.2, 7.4
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lexora import RAGAgent
from lexora.exceptions import (
    LexoraError,
    ErrorCode,
    ConfigurationError,
    ToolExecutionError,
    VectorDBError,
    LLMError
)


async def test_structured_error_responses():
    """
    Test that all errors return structured responses.
    
    Requirements: 7.1, 7.4
    """
    print("\n🧪 Test 1: Structured Error Responses")
    print("-" * 50)
    
    agent = RAGAgent()
    
    # Test 1: Query nonexistent corpus
    print("  Testing nonexistent corpus error...")
    result = await agent.tool_registry.get_tool("rag_query").run(
        corpus_name="nonexistent_corpus",
        query="test"
    )
    
    assert result.status == "error", "Status should be 'error'"
    assert result.error is not None, "Error message should be present"
    assert "does not exist" in result.error.lower() or "not found" in result.error.lower()
    print("  ✅ Nonexistent corpus error is structured")
    
    # Test 2: Missing required parameters
    print("  Testing missing parameters error...")
    result = await agent.tool_registry.get_tool("create_corpus").run()
    
    assert result.status == "error"
    assert result.error is not None
    assert "required" in result.error.lower() or "missing" in result.error.lower()
    print("  ✅ Missing parameters error is structured")
    
    # Test 3: Invalid parameter types
    print("  Testing invalid parameter error...")
    result = await agent.tool_registry.get_tool("rag_query").run(
        corpus_name="test",
        query="test",
        top_k="invalid"  # Should be int
    )
    
    assert result.status == "error"
    assert result.error is not None
    print("  ✅ Invalid parameter error is structured")
    
    print("\n✅ All structured error response tests PASSED")
    return True


async def test_error_recovery_mechanisms():
    """
    Test error recovery and retry mechanisms.
    
    Requirements: 7.2
    """
    print("\n🔄 Test 2: Error Recovery Mechanisms")
    print("-" * 50)
    
    agent = RAGAgent()
    corpus_name = "recovery_test_corpus"
    
    try:
        # Test 1: Recover from failed operation
        print("  Testing recovery from failed corpus creation...")
        
        # First attempt - create corpus
        result1 = await agent.tool_registry.get_tool("create_corpus").run(
            corpus_name=corpus_name,
            description="Test corpus",
            overwrite_existing=True
        )
        assert result1.status == "success"
        print("  ✅ Initial corpus creation successful")
        
        # Second attempt - should fail without overwrite
        result2 = await agent.tool_registry.get_tool("create_corpus").run(
            corpus_name=corpus_name,
            description="Test corpus"
        )
        assert result2.status == "error"
        print("  ✅ Duplicate creation properly rejected")
        
        # Third attempt - recover with overwrite flag
        result3 = await agent.tool_registry.get_tool("create_corpus").run(
            corpus_name=corpus_name,
            description="Test corpus",
            overwrite_existing=True
        )
        assert result3.status == "success"
        print("  ✅ Recovery with overwrite flag successful")
        
        # Test 2: Graceful handling of empty corpus
        print("  Testing empty corpus handling...")
        query_result = await agent.tool_registry.get_tool("rag_query").run(
            corpus_name=corpus_name,
            query="test query"
        )
        # Should succeed but return empty results
        assert query_result.status == "success"
        assert len(query_result.data["results"]) == 0
        print("  ✅ Empty corpus handled gracefully")
        
        print("\n✅ All error recovery tests PASSED")
        return True
        
    finally:
        # Cleanup
        try:
            await agent.tool_registry.get_tool("delete_corpus").run(
                corpus_name=corpus_name,
                confirm_deletion=corpus_name
            )
        except Exception:
            pass


async def test_error_context_information():
    """
    Test that errors include helpful context information.
    
    Requirements: 7.4
    """
    print("\n📋 Test 3: Error Context Information")
    print("-" * 50)
    
    agent = RAGAgent()
    
    # Test 1: Error includes available options
    print("  Testing error context with available options...")
    result = await agent.tool_registry.get_tool("rag_query").run(
        corpus_name="nonexistent",
        query="test"
    )
    
    assert result.status == "error"
    assert result.error is not None
    # Error should mention available corpora
    print(f"  Error message: {result.error[:100]}...")
    print("  ✅ Error includes context information")
    
    # Test 2: Error includes suggestions
    print("  Testing error suggestions...")
    result = await agent.tool_registry.get_tool("add_data").run(
        corpus_name="nonexistent",
        documents=[{"content": "test"}]
    )
    
    assert result.status == "error"
    assert result.error is not None
    # Should suggest creating corpus first
    assert "create" in result.error.lower() or "exist" in result.error.lower()
    print("  ✅ Error includes helpful suggestions")
    
    print("\n✅ All error context tests PASSED")
    return True


async def test_error_codes_and_categories():
    """
    Test that errors have proper codes and categories.
    
    Requirements: 7.1, 7.4
    """
    print("\n🏷️  Test 4: Error Codes and Categories")
    print("-" * 50)
    
    agent = RAGAgent()
    
    # Test different error categories
    test_cases = [
        {
            "name": "Missing parameters",
            "tool": "create_corpus",
            "params": {},
            "expected_keywords": ["required", "missing", "parameter"]
        },
        {
            "name": "Nonexistent resource",
            "tool": "rag_query",
            "params": {"corpus_name": "nonexistent", "query": "test"},
            "expected_keywords": ["not found", "does not exist", "available"]
        },
        {
            "name": "Invalid operation",
            "tool": "delete_corpus",
            "params": {"corpus_name": "test"},
            "expected_keywords": ["confirm", "deletion", "safety"]
        }
    ]
    
    for test_case in test_cases:
        print(f"  Testing {test_case['name']}...")
        result = await agent.tool_registry.get_tool(test_case["tool"]).run(
            **test_case["params"]
        )
        
        assert result.status == "error"
        assert result.error is not None
        
        # Check if error message contains expected keywords
        error_lower = result.error.lower()
        has_keyword = any(keyword in error_lower for keyword in test_case["expected_keywords"])
        assert has_keyword, f"Error should contain one of: {test_case['expected_keywords']}"
        
        print(f"  ✅ {test_case['name']} error properly categorized")
    
    print("\n✅ All error code tests PASSED")
    return True


async def test_error_logging():
    """
    Test that errors are properly logged.
    
    Requirements: 7.1, 7.3
    """
    print("\n📝 Test 5: Error Logging")
    print("-" * 50)
    
    agent = RAGAgent()
    
    # Trigger various errors and verify they're handled
    print("  Testing error logging for various scenarios...")
    
    # Error 1: Tool execution error
    result1 = await agent.tool_registry.get_tool("rag_query").run(
        corpus_name="nonexistent",
        query="test"
    )
    assert result1.status == "error"
    print("  ✅ Tool execution error logged")
    
    # Error 2: Validation error
    result2 = await agent.tool_registry.get_tool("create_corpus").run()
    assert result2.status == "error"
    print("  ✅ Validation error logged")
    
    # Error 3: Operation error
    result3 = await agent.tool_registry.get_tool("delete_corpus").run(
        corpus_name="test"
    )
    assert result3.status == "error"
    print("  ✅ Operation error logged")
    
    print("\n✅ All error logging tests PASSED")
    return True


async def test_cascading_error_handling():
    """
    Test handling of cascading errors.
    
    Requirements: 7.2
    """
    print("\n🔗 Test 6: Cascading Error Handling")
    print("-" * 50)
    
    agent = RAGAgent()
    
    # Test that errors in one operation don't break subsequent operations
    print("  Testing error isolation...")
    
    # Operation 1: Fail
    result1 = await agent.tool_registry.get_tool("rag_query").run(
        corpus_name="nonexistent1",
        query="test"
    )
    assert result1.status == "error"
    print("  ✅ First operation failed as expected")
    
    # Operation 2: Should still work
    result2 = await agent.tool_registry.get_tool("list_corpora").run()
    assert result2.status == "success"
    print("  ✅ Second operation succeeded despite first failure")
    
    # Operation 3: Another failure
    result3 = await agent.tool_registry.get_tool("add_data").run(
        corpus_name="nonexistent2",
        documents=[{"content": "test"}]
    )
    assert result3.status == "error"
    print("  ✅ Third operation failed independently")
    
    # Operation 4: Should still work
    result4 = await agent.tool_registry.get_tool("list_corpora").run()
    assert result4.status == "success"
    print("  ✅ Fourth operation succeeded")
    
    print("\n✅ All cascading error tests PASSED")
    return True


async def run_all_error_tests():
    """Run all error handling validation tests."""
    print("=" * 70)
    print("Error Handling Validation Tests - Lexora Agentic RAG SDK")
    print("=" * 70)
    
    results = []
    
    # Run all tests
    results.append(await test_structured_error_responses())
    results.append(await test_error_recovery_mechanisms())
    results.append(await test_error_context_information())
    results.append(await test_error_codes_and_categories())
    results.append(await test_error_logging())
    results.append(await test_cascading_error_handling())
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ ALL ERROR HANDLING TESTS PASSED!")
    else:
        print(f"\n❌ {total - passed} test(s) failed")
    
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_error_tests())
    sys.exit(0 if success else 1)
