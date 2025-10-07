"""
Pytest configuration and fixtures for the Lexora Agentic RAG SDK tests.

This module provides common test fixtures, configuration, and utilities
used across all test modules.
"""

import pytest
import asyncio
import logging
from typing import Generator, AsyncGenerator
from unittest.mock import Mock, patch

from lexora.llm.base_llm import MockLLMProvider
from lexora.utils.logging import configure_logging


# Configure logging for tests to reduce noise
configure_logging(level="ERROR", structured=False)


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """
    Create an event loop for the entire test session.
    
    This ensures that async tests can run properly and share the same event loop.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    """
    Create a basic MockLLMProvider for testing.
    
    Returns:
        Configured MockLLMProvider instance
    """
    return MockLLMProvider(
        model="test-mock-llm",
        response_template="Test response: {prompt}",
        simulate_delay=0.01,
        fail_probability=0.0
    )


@pytest.fixture
def unreliable_mock_llm() -> MockLLMProvider:
    """
    Create a MockLLMProvider that occasionally fails for testing retry logic.
    
    Returns:
        MockLLMProvider with 30% failure rate
    """
    return MockLLMProvider(
        model="unreliable-mock-llm",
        response_template="Unreliable response: {prompt}",
        simulate_delay=0.01,
        fail_probability=0.3
    )


@pytest.fixture
def failing_mock_llm() -> MockLLMProvider:
    """
    Create a MockLLMProvider that always fails for testing error handling.
    
    Returns:
        MockLLMProvider with 100% failure rate
    """
    return MockLLMProvider(
        model="failing-mock-llm",
        simulate_delay=0.01,
        fail_probability=1.0
    )


@pytest.fixture
def sample_json_schema() -> dict:
    """
    Provide a sample JSON schema for testing structured generation.
    
    Returns:
        Sample JSON schema dictionary
    """
    return {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The answer to the question"
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score between 0 and 1",
                "minimum": 0,
                "maximum": 1
            },
            "reasoning": {
                "type": "string",
                "description": "Explanation of the reasoning"
            }
        },
        "required": ["answer", "confidence"]
    }


@pytest.fixture
def sample_prompts() -> list:
    """
    Provide sample prompts for batch testing.
    
    Returns:
        List of sample prompts
    """
    return [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "How do you make a paper airplane?",
        "What are the benefits of renewable energy?",
        "Describe the process of photosynthesis."
    ]


@pytest.fixture
def mock_litellm_response():
    """
    Create a mock response object that mimics litellm's response structure.
    
    Returns:
        Mock response object
    """
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Mocked LLM response"
    return mock_response


@pytest.fixture
def mock_litellm_structured_response():
    """
    Create a mock structured response for testing JSON generation.
    
    Returns:
        Mock response with JSON content
    """
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '{"answer": "Paris", "confidence": 0.95}'
    return mock_response


class AsyncMockContext:
    """
    Context manager for mocking async functions in tests.
    
    This helper class makes it easier to mock async functions and
    control their behavior during tests.
    """
    
    def __init__(self, mock_func, return_value=None, side_effect=None):
        """
        Initialize async mock context.
        
        Args:
            mock_func: The function to mock
            return_value: Value to return from the mock
            side_effect: Side effect for the mock (e.g., exception)
        """
        self.mock_func = mock_func
        self.return_value = return_value
        self.side_effect = side_effect
        self.patcher = None
    
    def __enter__(self):
        """Enter the context and start mocking."""
        self.patcher = patch(self.mock_func)
        mock = self.patcher.start()
        
        if self.side_effect:
            mock.side_effect = self.side_effect
        else:
            mock.return_value = self.return_value
        
        return mock
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and stop mocking."""
        if self.patcher:
            self.patcher.stop()


@pytest.fixture
def async_mock_context():
    """
    Provide AsyncMockContext for easy async function mocking.
    
    Returns:
        AsyncMockContext class
    """
    return AsyncMockContext


# Pytest markers for test categorization
pytest_plugins = []

# Custom markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "async_test: mark test as async"
    )


# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """
    Modify test items during collection.
    
    This function automatically adds markers to tests based on their names
    and characteristics.
    """
    for item in items:
        # Mark async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.async_test)
        
        # Mark slow tests
        if "batch" in item.name or "concurrent" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.name or "end_to_end" in item.name:
            item.add_marker(pytest.mark.integration)
        else:
            # Default to unit test
            item.add_marker(pytest.mark.unit)