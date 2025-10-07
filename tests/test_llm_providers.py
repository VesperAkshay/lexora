"""
Unit tests for LLM providers in the Lexora Agentic RAG SDK.

This module tests the BaseLLM interface compliance, error handling,
retry mechanisms, and provider-specific functionality.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

from llm.base_llm import BaseLLM, MockLLMProvider, create_mock_llm, validate_llm_provider
from llm.litellm_provider import LitellmProvider, create_litellm_provider, get_available_models
from exceptions import LexoraError, ErrorCode
from utils.logging import configure_logging


# Configure logging for tests
configure_logging(level="ERROR", structured=False)


class TestBaseLLM:
    """Test cases for the BaseLLM abstract interface."""
    
    def test_base_llm_is_abstract(self):
        """Test that BaseLLM cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLLM("test-model")
    
    def test_base_llm_interface_methods(self):
        """Test that BaseLLM defines the required abstract methods."""
        # Check that the abstract methods are defined
        assert hasattr(BaseLLM, 'generate')
        assert hasattr(BaseLLM, 'generate_structured')
        
        # Check that they are marked as abstract
        assert getattr(BaseLLM.generate, '__isabstractmethod__', False)
        assert getattr(BaseLLM.generate_structured, '__isabstractmethod__', False)
    
    def test_base_llm_concrete_methods(self):
        """Test that BaseLLM provides concrete utility methods."""
        # Create a mock implementation
        mock_llm = MockLLMProvider("test-model")
        
        # Test concrete methods
        assert hasattr(mock_llm, 'get_token_count')
        assert hasattr(mock_llm, 'get_model_name')
        assert hasattr(mock_llm, 'get_config')
        assert hasattr(mock_llm, 'validate_config')
        
        # Test method functionality
        assert mock_llm.get_model_name() == "test-model"
        assert isinstance(mock_llm.get_config(), dict)
        
        # Test token counting
        token_count = mock_llm.get_token_count("This is a test sentence.")
        assert isinstance(token_count, int)
        assert token_count > 0


class TestMockLLMProvider:
    """Test cases for the MockLLMProvider implementation."""
    
    def test_mock_llm_initialization(self):
        """Test MockLLMProvider initialization with various parameters."""
        # Test default initialization
        mock_llm = MockLLMProvider()
        assert mock_llm.get_model_name() == "mock-llm"
        
        # Test custom initialization
        custom_llm = MockLLMProvider(
            model="custom-mock",
            response_template="Custom: {prompt}",
            simulate_delay=0.05,
            fail_probability=0.1
        )
        assert custom_llm.get_model_name() == "custom-mock"
        assert custom_llm.response_template == "Custom: {prompt}"
        assert custom_llm.simulate_delay == 0.05
        assert custom_llm.fail_probability == 0.1
    
    @pytest.mark.asyncio
    async def test_mock_llm_generate(self):
        """Test MockLLMProvider text generation."""
        mock_llm = MockLLMProvider(
            response_template="Mock response to: {prompt}",
            simulate_delay=0.01
        )
        
        # Test basic generation
        response = await mock_llm.generate("Hello, world!")
        assert isinstance(response, str)
        assert "Mock response to: Hello, world!" in response
        
        # Test with different prompt
        response2 = await mock_llm.generate("Different prompt")
        assert "Different prompt" in response2
    
    @pytest.mark.asyncio
    async def test_mock_llm_generate_structured(self):
        """Test MockLLMProvider structured generation."""
        mock_llm = MockLLMProvider(
            structured_template={"answer": "mock answer", "confidence": 0.95},
            simulate_delay=0.01
        )
        
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"}
            }
        }
        
        # Test structured generation
        response = await mock_llm.generate_structured("Test prompt", schema)
        assert isinstance(response, dict)
        assert "answer" in response
        assert response["answer"] == "mock answer"
        assert "prompt_preview" in response
    
    @pytest.mark.asyncio
    async def test_mock_llm_failure_simulation(self):
        """Test MockLLMProvider failure simulation."""
        # Create mock with 100% failure rate
        failing_llm = MockLLMProvider(fail_probability=1.0)
        
        # Test that it raises LexoraError
        with pytest.raises(LexoraError) as exc_info:
            await failing_llm.generate("Test prompt")
        
        assert exc_info.value.error_code == ErrorCode.LLM_CONNECTION_FAILED
        
        # Test structured generation failure
        with pytest.raises(LexoraError) as exc_info:
            await failing_llm.generate_structured("Test prompt", {})
        
        assert exc_info.value.error_code == ErrorCode.LLM_INVALID_RESPONSE
    
    @pytest.mark.asyncio
    async def test_mock_llm_retry_mechanism(self):
        """Test retry mechanism with MockLLMProvider."""
        # Create mock with 50% failure rate
        unreliable_llm = MockLLMProvider(fail_probability=0.5, simulate_delay=0.01)
        
        # Test retry with sufficient attempts (should eventually succeed)
        response = await unreliable_llm.generate_with_retry(
            "Test prompt", 
            max_retries=10, 
            retry_delay=0.01
        )
        assert isinstance(response, str)
    
    @pytest.mark.asyncio
    async def test_mock_llm_batch_generation(self):
        """Test batch generation with MockLLMProvider."""
        mock_llm = MockLLMProvider(simulate_delay=0.01)
        
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        responses = await mock_llm.batch_generate(prompts, max_concurrent=2)
        
        assert len(responses) == len(prompts)
        assert all(isinstance(r, str) for r in responses)
        
        # Check that responses correspond to prompts
        for i, response in enumerate(responses):
            assert f"Prompt {i + 1}" in response


class TestLitellmProvider:
    """Test cases for the LitellmProvider implementation."""
    
    def test_litellm_provider_initialization(self):
        """Test LitellmProvider initialization."""
        # Test basic initialization
        provider = LitellmProvider("gpt-3.5-turbo")
        assert provider.get_model_name() == "gpt-3.5-turbo"
        assert provider.temperature == 0.7
        assert provider.max_tokens == 2000
        
        # Test custom initialization
        custom_provider = LitellmProvider(
            model="gpt-4",
            temperature=0.5,
            max_tokens=1500,
            timeout=30.0
        )
        assert custom_provider.temperature == 0.5
        assert custom_provider.max_tokens == 1500
        assert custom_provider.timeout == 30.0
    
    def test_litellm_provider_detection(self):
        """Test provider detection logic."""
        provider = LitellmProvider("gpt-3.5-turbo")
        
        # Test OpenAI detection
        assert provider._detect_provider("gpt-4") == "openai"
        assert provider._detect_provider("gpt-3.5-turbo") == "openai"
        
        # Test Anthropic detection
        assert provider._detect_provider("claude-3-sonnet") == "anthropic"
        assert provider._detect_provider("claude-3-opus") == "anthropic"
        
        # Test Azure detection
        assert provider._detect_provider("azure/gpt-35-turbo") == "azure"
        
        # Test unknown model (defaults to openai)
        assert provider._detect_provider("unknown-model") == "openai"
    
    def test_litellm_message_preparation(self):
        """Test message preparation for litellm."""
        provider = LitellmProvider("gpt-3.5-turbo")
        
        # Test user message only
        messages = provider._prepare_messages("Hello, world!")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, world!"
        
        # Test with system message
        messages = provider._prepare_messages(
            "Hello, world!", 
            "You are a helpful assistant"
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello, world!"
    
    def test_litellm_completion_kwargs_preparation(self):
        """Test completion kwargs preparation."""
        provider = LitellmProvider("gpt-3.5-turbo", temperature=0.5)
        
        # Test default kwargs
        kwargs = provider._prepare_completion_kwargs()
        assert kwargs["model"] == "gpt-3.5-turbo"
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 2000
        
        # Test with overrides
        kwargs = provider._prepare_completion_kwargs(
            temperature=0.8, 
            max_tokens=1000,
            custom_param="test"
        )
        assert kwargs["temperature"] == 0.8
        assert kwargs["max_tokens"] == 1000
        assert kwargs["custom_param"] == "test"
    
    def test_litellm_json_prompt_creation(self):
        """Test JSON prompt creation for structured output."""
        provider = LitellmProvider("gpt-3.5-turbo")
        
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "The answer"},
                "confidence": {"type": "number", "description": "Confidence score"}
            },
            "required": ["answer"]
        }
        
        json_prompt = provider._create_json_prompt("What is 2+2?", schema)
        
        assert "What is 2+2?" in json_prompt
        assert "JSON" in json_prompt
        assert "answer" in json_prompt
        assert "confidence" in json_prompt
        assert "required" in json_prompt or "answer" in json_prompt
    
    def test_litellm_schema_validation(self):
        """Test schema validation logic."""
        provider = LitellmProvider("gpt-3.5-turbo")
        
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"}
            },
            "required": ["answer"]
        }
        
        # Test valid data
        valid_data = {"answer": "4", "confidence": 0.95}
        assert provider._validate_against_schema(valid_data, schema)
        
        # Test missing required field
        invalid_data = {"confidence": 0.95}
        assert not provider._validate_against_schema(invalid_data, schema)
        
        # Test wrong type
        wrong_type_data = {"answer": 123, "confidence": 0.95}
        assert not provider._validate_against_schema(wrong_type_data, schema)
    
    def test_litellm_json_extraction(self):
        """Test JSON extraction from mixed text."""
        provider = LitellmProvider("gpt-3.5-turbo")
        
        # Test clean JSON extraction
        json_data = {"answer": "4", "confidence": 0.9}
        text_with_json = f"Here is the result: {json.dumps(json_data)} Hope this helps!"
        
        extracted = provider._extract_json_from_text(text_with_json)
        assert extracted is not None
        assert extracted["answer"] == "4"
        assert extracted["confidence"] == 0.9
        
        # Test no JSON in text
        no_json_text = "This is just plain text without any JSON."
        extracted = provider._extract_json_from_text(no_json_text)
        assert extracted is None
        
        # Test malformed JSON
        malformed_text = "Here is broken JSON: {answer: 4, confidence:} end"
        extracted = provider._extract_json_from_text(malformed_text)
        assert extracted is None
    
    def test_litellm_response_format_fixing(self):
        """Test response format fixing logic."""
        provider = LitellmProvider("gpt-3.5-turbo")
        
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"}
            },
            "required": ["answer"]
        }
        
        # Test fixing type issues
        data_with_wrong_types = {"answer": 123, "confidence": "0.95"}
        fixed_data = provider._fix_response_format(data_with_wrong_types, schema)
        
        assert fixed_data is not None
        assert isinstance(fixed_data["answer"], str)
        assert isinstance(fixed_data["confidence"], float)
        assert fixed_data["answer"] == "123"
        assert fixed_data["confidence"] == 0.95
    
    def test_litellm_configuration_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        provider = LitellmProvider("gpt-3.5-turbo", temperature=0.5, max_tokens=1000)
        provider.validate_config()  # Should not raise
        
        # Test invalid temperature
        with pytest.raises(LexoraError) as exc_info:
            LitellmProvider("gpt-3.5-turbo", temperature=3.0)
        assert exc_info.value.error_code == ErrorCode.INVALID_CONFIG
        
        # Test invalid max_tokens
        with pytest.raises(LexoraError) as exc_info:
            LitellmProvider("gpt-3.5-turbo", max_tokens=-100)
        assert exc_info.value.error_code == ErrorCode.INVALID_CONFIG
        
        # Test invalid timeout
        with pytest.raises(LexoraError) as exc_info:
            LitellmProvider("gpt-3.5-turbo", timeout=-1.0)
        assert exc_info.value.error_code == ErrorCode.INVALID_CONFIG
    
    def test_litellm_model_support_checking(self):
        """Test model support checking."""
        provider = LitellmProvider("gpt-3.5-turbo")
        
        # Test supported models
        assert provider.is_model_supported("gpt-4")
        assert provider.is_model_supported("claude-3-sonnet")
        assert provider.is_model_supported("azure/gpt-35-turbo")
        
        # Test getting supported models list
        supported_models = provider.get_supported_models()
        assert isinstance(supported_models, list)
        assert len(supported_models) > 0
        assert "gpt-3.5-turbo" in supported_models or any("gpt" in model for model in supported_models)
    
    @pytest.mark.asyncio
    async def test_litellm_generate_with_mock(self):
        """Test LitellmProvider generation with mocked litellm."""
        with patch('llm.litellm_provider.acompletion') as mock_completion:
            # Mock successful response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Mocked response"
            mock_completion.return_value = mock_response
            
            provider = LitellmProvider("gpt-3.5-turbo")
            response = await provider.generate("Test prompt")
            
            assert response == "Mocked response"
            mock_completion.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_litellm_generate_error_handling(self):
        """Test LitellmProvider error handling during generation."""
        with patch('llm.litellm_provider.acompletion') as mock_completion:
            provider = LitellmProvider("gpt-3.5-turbo")
            
            # Test rate limit error
            mock_completion.side_effect = Exception("rate limit exceeded")
            with pytest.raises(LexoraError) as exc_info:
                await provider.generate("Test prompt")
            assert exc_info.value.error_code == ErrorCode.LLM_RATE_LIMIT_EXCEEDED
            
            # Test authentication error
            mock_completion.side_effect = Exception("authentication failed")
            with pytest.raises(LexoraError) as exc_info:
                await provider.generate("Test prompt")
            assert exc_info.value.error_code == ErrorCode.LLM_AUTHENTICATION_FAILED
            
            # Test timeout error
            mock_completion.side_effect = Exception("timeout occurred")
            with pytest.raises(LexoraError) as exc_info:
                await provider.generate("Test prompt")
            assert exc_info.value.error_code == ErrorCode.LLM_TIMEOUT
            
            # Test model not found error
            mock_completion.side_effect = Exception("model not found")
            with pytest.raises(LexoraError) as exc_info:
                await provider.generate("Test prompt")
            assert exc_info.value.error_code == ErrorCode.LLM_MODEL_NOT_FOUND
            
            # Test quota exceeded error
            mock_completion.side_effect = Exception("quota exceeded")
            with pytest.raises(LexoraError) as exc_info:
                await provider.generate("Test prompt")
            assert exc_info.value.error_code == ErrorCode.LLM_QUOTA_EXCEEDED
            
            # Test generic error
            mock_completion.side_effect = Exception("generic error")
            with pytest.raises(LexoraError) as exc_info:
                await provider.generate("Test prompt")
            assert exc_info.value.error_code == ErrorCode.LLM_CONNECTION_FAILED
    
    @pytest.mark.asyncio
    async def test_litellm_generate_structured_with_mock(self):
        """Test LitellmProvider structured generation with mocked litellm."""
        with patch('llm.litellm_provider.acompletion') as mock_completion:
            # Mock successful JSON response
            json_response = '{"answer": "4", "confidence": 0.95}'
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = json_response
            mock_completion.return_value = mock_response
            
            provider = LitellmProvider("gpt-3.5-turbo")
            schema = {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "confidence": {"type": "number"}
                },
                "required": ["answer"]
            }
            
            response = await provider.generate_structured("What is 2+2?", schema)
            
            assert isinstance(response, dict)
            assert response["answer"] == "4"
            assert response["confidence"] == 0.95


class TestLLMUtilities:
    """Test cases for LLM utility functions."""
    
    def test_create_mock_llm(self):
        """Test create_mock_llm utility function."""
        mock_llm = create_mock_llm(
            model="test-mock",
            response_template="Test: {prompt}",
            simulate_delay=0.05
        )
        
        assert isinstance(mock_llm, MockLLMProvider)
        assert mock_llm.get_model_name() == "test-mock"
        assert mock_llm.response_template == "Test: {prompt}"
        assert mock_llm.simulate_delay == 0.05
    
    def test_create_litellm_provider(self):
        """Test create_litellm_provider utility function."""
        provider = create_litellm_provider(
            model="gpt-4",
            temperature=0.3,
            max_tokens=1500
        )
        
        assert isinstance(provider, LitellmProvider)
        assert provider.get_model_name() == "gpt-4"
        assert provider.temperature == 0.3
        assert provider.max_tokens == 1500
    
    def test_get_available_models(self):
        """Test get_available_models utility function."""
        models = get_available_models()
        
        assert isinstance(models, dict)
        assert len(models) > 0
        assert "openai" in models
        assert "anthropic" in models
        
        # Check that each provider has models
        for provider, model_list in models.items():
            assert isinstance(model_list, list)
            assert len(model_list) > 0
    
    def test_validate_llm_provider(self):
        """Test validate_llm_provider utility function."""
        # Test with valid provider
        mock_llm = MockLLMProvider("test-model")
        validate_llm_provider(mock_llm)  # Should not raise
        
        # Test with invalid provider (not inheriting from BaseLLM)
        class InvalidProvider:
            pass
        
        with pytest.raises(LexoraError) as exc_info:
            validate_llm_provider(InvalidProvider())
        assert exc_info.value.error_code == ErrorCode.INVALID_CONFIG
        
        # Test with provider missing required methods
        # Since we can't instantiate incomplete abstract classes,
        # we'll test with a mock that's missing methods
        mock_provider = Mock()
        mock_provider.generate = "not_callable"  # Not a callable
        mock_provider.generate_structured = Mock()  # This one is callable
        
        with pytest.raises(LexoraError) as exc_info:
            validate_llm_provider(mock_provider)
        assert exc_info.value.error_code == ErrorCode.INVALID_CONFIG


class TestRetryMechanisms:
    """Test cases for retry mechanisms and error recovery."""
    
    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff(self):
        """Test retry mechanism with exponential backoff."""
        # Create a mock that fails twice then succeeds
        call_count = 0
        
        async def failing_generate(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise LexoraError("Temporary failure", ErrorCode.LLM_CONNECTION_FAILED)
            return "Success after retries"
        
        mock_llm = MockLLMProvider("test-model")
        mock_llm.generate = failing_generate
        
        # Test successful retry
        response = await mock_llm.generate_with_retry(
            "Test prompt",
            max_retries=3,
            retry_delay=0.01,
            backoff_factor=2.0
        )
        
        assert response == "Success after retries"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_exhaustion(self):
        """Test retry mechanism when all attempts fail."""
        async def always_failing_generate(prompt, **kwargs):
            raise LexoraError("Persistent failure", ErrorCode.LLM_CONNECTION_FAILED)
        
        mock_llm = MockLLMProvider("test-model")
        mock_llm.generate = always_failing_generate
        
        # Test retry exhaustion
        with pytest.raises(LexoraError) as exc_info:
            await mock_llm.generate_with_retry(
                "Test prompt",
                max_retries=2,
                retry_delay=0.01
            )
        
        assert exc_info.value.error_code == ErrorCode.LLM_CONNECTION_FAILED
        assert "after 3 attempts" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_no_retry_on_auth_errors(self):
        """Test that certain errors don't trigger retries."""
        call_count = 0
        
        async def auth_failing_generate(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            raise LexoraError("Auth failed", ErrorCode.LLM_AUTHENTICATION_FAILED)
        
        mock_llm = MockLLMProvider("test-model")
        mock_llm.generate = auth_failing_generate
        
        # Test that auth errors don't retry
        with pytest.raises(LexoraError) as exc_info:
            await mock_llm.generate_with_retry(
                "Test prompt",
                max_retries=3,
                retry_delay=0.01
            )
        
        assert exc_info.value.error_code == ErrorCode.LLM_AUTHENTICATION_FAILED
        assert call_count == 1  # Should not have retried


class TestConcurrencyAndPerformance:
    """Test cases for concurrent operations and performance."""
    
    @pytest.mark.asyncio
    async def test_batch_generation_concurrency(self):
        """Test that batch generation handles concurrency correctly."""
        mock_llm = MockLLMProvider("test-model", simulate_delay=0.1)
        
        prompts = [f"Prompt {i}" for i in range(10)]
        
        # Test with different concurrency limits
        import time
        
        # Test with max_concurrent=1 (sequential)
        start_time = time.time()
        responses_sequential = await mock_llm.batch_generate(prompts, max_concurrent=1)
        sequential_time = time.time() - start_time
        
        # Test with max_concurrent=5 (concurrent)
        start_time = time.time()
        responses_concurrent = await mock_llm.batch_generate(prompts, max_concurrent=5)
        concurrent_time = time.time() - start_time
        
        # Concurrent should be faster
        assert concurrent_time < sequential_time
        assert len(responses_sequential) == len(responses_concurrent) == len(prompts)
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_isolation(self):
        """Test that concurrent requests don't interfere with each other."""
        mock_llm = MockLLMProvider("test-model", simulate_delay=0.05)
        
        # Create multiple concurrent requests
        tasks = []
        for i in range(5):
            task = mock_llm.generate(f"Concurrent prompt {i}")
            tasks.append(task)
        
        # Wait for all to complete
        responses = await asyncio.gather(*tasks)
        
        # Check that each response corresponds to its prompt
        assert len(responses) == 5
        for i, response in enumerate(responses):
            assert f"Concurrent prompt {i}" in response


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])