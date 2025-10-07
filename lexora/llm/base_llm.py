"""
Base LLM interface for the Lexora Agentic RAG SDK.

This module provides the abstract base class that all LLM providers must implement,
ensuring consistent interfaces across different language model providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import asyncio
import time

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from ..exceptions import LexoraError, ErrorCode, create_llm_error
from ..utils.logging import get_logger


class BaseLLM(ABC):
    """
    Abstract base class for LLM providers.
    
    This class defines the interface that all LLM providers must implement
    to ensure consistent behavior across different language model services.
    """
    
    def __init__(self, model: str, **kwargs):
        """
        Initialize the LLM provider.
        
        Args:
            model: Name of the model to use
            **kwargs: Provider-specific configuration options
        """
        self.model = model
        self.config = kwargs
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize token counter if available
        self._tokenizer = None
        if TIKTOKEN_AVAILABLE:
            try:
                # Try to get encoding for the model
                if "gpt" in model.lower():
                    self._tokenizer = tiktoken.encoding_for_model(model)
                else:
                    # Fall back to cl100k_base for other models
                    self._tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                # If tiktoken fails, we'll use a simple approximation
                self._tokenizer = None
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text response from prompt.
        
        Args:
            prompt: Input prompt for the model
            **kwargs: Generation parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated text response
            
        Raises:
            LexoraError: If generation fails
        """
        pass
    
    @abstractmethod
    async def generate_structured(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate structured response matching the provided schema.
        
        Args:
            prompt: Input prompt for the model
            schema: JSON schema that the response should match
            
        Returns:
            Structured response as a dictionary
            
        Raises:
            LexoraError: If generation fails or response doesn't match schema
        """
        pass
    
    def get_token_count(self, text: str) -> int:
        """
        Count tokens in the given text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Approximate number of tokens
        """
        if self._tokenizer:
            try:
                return len(self._tokenizer.encode(text))
            except Exception:
                # Fall back to approximation if tokenizer fails
                pass
        
        # Simple approximation: ~4 characters per token
        return len(text) // 4
    
    def get_model_name(self) -> str:
        """
        Get the name of the model being used.
        
        Returns:
            Model name
        """
        return self.model
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()
    
    async def generate_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
        **kwargs
    ) -> str:
        """
        Generate text with automatic retry logic.
        
        Args:
            prompt: Input prompt for the model
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries in seconds
            backoff_factor: Multiplier for delay after each retry
            **kwargs: Generation parameters
            
        Returns:
            Generated text response
            
        Raises:
            LexoraError: If all retry attempts fail
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                result = await self.generate(prompt, **kwargs)
                duration = time.time() - start_time
                
                self.logger.log_llm_request(
                    provider=self.__class__.__name__,
                    model=self.model,
                    prompt_length=len(prompt),
                    response_length=len(result),
                    duration=duration
                )
                
                return result
                
            except LexoraError as e:
                last_error = e
                
                # Don't retry on certain error types
                if e.error_code in (ErrorCode.LLM_AUTHENTICATION_FAILED, ErrorCode.LLM_MODEL_NOT_FOUND):
                    raise e
                
                if attempt < max_retries:
                    delay = retry_delay * (backoff_factor ** attempt)
                    self.logger.warning(
                        f"LLM request failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.1f}s",
                        error=str(e),
                        attempt=attempt + 1,
                        max_retries=max_retries + 1
                    )
                    await asyncio.sleep(delay)
                else:
                    self.logger.log_llm_request(
                        provider=self.__class__.__name__,
                        model=self.model,
                        prompt_length=len(prompt),
                        error=e
                    )
        
        # If we get here, all retries failed
        raise create_llm_error(
            f"LLM request failed after {max_retries + 1} attempts: {str(last_error)}",
            self.__class__.__name__,
            self.model,
            ErrorCode.LLM_CONNECTION_FAILED,
            last_error
        )
    
    async def generate_structured_with_retry(
        self,
        prompt: str,
        schema: Dict[str, Any],
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured response with automatic retry logic.
        
        Args:
            prompt: Input prompt for the model
            schema: JSON schema for the response
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries in seconds
            backoff_factor: Multiplier for delay after each retry
            **kwargs: Generation parameters
            
        Returns:
            Structured response as a dictionary
            
        Raises:
            LexoraError: If all retry attempts fail
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                result = await self.generate_structured(prompt, schema, **kwargs)
                duration = time.time() - start_time
                
                self.logger.log_llm_request(
                    provider=self.__class__.__name__,
                    model=self.model,
                    prompt_length=len(prompt),
                    response_length=len(str(result)),
                    duration=duration
                )
                
                return result
                
            except LexoraError as e:
                last_error = e
                
                # Don't retry on certain error types
                if e.error_code in (ErrorCode.LLM_AUTHENTICATION_FAILED, ErrorCode.LLM_MODEL_NOT_FOUND):
                    raise e
                
                if attempt < max_retries:
                    delay = retry_delay * (backoff_factor ** attempt)
                    self.logger.warning(
                        f"Structured LLM request failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.1f}s",
                        error=str(e),
                        attempt=attempt + 1,
                        max_retries=max_retries + 1
                    )
                    await asyncio.sleep(delay)
                else:
                    self.logger.log_llm_request(
                        provider=self.__class__.__name__,
                        model=self.model,
                        prompt_length=len(prompt),
                        error=e
                    )
        
        # If we get here, all retries failed
        raise create_llm_error(
            f"Structured LLM request failed after {max_retries + 1} attempts: {str(last_error)}",
            self.__class__.__name__,
            self.model,
            ErrorCode.LLM_CONNECTION_FAILED,
            last_error
        )
    
    async def batch_generate(
        self,
        prompts: List[str],
        max_concurrent: int = 5,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for multiple prompts concurrently.
        
        Args:
            prompts: List of prompts to process
            max_concurrent: Maximum number of concurrent requests
            **kwargs: Generation parameters
            
        Returns:
            List of generated responses in the same order as prompts
            
        Raises:
            LexoraError: If any generation fails
        """
        if not prompts:
            return []
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_single(prompt: str) -> str:
            async with semaphore:
                return await self.generate_with_retry(prompt, **kwargs)
        
        tasks = [generate_single(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    def validate_config(self) -> None:
        """
        Validate the provider configuration.
        
        Raises:
            LexoraError: If configuration is invalid
        """
        if not self.model:
            raise create_llm_error(
                "Model name is required",
                self.__class__.__name__,
                self.model,
                ErrorCode.INVALID_CONFIG
            )
    
    def __str__(self) -> str:
        """Return string representation of the LLM provider."""
        return f"{self.__class__.__name__}(model={self.model})"
    
    def __repr__(self) -> str:
        """Return detailed representation of the LLM provider."""
        return f"{self.__class__.__name__}(model='{self.model}', config={self.config})"


class MockLLMProvider(BaseLLM):
    """
    Mock LLM provider for testing and development.
    
    This provider generates deterministic responses for testing purposes
    without making actual API calls.
    """
    
    def __init__(self, model: str = "mock-llm", **kwargs):
        """
        Initialize mock LLM provider.
        
        Args:
            model: Mock model name
            **kwargs: Additional configuration (ignored)
        """
        super().__init__(model, **kwargs)
        self.response_template = kwargs.get('response_template', "Mock response to: {prompt}")
        self.structured_template = kwargs.get('structured_template', {"response": "Mock structured response"})
        self.simulate_delay = kwargs.get('simulate_delay', 0.1)
        self.fail_probability = kwargs.get('fail_probability', 0.0)
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate mock text response.
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters (ignored)
            
        Returns:
            Mock response text
            
        Raises:
            LexoraError: If configured to simulate failures
        """
        # Simulate processing delay
        if self.simulate_delay > 0:
            await asyncio.sleep(self.simulate_delay)
        
        # Simulate random failures if configured
        if self.fail_probability > 0:
            import random
            if random.random() < self.fail_probability:
                raise create_llm_error(
                    "Simulated LLM failure",
                    self.__class__.__name__,
                    self.model,
                    ErrorCode.LLM_CONNECTION_FAILED
                )
        
        # Generate deterministic response
        if isinstance(self.response_template, str):
            return self.response_template.format(prompt=prompt[:100])
        else:
            return f"Mock response to: {prompt[:100]}"
    
    async def generate_structured(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate mock structured response.
        
        Args:
            prompt: Input prompt
            schema: Expected response schema (used to shape response)
            
        Returns:
            Mock structured response
            
        Raises:
            LexoraError: If configured to simulate failures
        """
        # Simulate processing delay
        if self.simulate_delay > 0:
            await asyncio.sleep(self.simulate_delay)
        
        # Simulate random failures if configured
        if self.fail_probability > 0:
            import random
            if random.random() < self.fail_probability:
                raise create_llm_error(
                    "Simulated structured LLM failure",
                    self.__class__.__name__,
                    self.model,
                    ErrorCode.LLM_INVALID_RESPONSE
                )
        
        # Generate response based on schema
        if isinstance(self.structured_template, dict):
            response = self.structured_template.copy()
            response["prompt_preview"] = prompt[:50]
            return response
        else:
            return {
                "response": "Mock structured response",
                "prompt_preview": prompt[:50],
                "schema_properties": list(schema.get("properties", {}).keys()) if "properties" in schema else []
            }


# Utility functions for LLM management

def create_mock_llm(
    model: str = "mock-llm",
    response_template: Optional[str] = None,
    structured_template: Optional[Dict[str, Any]] = None,
    simulate_delay: float = 0.1,
    fail_probability: float = 0.0
) -> MockLLMProvider:
    """
    Create a mock LLM provider for testing.
    
    Args:
        model: Mock model name
        response_template: Template for text responses
        structured_template: Template for structured responses
        simulate_delay: Delay to simulate processing time
        fail_probability: Probability of simulating failures (0.0-1.0)
        
    Returns:
        Configured mock LLM provider
    """
    return MockLLMProvider(
        model=model,
        response_template=response_template,
        structured_template=structured_template,
        simulate_delay=simulate_delay,
        fail_probability=fail_probability
    )


def validate_llm_provider(provider: BaseLLM) -> None:
    """
    Validate that an object implements the BaseLLM interface correctly.
    
    Args:
        provider: LLM provider to validate
        
    Raises:
        LexoraError: If provider doesn't implement the interface correctly
    """
    if not isinstance(provider, BaseLLM):
        raise LexoraError(
            f"Provider must inherit from BaseLLM, got {type(provider).__name__}",
            ErrorCode.INVALID_CONFIG
        )
    
    # Check that required methods are implemented
    required_methods = ['generate', 'generate_structured']
    for method_name in required_methods:
        if not hasattr(provider, method_name):
            raise LexoraError(
                f"Provider missing required method: {method_name}",
                ErrorCode.INVALID_CONFIG
            )
        
        method = getattr(provider, method_name)
        if not callable(method):
            raise LexoraError(
                f"Provider method {method_name} is not callable",
                ErrorCode.INVALID_CONFIG
            )
    
    # Validate configuration
    provider.validate_config()