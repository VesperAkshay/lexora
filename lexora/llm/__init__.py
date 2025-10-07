"""
LLM module - Language model abstraction layer.

This module provides:
- BaseLLM: Abstract base class for LLM providers
- LitellmProvider: Integration with litellm for multi-provider support
"""

from .base_llm import BaseLLM, MockLLMProvider, create_mock_llm, validate_llm_provider
from .litellm_provider import LitellmProvider, create_litellm_provider, get_available_models

__all__ = [
    "BaseLLM",
    "MockLLMProvider",
    "LitellmProvider",
    "create_mock_llm",
    "create_litellm_provider",
    "validate_llm_provider",
    "get_available_models",
]