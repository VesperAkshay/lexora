"""
Litellm provider for the Lexora Agentic RAG SDK.

This module provides an LLM provider that uses litellm for multi-provider support,
allowing access to various LLM services through a unified interface.
"""

import json
import re
import asyncio
from typing import Any, Dict, List, Optional, Union
import time

try:
    import litellm
    from litellm import acompletion, completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

from .base_llm import BaseLLM
from ..exceptions import LexoraError, ErrorCode, create_llm_error
from ..utils.logging import get_logger


class LitellmProvider(BaseLLM):
    """
    LLM provider using litellm for multi-provider support.
    
    This provider supports various LLM services including OpenAI, Anthropic, Cohere,
    and many others through the litellm library.
    """
    
    # Supported providers and their model prefixes
    SUPPORTED_PROVIDERS = {
        "openai": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"],
        "anthropic": ["claude-3-sonnet", "claude-3-opus", "claude-3-haiku"],
        "cohere": ["command", "command-light"],
        "azure": ["azure/"],
        "bedrock": ["bedrock/"],
        "vertex_ai": ["vertex_ai/"],
        "huggingface": ["huggingface/"],
        "ollama": ["ollama/"],
        "together_ai": ["together_ai/"],
        "anyscale": ["anyscale/"],
        "replicate": ["replicate/"],
    }
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: float = 60.0,
        **kwargs
    ):
        """
        Initialize Litellm provider.
        
        Args:
            model: Model name (e.g., "gpt-3.5-turbo", "claude-3-sonnet")
            api_key: API key for the provider
            api_base: Custom API base URL
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            **kwargs: Additional litellm parameters
            
        Raises:
            LexoraError: If litellm is not available or configuration is invalid
        """
        if not LITELLM_AVAILABLE:
            raise create_llm_error(
                "litellm library not available. Install with: pip install litellm",
                "litellm",
                model,
                ErrorCode.LLM_CONNECTION_FAILED
            )
        
        super().__init__(model, **kwargs)
        
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Store additional litellm parameters
        self.litellm_kwargs = kwargs
        
        # Detect and store provider (but don't mutate global state)
        self.provider = self._detect_provider(model)
        
        # Validate configuration
        self.validate_config()
    
    def _detect_provider(self, model: str) -> str:
        """
        Detect the provider based on model name.
        
        Args:
            model: Model name
            
        Returns:
            Provider name
        """
        model_lower = model.lower()
        
        for provider, model_patterns in self.SUPPORTED_PROVIDERS.items():
            for pattern in model_patterns:
                if pattern.endswith("/"):
                    # Prefix match for providers like azure/, bedrock/
                    if model_lower.startswith(pattern.lower()):
                        return provider
                else:
                    # Token/segment match with boundaries to avoid false positives
                    # Pattern must appear at start/end or be surrounded by separators (-, _, /)
                    pattern_lower = pattern.lower()
                    # Create regex pattern that enforces word boundaries or separators
                    # Matches pattern at start, end, or surrounded by -, _, /
                    regex_pattern = r'(?:^|[-_/])' + re.escape(pattern_lower) + r'(?:[-_/]|$)'
                    if re.search(regex_pattern, model_lower):
                        return provider
        
        # Default to openai for unknown models
        return "openai"
    
    def _prepare_messages(self, prompt: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Prepare messages in the format expected by litellm.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    def _prepare_completion_kwargs(self, **kwargs) -> Dict[str, Any]:
        """
        Prepare keyword arguments for litellm completion.
        
        Args:
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of completion parameters
        """
        completion_kwargs = {
            "model": self.model,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "timeout": kwargs.get("timeout", self.timeout),
        }
        
        # Add API credentials per-call instead of using global state
        if self.api_key:
            completion_kwargs["api_key"] = self.api_key
        
        if self.api_base:
            completion_kwargs["api_base"] = self.api_base
        
        # Add any additional litellm parameters
        completion_kwargs.update(self.litellm_kwargs)
        
        # Override with any kwargs passed to the method
        for key, value in kwargs.items():
            if key not in ["system_prompt"]:  # Exclude our custom parameters
                completion_kwargs[key] = value
        
        return completion_kwargs
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text response from prompt using litellm.
        
        Args:
            prompt: Input prompt for the model
            **kwargs: Generation parameters
            
        Returns:
            Generated text response
            
        Raises:
            LexoraError: If generation fails
        """
        try:
            # Prepare messages
            system_prompt = kwargs.get("system_prompt")
            messages = self._prepare_messages(prompt, system_prompt)
            
            # Prepare completion parameters
            completion_kwargs = self._prepare_completion_kwargs(**kwargs)
            completion_kwargs["messages"] = messages
            
            # Make async completion call
            response = await acompletion(**completion_kwargs)
            
            # Extract content from response
            if response and response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content:
                    return content.strip()
                else:
                    raise create_llm_error(
                        "Empty response from LLM",
                        "litellm",
                        self.model,
                        ErrorCode.LLM_INVALID_RESPONSE
                    )
            else:
                raise create_llm_error(
                    "Invalid response structure from LLM",
                    "litellm",
                    self.model,
                    ErrorCode.LLM_INVALID_RESPONSE
                )
        
        except Exception as e:
            # Robust exception handling with structured error detection
            
            # Step 1: Check for known litellm exception classes
            # Try to detect provider-specific exceptions if available
            exception_type = type(e).__name__
            
            # Step 2: Inspect structured attributes (status_code, error_code)
            status_code = None
            if hasattr(e, 'status_code'):
                status_code = e.status_code
            elif hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                status_code = e.response.status_code
            
            # Map status codes to error codes
            if status_code:
                if status_code == 429:
                    raise create_llm_error(
                        f"Rate limit exceeded (HTTP {status_code}): {str(e)}",
                        "litellm",
                        self.model,
                        ErrorCode.LLM_RATE_LIMIT_EXCEEDED,
                        e
                    )
                elif status_code in (401, 403):
                    raise create_llm_error(
                        f"Authentication failed (HTTP {status_code}): {str(e)}",
                        "litellm",
                        self.model,
                        ErrorCode.LLM_AUTHENTICATION_FAILED,
                        e
                    )
                elif status_code == 404:
                    raise create_llm_error(
                        f"Model not found (HTTP {status_code}): {str(e)}",
                        "litellm",
                        self.model,
                        ErrorCode.LLM_MODEL_NOT_FOUND,
                        e
                    )
                elif status_code in (408, 504):
                    raise create_llm_error(
                        f"Request timeout (HTTP {status_code}): {str(e)}",
                        "litellm",
                        self.model,
                        ErrorCode.LLM_TIMEOUT,
                        e
                    )
                elif status_code == 402:
                    raise create_llm_error(
                        f"Quota exceeded (HTTP {status_code}): {str(e)}",
                        "litellm",
                        self.model,
                        ErrorCode.LLM_QUOTA_EXCEEDED,
                        e
                    )
            
            # Step 3: Check for error_code attribute
            if hasattr(e, 'error_code'):
                error_code = str(e.error_code).lower()
                if 'rate_limit' in error_code or 'too_many_requests' in error_code:
                    raise create_llm_error(
                        f"Rate limit exceeded: {str(e)}",
                        "litellm",
                        self.model,
                        ErrorCode.LLM_RATE_LIMIT_EXCEEDED,
                        e
                    )
                elif 'auth' in error_code or 'unauthorized' in error_code:
                    raise create_llm_error(
                        f"Authentication failed: {str(e)}",
                        "litellm",
                        self.model,
                        ErrorCode.LLM_AUTHENTICATION_FAILED,
                        e
                    )
                elif 'quota' in error_code or 'insufficient_quota' in error_code:
                    raise create_llm_error(
                        f"Quota exceeded: {str(e)}",
                        "litellm",
                        self.model,
                        ErrorCode.LLM_QUOTA_EXCEEDED,
                        e
                    )
            
            # Step 4: Fall back to limited string matching as last resort
            error_message = str(e).lower()
            
            if "rate limit" in error_message or "too many requests" in error_message:
                raise create_llm_error(
                    f"Rate limit exceeded: {str(e)}",
                    "litellm",
                    self.model,
                    ErrorCode.LLM_RATE_LIMIT_EXCEEDED,
                    e
                )
            elif "authentication" in error_message or "unauthorized" in error_message or "invalid api key" in error_message:
                raise create_llm_error(
                    f"Authentication failed: {str(e)}",
                    "litellm",
                    self.model,
                    ErrorCode.LLM_AUTHENTICATION_FAILED,
                    e
                )
            elif "timeout" in error_message or "timed out" in error_message:
                raise create_llm_error(
                    f"Request timeout: {str(e)}",
                    "litellm",
                    self.model,
                    ErrorCode.LLM_TIMEOUT,
                    e
                )
            elif "model not found" in error_message or "model does not exist" in error_message:
                raise create_llm_error(
                    f"Model not found: {str(e)}",
                    "litellm",
                    self.model,
                    ErrorCode.LLM_MODEL_NOT_FOUND,
                    e
                )
            elif "quota" in error_message or "billing" in error_message or "insufficient_quota" in error_message:
                raise create_llm_error(
                    f"Quota exceeded: {str(e)}",
                    "litellm",
                    self.model,
                    ErrorCode.LLM_QUOTA_EXCEEDED,
                    e
                )
            else:
                # Unknown error - use generic connection failed
                raise create_llm_error(
                    f"LLM generation failed: {str(e)}",
                    "litellm",
                    self.model,
                    ErrorCode.LLM_CONNECTION_FAILED,
                    e
                )
    
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
        # Enhance prompt to request JSON output
        json_prompt = self._create_json_prompt(prompt, schema)
        
        try:
            # Try to use JSON mode if supported by the model
            kwargs = {"response_format": {"type": "json_object"}}
            
            # For models that don't support JSON mode, we'll rely on prompt engineering
            try:
                response_text = await self.generate(json_prompt, **kwargs)
            except Exception:
                # Fall back to regular generation if JSON mode fails
                response_text = await self.generate(json_prompt)
            
            # Parse JSON response
            try:
                response_data = json.loads(response_text)
                
                # Basic schema validation
                if self._validate_against_schema(response_data, schema):
                    return response_data
                else:
                    # Try to fix common issues
                    fixed_response = self._fix_response_format(response_data, schema)
                    if fixed_response:
                        return fixed_response
                    
                    raise create_llm_error(
                        f"Response doesn't match schema: {response_text[:200]}",
                        "litellm",
                        self.model,
                        ErrorCode.LLM_INVALID_RESPONSE
                    )
            
            except json.JSONDecodeError as e:
                # Try to extract JSON from the response
                extracted_json = self._extract_json_from_text(response_text)
                if extracted_json:
                    # Validate extracted JSON against schema before returning
                    if self._validate_against_schema(extracted_json, schema):
                        return extracted_json
                    else:
                        # Try to fix common issues in extracted JSON
                        fixed_response = self._fix_response_format(extracted_json, schema)
                        if fixed_response:
                            return fixed_response
                        
                        # Validation failed - raise error instead of returning invalid data
                        raise create_llm_error(
                            f"Extracted JSON doesn't match schema. Response: {response_text[:200]}",
                            "litellm",
                            self.model,
                            ErrorCode.LLM_INVALID_RESPONSE
                        )
                
                raise create_llm_error(
                    f"Invalid JSON response: {str(e)}. Response: {response_text[:200]}",
                    "litellm",
                    self.model,
                    ErrorCode.LLM_INVALID_RESPONSE,
                    e
                )
        
        except LexoraError:
            # Re-raise LexoraErrors as-is
            raise
        except Exception as e:
            raise create_llm_error(
                f"Structured generation failed: {str(e)}",
                "litellm",
                self.model,
                ErrorCode.LLM_CONNECTION_FAILED,
                e
            )
    
    def _create_json_prompt(self, prompt: str, schema: Dict[str, Any]) -> str:
        """
        Create a prompt that encourages JSON output matching the schema.
        
        Args:
            prompt: Original prompt
            schema: JSON schema
            
        Returns:
            Enhanced prompt for JSON generation
        """
        schema_description = self._describe_schema(schema)
        
        json_prompt = f"""{prompt}

Please respond with a valid JSON object that matches the following schema:

{json.dumps(schema, indent=2)}

{schema_description}

Respond only with the JSON object, no additional text or formatting."""
        
        return json_prompt
    
    def _describe_schema(self, schema: Dict[str, Any]) -> str:
        """
        Create a human-readable description of the schema.
        
        Args:
            schema: JSON schema
            
        Returns:
            Schema description
        """
        if "properties" in schema:
            properties = schema["properties"]
            required = schema.get("required", [])
            
            descriptions = []
            for prop, prop_schema in properties.items():
                prop_type = prop_schema.get("type", "any")
                is_required = prop in required
                req_text = " (required)" if is_required else " (optional)"
                
                desc = prop_schema.get("description", "")
                if desc:
                    descriptions.append(f"- {prop} ({prop_type}){req_text}: {desc}")
                else:
                    descriptions.append(f"- {prop} ({prop_type}){req_text}")
            
            return "The JSON object should include:\n" + "\n".join(descriptions)
        
        return "Please ensure the response is valid JSON."
    
    def _validate_against_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        Basic validation of data against schema.
        
        Args:
            data: Data to validate
            schema: Schema to validate against
            
        Returns:
            True if data matches schema (basic validation)
        """
        if not isinstance(data, dict):
            return False
        
        # Check required properties
        required = schema.get("required", [])
        for prop in required:
            if prop not in data:
                return False
        
        # Check property types (basic)
        properties = schema.get("properties", {})
        for prop, value in data.items():
            if prop in properties:
                expected_type = properties[prop].get("type")
                if expected_type and not self._check_type(value, expected_type):
                    return False
        
        return True
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """
        Check if value matches expected JSON schema type.
        
        Args:
            value: Value to check
            expected_type: Expected type string
            
        Returns:
            True if type matches
        """
        type_mapping = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True  # Unknown type, assume valid
    
    def _fix_response_format(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Try to fix common formatting issues in the response.
        
        Args:
            data: Response data
            schema: Expected schema
            
        Returns:
            Fixed data or None if unfixable
        """
        # This is a basic implementation - could be enhanced
        fixed_data = data.copy()
        
        # Try to convert types if needed
        properties = schema.get("properties", {})
        for prop, prop_schema in properties.items():
            if prop in fixed_data:
                expected_type = prop_schema.get("type")
                value = fixed_data[prop]
                
                if expected_type == "string" and not isinstance(value, str):
                    fixed_data[prop] = str(value)
                elif expected_type == "integer" and isinstance(value, (str, float)):
                    try:
                        fixed_data[prop] = int(float(value))
                    except (ValueError, TypeError):
                        pass
                elif expected_type == "number" and isinstance(value, str):
                    try:
                        fixed_data[prop] = float(value)
                    except (ValueError, TypeError):
                        pass
        
        # Validate the fixed data
        if self._validate_against_schema(fixed_data, schema):
            return fixed_data
        
        return None
    
    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Try to extract JSON from text that might contain additional content.
        
        Args:
            text: Text that might contain JSON
            
        Returns:
            Extracted JSON data or None
        """
        # Use brace-counting parser to handle nested objects and string literals
        i = 0
        while i < len(text):
            # Find the next opening brace
            if text[i] != '{':
                i += 1
                continue
            
            # Start tracking braces from this position
            start_pos = i
            brace_count = 0
            in_string = False
            escape_next = False
            
            j = i
            while j < len(text):
                char = text[j]
                
                # Handle escape sequences
                if escape_next:
                    escape_next = False
                    j += 1
                    continue
                
                if char == '\\':
                    escape_next = True
                    j += 1
                    continue
                
                # Handle string literals
                if char == '"':
                    in_string = not in_string
                    j += 1
                    continue
                
                # Only count braces outside of strings
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        
                        # Found balanced JSON object
                        if brace_count == 0:
                            json_text = text[start_pos:j + 1]
                            try:
                                return json.loads(json_text)
                            except json.JSONDecodeError:
                                # Invalid JSON, continue searching
                                break
                
                j += 1
            
            # Move to next character to search for another opening brace
            i += 1
        
        # No valid JSON found
        return None
    
    def validate_config(self) -> None:
        """
        Validate the provider configuration.
        
        Raises:
            LexoraError: If configuration is invalid
        """
        super().validate_config()
        
        # Validate temperature
        if not 0.0 <= self.temperature <= 2.0:
            raise create_llm_error(
                f"Temperature must be between 0.0 and 2.0, got {self.temperature}",
                "litellm",
                self.model,
                ErrorCode.INVALID_CONFIG
            )
        
        # Validate max_tokens
        if self.max_tokens <= 0:
            raise create_llm_error(
                f"Max tokens must be positive, got {self.max_tokens}",
                "litellm",
                self.model,
                ErrorCode.INVALID_CONFIG
            )
        
        # Validate timeout
        if self.timeout <= 0:
            raise create_llm_error(
                f"Timeout must be positive, got {self.timeout}",
                "litellm",
                self.model,
                ErrorCode.INVALID_CONFIG
            )
    
    def get_supported_models(self) -> List[str]:
        """
        Get list of supported model names.
        
        Returns:
            List of supported model names
        """
        models = []
        for provider_models in self.SUPPORTED_PROVIDERS.values():
            models.extend(provider_models)
        return models
    
    def is_model_supported(self, model: str) -> bool:
        """
        Check if a model is supported.
        
        Args:
            model: Model name to check
            
        Returns:
            True if model is supported, False otherwise
        """
        # Normalize model name to lowercase for comparison
        model_lower = model.lower()
        
        # Iterate through all supported providers and their patterns
        for provider_models in self.SUPPORTED_PROVIDERS.values():
            for pattern in provider_models:
                # Check if pattern ends with "/" (prefix match)
                if pattern.endswith("/"):
                    # Prefix match for providers like azure/, bedrock/
                    if model_lower.startswith(pattern.lower()):
                        return True
                else:
                    # Substring match for specific model names
                    if pattern.lower() in model_lower:
                        return True
        
        # No pattern matched
        return False


# Convenience functions

def create_litellm_provider(
    model: str,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    **kwargs
) -> LitellmProvider:
    """
    Create a Litellm provider with common configuration.
    
    Args:
        model: Model name
        api_key: API key for the provider
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        **kwargs: Additional configuration
        
    Returns:
        Configured Litellm provider
    """
    return LitellmProvider(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )


def get_available_models() -> Dict[str, List[str]]:
    """
    Get dictionary of available models by provider.
    
    Returns:
        Dictionary mapping provider names to model lists
    """
    return LitellmProvider.SUPPORTED_PROVIDERS.copy()