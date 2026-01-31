"""
LLM provider abstraction layer supporting multiple providers.

This module provides a unified interface for interacting with different LLM providers
including OpenAI, Anthropic, and local Ollama models. Each provider implements the
same interface for consistent evaluation across platforms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator, Any
import time
import asyncio
from functools import wraps
import logging

import aiohttp
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# Import tiktoken for token counting
try:
    import tiktoken
except ImportError:
    tiktoken = None

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """
    Unified response format from any LLM provider.

    Attributes:
        content: The generated text response
        model: Model identifier used for generation
        provider: Provider name (openai, anthropic, ollama)
        input_tokens: Number of tokens in the input prompt
        output_tokens: Number of tokens in the generated response
        total_tokens: Total tokens used (input + output)
        latency_ms: Total request latency in milliseconds
        time_to_first_token_ms: Time to first token in milliseconds (None if not streaming)
        cost_usd: Estimated cost in USD
        raw_response: Raw response dict from the provider
        finish_reason: Reason for completion (length, stop, etc.)
        metadata: Additional provider-specific metadata
    """

    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    time_to_first_token_ms: Optional[float] = None
    cost_usd: float = 0.0
    raw_response: dict = field(default_factory=dict)
    finish_reason: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert response to dictionary for serialization."""
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "cost_usd": self.cost_usd,
            "finish_reason": self.finish_reason,
            "metadata": self.metadata,
        }


def with_retry(max_retries: int = 3, base_wait: float = 1.0):
    """
    Decorator for adding retry logic with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_wait: Base wait time in seconds (exponential)
    """

    def decorator(func):
        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=base_wait, max=60),
            retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
            reraise=True,
        )
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        return wrapper

    return decorator


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All provider implementations must inherit from this class and implement
    the required methods for text generation, streaming, token counting,
    and cost estimation.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        """
        Initialize the LLM provider.

        Args:
            model: Model identifier
            api_key: API key for authentication (if required)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
        """
        self.model = model
        self.api_key = api_key or ""
        self.timeout = timeout
        self.max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the provider name.

        Returns:
            Provider identifier string
        """
        pass

    @property
    def session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a non-streaming response.

        Args:
            prompt: User prompt text
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with generated content and metadata
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response.

        Args:
            prompt: User prompt text
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Yields:
            Chunks of generated text as they arrive
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        pass

    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost in USD.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class OpenAIProvider(LLMProvider):
    """
    OpenAI API provider supporting GPT models.

    Pricing (per 1K tokens as of 2024):
    - gpt-4o: $0.005 input, $0.015 output
    - gpt-4o-mini: $0.00015 input, $0.0006 output
    - gpt-4-turbo: $0.01 input, $0.03 output
    - gpt-3.5-turbo: $0.0005 input, $0.0015 output
    """

    PRICING: dict[str, dict[str, float]] = {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-3.5-turbo-16k": {"input": 0.0005, "output": 0.0015},
    }

    BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        """
        Initialize OpenAI provider.

        Args:
            model: Model identifier (default: gpt-4o-mini)
            api_key: OpenAI API key
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        super().__init__(model, api_key, timeout, max_retries)

        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        # Initialize tokenizer
        self._tokenizer = None

    @property
    def name(self) -> str:
        """Get provider name."""
        return "openai"

    @property
    def tokenizer(self):
        """Lazy-load tiktoken tokenizer."""
        if self._tokenizer is None:
            if tiktoken is None:
                raise ImportError("tiktoken is required for token counting")
            try:
                self._tokenizer = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # Fall back to cl100k_base for unknown models
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
        return self._tokenizer

    @with_retry(max_retries=3)
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response using OpenAI API."""
        start_time = time.perf_counter()

        # Build request payload
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Make request
        url = f"{self.BASE_URL}/chat/completions"
        async with self.session.post(url, json=payload, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()

        # Parse response
        choice = data["choices"][0]
        content = choice["message"]["content"]
        latency = (time.perf_counter() - start_time) * 1000

        # Token counts
        input_tokens = data.get("usage", {}).get("prompt_tokens", 0)
        output_tokens = data.get("usage", {}).get("completion_tokens", 0)
        total_tokens = input_tokens + output_tokens

        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency,
            cost_usd=self.estimate_cost(input_tokens, output_tokens),
            raw_response=data,
            finish_reason=choice.get("finish_reason"),
        )

    @with_retry(max_retries=3)
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response using OpenAI API."""
        start_time = time.perf_counter()
        first_token_time = None
        full_content = ""
        buffer = []  # Buffer to accumulate chunks

        # Build request payload
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            **kwargs,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.BASE_URL}/chat/completions"

        try:
            async with self.session.post(url, json=payload, headers=headers) as response:
                response.raise_for_status()

                async for line in response.content:
                    line = line.decode("utf-8").strip()

                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Remove "data: " prefix

                    if data_str == "[DONE]":
                        break

                    try:
                        import json

                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        content_chunk = delta.get("content", "")

                        if content_chunk:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()

                            full_content += content_chunk
                            buffer.append(content_chunk)
                            yield content_chunk

                    except (json.JSONDecodeError, KeyError) as e:
                        logger.debug(f"Error parsing stream chunk: {e}")
                        continue
        except Exception as e:
            # Yield any buffered content before raising
            if buffer:
                logger.warning(f"Stream interrupted, yielding {len(buffer)} buffered chunks")
                for chunk in buffer:
                    yield chunk
            raise

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(self.tokenizer.encode(text))

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD."""
        pricing = self.PRICING.get(self.model, {"input": 0.0, "output": 0.0})
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude API provider.

    Pricing (per 1K tokens as of 2024):
    - claude-3-5-sonnet-20241022: $0.003 input, $0.015 output
    - claude-3-haiku-20240307: $0.00025 input, $0.00125 output
    """

    PRICING: dict[str, dict[str, float]] = {
        "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
        "claude-3-5-sonnet-20240620": {"input": 0.003, "output": 0.015},
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    }

    BASE_URL = "https://api.anthropic.com/v1"

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        """Initialize Anthropic provider."""
        super().__init__(model, api_key, timeout, max_retries)

        if not self.api_key:
            raise ValueError("Anthropic API key is required")

    @property
    def name(self) -> str:
        """Get provider name."""
        return "anthropic"

    @with_retry(max_retries=3)
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response using Anthropic API."""
        start_time = time.perf_counter()

        # Build request payload
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        if system_prompt:
            payload["system"] = system_prompt

        url = f"{self.BASE_URL}/messages"
        async with self.session.post(url, json=payload, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()

        # Parse response
        content = data["content"][0]["text"]
        latency = (time.perf_counter() - start_time) * 1000

        # Token counts
        input_tokens = data.get("usage", {}).get("input_tokens", 0)
        output_tokens = data.get("usage", {}).get("output_tokens", 0)
        total_tokens = input_tokens + output_tokens

        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency,
            cost_usd=self.estimate_cost(input_tokens, output_tokens),
            raw_response=data,
            finish_reason=data.get("stop_reason"),
        )

    @with_retry(max_retries=3)
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response using Anthropic API."""
        start_time = time.perf_counter()
        first_token_time = None
        full_content = ""

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            **kwargs,
        }

        if system_prompt:
            payload["system"] = system_prompt

        url = f"{self.BASE_URL}/messages"
        async with self.session.post(url, json=payload, headers=headers) as response:
            response.raise_for_status()

            async for line in response.content:
                line = line.decode("utf-8").strip()

                if not line.startswith("data: "):
                    continue

                data_str = line[6:]

                try:
                    import json

                    data = json.loads(data_str)

                    if data.get("type") == "content_block_delta":
                        content_chunk = data.get("delta", {}).get("text", "")

                        if content_chunk:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()

                            full_content += content_chunk
                            yield content_chunk

                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"Error parsing stream chunk: {e}")
                    continue

    def count_tokens(self, text: str) -> int:
        """
        Count tokens using approximation.

        Anthropic uses a proprietary tokenizer. This is a rough approximation:
        ~4 characters per token for English text.
        """
        return len(text) // 4

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD."""
        pricing = self.PRICING.get(self.model, {"input": 0.0, "output": 0.0})
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost


class OllamaProvider(LLMProvider):
    """
    Local Ollama provider for running models locally.

    Ollama is free and runs on your own hardware. Common models:
    - llama3.2:3b (3B parameters, fast)
    - llama3.2:1b (1B parameters, very fast)
    - mistral:7b (7B parameters)
    - qwen2.5:7b (7B parameters)
    """

    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        timeout: int = 300,  # Local models can be slower
        max_retries: int = 1,  # No need to retry local requests
    ):
        """
        Initialize Ollama provider.

        Args:
            model: Model identifier (e.g., llama3.2:3b)
            base_url: Ollama server URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        super().__init__(model, api_key=None, timeout=timeout, max_retries=max_retries)
        self.base_url = base_url.rstrip("/")

    @property
    def name(self) -> str:
        """Get provider name."""
        return "ollama"

    @with_retry(max_retries=1)
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response using local Ollama model."""
        start_time = time.perf_counter()

        # Build request payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        url = f"{self.base_url}/api/generate"
        async with self.session.post(url, json=payload) as response:
            response.raise_for_status()
            data = await response.json()

        # Parse response
        content = data.get("response", "")
        latency = (time.perf_counter() - start_time) * 1000

        # Token counts (Ollama provides these)
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)
        total_tokens = input_tokens + output_tokens

        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency,
            cost_usd=0.0,  # Local models are free
            raw_response=data,
            finish_reason=data.get("done_reason"),
        )

    @with_retry(max_retries=1)
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from local Ollama model."""
        start_time = time.perf_counter()
        first_token_time = None
        full_content = ""

        # Build request payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        url = f"{self.base_url}/api/generate"
        async with self.session.post(url, json=payload) as response:
            response.raise_for_status()

            async for line in response.content:
                line = line.decode("utf-8").strip()

                if not line:
                    continue

                try:
                    import json

                    data = json.loads(line)
                    content_chunk = data.get("response", "")

                    if content_chunk:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()

                        full_content += content_chunk
                        yield content_chunk

                    if data.get("done"):
                        break

                except json.JSONDecodeError as e:
                    logger.debug(f"Error parsing stream chunk: {e}")
                    continue

    def count_tokens(self, text: str) -> int:
        """
        Approximate token count for Ollama models.

        Most local models use ~4 characters per token for English.
        """
        return len(text) // 4

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Local models have no cost."""
        return 0.0


def create_provider(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> LLMProvider:
    """
    Factory function to create an LLM provider instance.

    Args:
        provider: Provider name (openai, anthropic, ollama)
        model: Model identifier
        api_key: API key for authentication (if required)
        **kwargs: Additional provider-specific parameters

    Returns:
        Initialized LLMProvider instance

    Raises:
        ValueError: If provider is not supported

    Examples:
        >>> provider = create_provider("openai", "gpt-4o-mini", api_key="sk-...")
        >>> response = await provider.generate("Hello, world!")

        >>> ollama = create_provider("ollama", "llama3.2:3b")
        >>> response = await ollama.generate("Explain quantum computing")
    """
    providers: dict[str, type[LLMProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
    }

    provider_class = providers.get(provider.lower())
    if provider_class is None:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: {list(providers.keys())}"
        )

    # Ollama doesn't require API key
    if provider.lower() == "ollama":
        return provider_class(model=model, **kwargs)

    # Other providers require API key
    if not api_key:
        raise ValueError(f"API key is required for {provider}")

    return provider_class(model=model, api_key=api_key, **kwargs)


# Export main classes and factory function
__all__ = [
    "LLMResponse",
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "create_provider",
]
