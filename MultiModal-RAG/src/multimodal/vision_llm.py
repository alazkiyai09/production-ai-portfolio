"""
Vision LLM - Integration with vision-capable language models.

Supports:
- GPT-4V (OpenAI)
- Claude Vision (Anthropic)
- LLaVA (local)
- Multi-modal query understanding
- Image + text combined prompts
"""

import base64
import io
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json

# Optional dependencies
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class VisionProvider(Enum):
    """Supported vision model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LLAVA = "llava"


class GPT4VModel(Enum):
    """OpenAI GPT-4V models."""
    GPT_4_VISION = "gpt-4-vision-preview"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"


class ClaudeModel(Enum):
    """Anthropic Claude models with vision."""
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"


@dataclass
class VisionMessage:
    """
    A multi-modal message for vision LLMs.

    Attributes:
        role: Message role ('user', 'assistant')
        text: Text content
        images: List of image data (paths, URLs, or base64)
        metadata: Additional metadata
    """
    role: str
    text: str
    images: List[Union[str, Path]] = None
    image_data: List[Dict[str, str]] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.images is None:
            self.images = []
        if self.image_data is None:
            self.image_data = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class VisionResponse:
    """
    Response from a vision LLM.

    Attributes:
        content: Generated text content
        model: Model used
        provider: Provider used
        usage: Token usage information
        metadata: Additional metadata
    """
    content: str
    model: str
    provider: VisionProvider
    usage: Dict[str, int] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.usage is None:
            self.usage = {}
        if self.metadata is None:
            self.metadata = {}


class VisionLLM:
    """
    Interface to vision-capable language models.

    Supports multiple providers:
    - OpenAI GPT-4V
    - Anthropic Claude Vision
    - Local LLaVA models
    """

    def __init__(
        self,
        provider: VisionProvider = VisionProvider.OPENAI,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ):
        """
        Initialize vision LLM client.

        Args:
            provider: Model provider
            model: Model name (None for default)
            api_key: API key (None to load from env)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
        """
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs

        # Set default model
        if model is None:
            if provider == VisionProvider.OPENAI:
                model = GPT4VModel.GPT_4O.value
            elif provider == VisionProvider.ANTHROPIC:
                model = ClaudeModel.CLAUDE_3_5_SONNET.value
            else:
                model = "llava"

        self.model = model

        # Initialize client
        self.client = None
        self._init_client(api_key)

    def _init_client(self, api_key: Optional[str]):
        """Initialize the API client."""
        if self.provider == VisionProvider.OPENAI:
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI library not available")

            self.client = openai.OpenAI(api_key=api_key)

        elif self.provider == VisionProvider.ANTHROPIC:
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic library not available")

            self.client = anthropic.Anthropic(api_key=api_key)

        elif self.provider == VisionProvider.LLAVA:
            # LLaVA would be loaded here
            # For now, placeholder
            self.client = None

    @staticmethod
    def encode_image(image_path: Union[str, Path]) -> str:
        """
        Encode image to base64.

        Args:
            image_path: Path to image

        Returns:
            Base64 encoded string
        """
        image_path = Path(image_path)

        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def get_image_type(image_path: Union[str, Path]) -> str:
        """
        Get image MIME type.

        Args:
            image_path: Path to image

        Returns:
            MIME type string
        """
        path = Path(image_path)
        ext = path.suffix.lower()

        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }

        return mime_types.get(ext, "image/jpeg")

    def prepare_openai_messages(
        self,
        messages: List[VisionMessage]
    ) -> List[Dict[str, Any]]:
        """
        Prepare messages for OpenAI GPT-4V.

        Args:
            messages: List of VisionMessage objects

        Returns:
            Formatted messages for OpenAI API
        """
        formatted = []

        for msg in messages:
            content = []

            # Add text
            if msg.text:
                content.append({
                    "type": "text",
                    "text": msg.text
                })

            # Add images
            for img in msg.images:
                base64_data = self.encode_image(img)
                img_type = self.get_image_type(img)

                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img_type};base64,{base64_data}",
                        "detail": "high"
                    }
                })

            formatted.append({
                "role": msg.role,
                "content": content
            })

        return formatted

    def prepare_anthropic_messages(
        self,
        messages: List[VisionMessage]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Prepare messages for Anthropic Claude.

        Args:
            messages: List of VisionMessage objects

        Returns:
            Tuple of (prompt_text, content_blocks)
        """
        content = []

        for msg in messages:
            if msg.text:
                content.append({
                    "type": "text",
                    "text": msg.text
                })

            for img in msg.images:
                base64_data = self.encode_image(img)
                img_type = self.get_image_type(img)

                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img_type,
                        "data": base64_data
                    }
                })

        # Extract prompt text (first text message)
        prompt = ""
        for msg in messages:
            if msg.text:
                prompt = msg.text
                break

        return prompt, content

    def generate(
        self,
        prompt: str,
        images: Optional[List[Union[str, Path]]] = None,
        messages: Optional[List[VisionMessage]] = None,
        **kwargs
    ) -> VisionResponse:
        """
        Generate a response with vision understanding.

        Args:
            prompt: Text prompt
            images: List of image paths
            messages: Alternative: full conversation history
            **kwargs: Additional provider-specific parameters

        Returns:
            VisionResponse object
        """
        # Build messages from prompt and images if not provided
        if messages is None:
            messages = [
                VisionMessage(
                    role="user",
                    text=prompt,
                    images=images or []
                )
            ]

        # Route to appropriate provider
        if self.provider == VisionProvider.OPENAI:
            return self._generate_openai(messages, **kwargs)
        elif self.provider == VisionProvider.ANTHROPIC:
            return self._generate_anthropic(messages, **kwargs)
        elif self.provider == VisionProvider.LLAVA:
            return self._generate_llava(messages, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _generate_openai(
        self,
        messages: List[VisionMessage],
        **kwargs
    ) -> VisionResponse:
        """Generate using OpenAI GPT-4V."""
        formatted_messages = self.prepare_openai_messages(messages)

        params = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.kwargs,
            **kwargs
        }

        try:
            response = self.client.chat.completions.create(**params)

            return VisionResponse(
                content=response.choices[0].message.content,
                model=response.model,
                provider=VisionProvider.OPENAI,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            )

        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")

    def _generate_anthropic(
        self,
        messages: List[VisionMessage],
        **kwargs
    ) -> VisionResponse:
        """Generate using Anthropic Claude."""
        prompt, content = self.prepare_anthropic_messages(messages)

        params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **self.kwargs,
            **kwargs
        }

        try:
            response = self.client.messages.create(
                messages=[{"role": "user", "content": content}],
                **params
            )

            return VisionResponse(
                content=response.content[0].text,
                model=response.model,
                provider=VisionProvider.ANTHROPIC,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                }
            )

        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")

    def _generate_llava(
        self,
        messages: List[VisionMessage],
        **kwargs
    ) -> VisionResponse:
        """Generate using local LLaVA model."""
        # Placeholder for LLaVA implementation
        # Would require transformers + local model loading
        raise NotImplementedError("LLaVA support not yet implemented")

    def analyze_image(
        self,
        image_path: Union[str, Path],
        prompt: str = "Describe this image in detail.",
        **kwargs
    ) -> VisionResponse:
        """
        Analyze a single image.

        Args:
            image_path: Path to image
            prompt: Analysis prompt
            **kwargs: Additional parameters

        Returns:
            VisionResponse with analysis
        """
        return self.generate(
            prompt=prompt,
            images=[image_path],
            **kwargs
        )

    def compare_images(
        self,
        image_paths: List[Union[str, Path]],
        prompt: str = "Compare these images and describe similarities and differences.",
        **kwargs
    ) -> VisionResponse:
        """
        Compare multiple images.

        Args:
            image_paths: List of image paths
            prompt: Comparison prompt
            **kwargs: Additional parameters

        Returns:
            VisionResponse with comparison
        """
        return self.generate(
            prompt=prompt,
            images=image_paths,
            **kwargs
        )

    def answer_with_context(
        self,
        question: str,
        context: str,
        images: Optional[List[Union[str, Path]]] = None,
        **kwargs
    ) -> VisionResponse:
        """
        Answer a question with text context and optional images.

        Args:
            question: Question to answer
            context: Relevant context text
            images: Optional images to reference
            **kwargs: Additional parameters

        Returns:
            VisionResponse with answer
        """
        prompt = f"""Context: {context}

Question: {question}

Please answer the question based on the provided context and images."""

        return self.generate(
            prompt=prompt,
            images=images,
            **kwargs
        )

    def extract_text_from_image(
        self,
        image_path: Union[str, Path],
        **kwargs
    ) -> VisionResponse:
        """
        Extract text from an image (OCR with vision understanding).

        Args:
            image_path: Path to image
            **kwargs: Additional parameters

        Returns:
            VisionResponse with extracted text
        """
        prompt = """Extract all text from this image. Preserve the structure and formatting as much as possible.
If there are tables, extract them in a structured format.
If there are handwritten notes, transcribe them to the best of your ability."""

        return self.analyze_image(image_path, prompt, **kwargs)

    def describe_table(
        self,
        image_path: Union[str, Path],
        **kwargs
    ) -> VisionResponse:
        """
        Describe a table in an image.

        Args:
            image_path: Path to image containing table
            **kwargs: Additional parameters

        Returns:
            VisionResponse with table description
        """
        prompt = """This image contains a table. Please:
1. Extract all data from the table
2. Identify column headers
3. Present the data in a structured format (Markdown table or JSON)
4. Summarize the key information in the table"""

        return self.analyze_image(image_path, prompt, **kwargs)

    def chat(
        self,
        messages: List[VisionMessage],
        **kwargs
    ) -> VisionResponse:
        """
        Multi-turn chat with vision support.

        Args:
            messages: Conversation history
            **kwargs: Additional parameters

        Returns:
            VisionResponse with assistant reply
        """
        return self.generate(messages=messages, **kwargs)

    def stream_generate(
        self,
        prompt: str,
        images: Optional[List[Union[str, Path]]] = None,
        **kwargs
    ):
        """
        Stream generation (for OpenAI and Anthropic).

        Args:
            prompt: Text prompt
            images: List of image paths
            **kwargs: Additional parameters

        Yields:
            Chunks of generated text
        """
        messages = [
            VisionMessage(
                role="user",
                text=prompt,
                images=images or []
            )
        ]

        if self.provider == VisionProvider.OPENAI:
            formatted = self.prepare_openai_messages(messages)

            stream = self.client.chat.completions.create(
                model=self.model,
                messages=formatted,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                **kwargs
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        elif self.provider == VisionProvider.ANTHROPIC:
            prompt_text, content = self.prepare_anthropic_messages(messages)

            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": content}],
                **kwargs
            ) as stream:
                for text in stream.text_stream:
                    yield text

        else:
            raise NotImplementedError(f"Streaming not supported for {self.provider}")


# Convenience functions
def analyze_image(
    image_path: Union[str, Path],
    prompt: str = "Describe this image.",
    provider: VisionProvider = VisionProvider.OPENAI,
    **kwargs
) -> str:
    """
    Convenience function to analyze an image.

    Args:
        image_path: Path to image
        prompt: Analysis prompt
        provider: Vision model provider
        **kwargs: Additional parameters

    Returns:
        Analysis text
    """
    llm = VisionLLM(provider=provider, **kwargs)
    response = llm.analyze_image(image_path, prompt)
    return response.content


def extract_text_from_image(
    image_path: Union[str, Path],
    provider: VisionProvider = VisionProvider.OPENAI,
    **kwargs
) -> str:
    """
    Convenience function to extract text from an image.

    Args:
        image_path: Path to image
        provider: Vision model provider
        **kwargs: Additional parameters

    Returns:
        Extracted text
    """
    llm = VisionLLM(provider=provider, **kwargs)
    response = llm.extract_text_from_image(image_path)
    return response.content
