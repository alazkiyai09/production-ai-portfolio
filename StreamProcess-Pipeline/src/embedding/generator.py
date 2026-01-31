"""
Embedding generation for StreamProcess-Pipeline.

Provides efficient embedding generation with caching.
"""

import hashlib
import os
from typing import Any, Dict, List, Optional

import numpy as np
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


# ============================================================================
# Configuration
# ============================================================================

class EmbeddingConfig(BaseModel):
    """Configuration for embedding generator."""
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Model name or path"
    )
    device: str = Field(default="cpu", description="Device to run on (cpu/cuda)")
    batch_size: int = Field(default=32, description="Batch size for encoding")
    dimension: int = Field(default=384, description="Embedding dimension")
    cache_enabled: bool = Field(default=True, description="Enable caching")
    max_length: int = Field(default=512, description="Max text length")


# ============================================================================
# Embedding Cache
# ============================================================================

class EmbeddingCache:
    """
    Cache for embeddings to avoid recomputation.

    Can be backed by Redis or in-memory.
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize cache.

        Args:
            max_size: Maximum cache size
        """
        self.max_size = max_size
        self._cache: Dict[str, List[float]] = {}
        self._access_count: Dict[str, int] = {}

    def get(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding.

        Args:
            text: Text to lookup

        Returns:
            Cached embedding or None
        """
        key = self._hash(text)

        if key in self._cache:
            self._access_count[key] = self._access_count.get(key, 0) + 1
            return self._cache[key]

        return None

    def set(self, text: str, embedding: List[float]) -> None:
        """
        Cache an embedding.

        Args:
            text: Text string
            embedding: Embedding vector
        """
        # Evict if cache is full
        if len(self._cache) >= self.max_size:
            # Find least accessed item
            min_key = min(self._access_count, key=self._access_count.get)
            del self._cache[min_key]
            del self._access_count[min_key]

        key = self._hash(text)
        self._cache[key] = embedding
        self._access_count[key] = 1

    def _hash(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.sha256(text.encode()).hexdigest()

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        self._access_count.clear()

    def size(self) -> int:
        """Get cache size."""
        return len(self._cache)


# ============================================================================
# Embedding Generator
# ============================================================================

class EmbeddingGenerator:
    """
    Generate embeddings for text using sentence-transformers.

    Features:
    - Batch encoding
    - Caching
    - GPU support
    - Progress tracking
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        cache: Optional[EmbeddingCache] = None,
    ):
        """
        Initialize embedding generator.

        Args:
            config: Optional configuration
            cache: Optional cache instance
        """
        self.config = config or EmbeddingConfig()
        self.cache = cache or (EmbeddingCache() if self.config.cache_enabled else None)

        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Get or load model."""
        if self._model is None:
            self._model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
            )
        return self._model

    def generate(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> List[List[float]]:
        """
        Generate embeddings for texts.

        Args:
            texts: List of text strings
            show_progress: Show progress bar

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Check cache
        embeddings = []
        cache_hits = []
        texts_to_encode = []
        indices_to_encode = []

        for i, text in enumerate(texts):
            if self.cache:
                cached = self.cache.get(text)
                if cached is not None:
                    embeddings.append(cached)
                    cache_hits.append(i)
                    continue

            texts_to_encode.append(text)
            indices_to_encode.append(i)
            embeddings.append(None)  # Placeholder

        # Generate embeddings for uncached texts
        if texts_to_encode:
            new_embeddings = self.model.encode(
                texts_to_encode,
                batch_size=self.config.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )

            # Fill in placeholders
            for idx, text, embedding in zip(indices_to_encode, texts_to_encode, new_embeddings):
                embedding_list = embedding.tolist()
                embeddings[idx] = embedding_list

                # Cache if enabled
                if self.cache:
                    self.cache.set(text, embedding_list)

        # Reorder to match input order
        result = []
        cache_hit_list = []

        for i, emb in enumerate(embeddings):
            result.append(emb)
            cache_hit_list.append(i in cache_hits)

        return result

    def generate_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text string

        Returns:
            Embedding vector
        """
        embeddings = self.generate([text])
        return embeddings[0] if embeddings else []

    def generate_async(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> List[List[float]]:
        """
        Async wrapper for generate (for compatibility).

        In production, this would use a thread pool executor
        for true async processing.

        Args:
            texts: List of text strings
            show_progress: Show progress bar

        Returns:
            List of embedding vectors
        """
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.generate, texts, show_progress)
            return future.result()


# ============================================================================
# Model Registry
# ============================================================================

class ModelRegistry:
    """Registry for embedding models."""

    _models: Dict[str, SentenceTransformer] = {}

    @classmethod
    def get_model(cls, model_name: str, device: str = "cpu") -> SentenceTransformer:
        """
        Get or load model.

        Args:
            model_name: Model name or path
            device: Device to load on

        Returns:
            SentenceTransformer instance
        """
        key = f"{model_name}:{device}"

        if key not in cls._models:
            cls._models[key] = SentenceTransformer(model_name, device=device)

        return cls._models[key]

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached models."""
        cls._models.clear()


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_embedding(text: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[float]:
    """
    Quick embedding generation for a single text.

    Args:
        text: Text string
        model_name: Model name

    Returns:
        Embedding vector
    """
    generator = EmbeddingGenerator(
        config=EmbeddingConfig(model_name=model_name)
    )
    return generator.generate_single(text)


def batch_embeddings(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> List[List[float]]:
    """
    Batch embedding generation.

    Args:
        texts: List of text strings
        model_name: Model name
        batch_size: Batch size

    Returns:
        List of embedding vectors
    """
    generator = EmbeddingGenerator(
        config=EmbeddingConfig(model_name=model_name, batch_size=batch_size)
    )
    return generator.generate(texts)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "EmbeddingConfig",
    "EmbeddingCache",
    "EmbeddingGenerator",
    "ModelRegistry",
    "quick_embedding",
    "batch_embeddings",
]
