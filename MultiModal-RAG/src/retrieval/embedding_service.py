# ============================================================
# Enterprise-RAG: Embedding Service
# ============================================================
"""
High-performance embedding service with caching and GPU support.

This module provides efficient text embedding generation with:
- Lazy model loading for fast startup
- Batch processing for throughput
- LRU caching for repeated texts
- Automatic GPU detection and CPU fallback
- Support for multiple sentence-transformer models
- Similarity calculation utilities
- Performance benchmarking

Example:
    >>> from src.retrieval import EmbeddingService
    >>> service = EmbeddingService(model_name="all-MiniLM-L6-v2", device="auto")
    >>> embeddings = service.embed_texts(["Hello world", "Test document"])
    >>> print(embeddings.shape)  # (2, 384)
    >>> sim = service.similarity(embeddings[0], embeddings[1])
    >>> print(f"Similarity: {sim:.3f}")
"""

import gc
import hashlib
import logging
import platform
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import psutil

from sentence_transformers import SentenceTransformer

from src.config import settings
from src.exceptions import ConfigurationError, EmbeddingError, ModelNotAvailableError
from src.logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)


# ============================================================
# Model Registry
# ============================================================

@dataclass(frozen=True)
class ModelInfo:
    """Information about a supported embedding model."""

    name: str
    dimension: int
    max_seq_length: int
    description: str
    download_size_mb: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "dimension": self.dimension,
            "max_seq_length": self.max_seq_length,
            "description": self.description,
            "download_size_mb": self.download_size_mb,
        }


# Supported models registry
MODEL_REGISTRY = {
    "all-MiniLM-L6-v2": ModelInfo(
        name="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384,
        max_seq_length=256,
        description="Fast and efficient model for general purpose",
        download_size_mb=80,
    ),
    "all-mpnet-base-v2": ModelInfo(
        name="sentence-transformers/all-mpnet-base-v2",
        dimension=768,
        max_seq_length=384,
        description="High quality model, better accuracy but slower",
        download_size_mb=400,
    ),
    "e5-large-v2": ModelInfo(
        name="intfloat/e5-large-v2",
        dimension=1024,
        max_seq_length=512,
        description="Instruction-tuned model for better retrieval",
        download_size_mb=1200,
    ),
    "bge-base-en-v1.5": ModelInfo(
        name="BAAI/bge-base-en-v1.5",
        dimension=768,
        max_seq_length=512,
        description="State-of-the-art retrieval model",
        download_size_mb=400,
    ),
    "bge-small-en-v1.5": ModelInfo(
        name="BAAI/bge-small-en-v1.5",
        dimension=384,
        max_seq_length=512,
        description="Compact state-of-the-art model",
        download_size_mb=200,
    ),
}


# ============================================================
# Embedding Service
# ============================================================

class EmbeddingService:
    """
    High-performance embedding generation service.

    Features:
        - Lazy model loading (loads on first use)
        - Batch processing for efficiency
        - LRU caching for repeated texts
        - Automatic GPU detection with CPU fallback
        - Similarity calculations
        - Performance monitoring

    Example:
        >>> service = EmbeddingService(model_name="all-MiniLM-L6-v2", device="auto")
        >>> # Single text
        >>> emb = service.embed_single("Hello world")
        >>> print(emb.shape)  # (384,)
        >>> # Batch processing
        >>> embs = service.embed_texts(["Doc 1", "Doc 2", "Doc 3"])
        >>> print(embs.shape)  # (3, 384)
        >>> # Similarity
        >>> sim = service.similarity(embs[0], embs[1])
    """

    # Class-level model cache (shared across instances)
    _model_cache: dict[str, SentenceTransformer] = {}
    _model_lock = threading.Lock()

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "auto",
        batch_size: Optional[int] = None,
        normalize: bool = True,
        cache_size: int = 1000,
        enable_cache: bool = True,
    ) -> None:
        """
        Initialize the embedding service.

        Args:
            model_name: Model name (uses settings if None)
            device: Device for inference ("auto", "cpu", "cuda", "mps")
            batch_size: Batch size for embedding (uses settings if None)
            normalize: Whether to normalize embeddings to unit length
            cache_size: Size of LRU cache for embeddings
            enable_cache: Whether to enable embedding caching

        Example:
            >>> service = EmbeddingService(
            ...     model_name="all-MiniLM-L6-v2",
            ...     device="auto",
            ...     batch_size=32,
            ... )
        """
        # Configuration
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = device
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self.normalize = normalize
        self.cache_size = cache_size
        self.enable_cache = enable_cache

        # Get model info from registry
        self._model_info = self._get_model_info(self.model_name)

        # Lazy-loaded model
        self._model: Optional[SentenceTransformer] = None
        self._is_loaded = False

        # Statistics
        self._stats = {
            "embeddings_generated": 0,
            "texts_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_time": 0.0,
        }
        self._stats_lock = threading.Lock()

        # Device detection
        self._actual_device = self._detect_device()

        # Simple LRU cache for embeddings
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._cache_access_order: list[str] = []
        self._cache_lock = threading.Lock()

        logger.info(
            "EmbeddingService initialized",
            extra={
                "model": self.model_name,
                "device": self._actual_device,
                "batch_size": self.batch_size,
                "normalize": self.normalize,
                "cache_enabled": self.enable_cache,
            },
        )

    # ============================================================
    # Model Loading
    # ============================================================

    @property
    def model(self) -> SentenceTransformer:
        """
        Lazy load the embedding model.

        The model is loaded on first access and cached for reuse.
        Uses class-level cache to share models across instances.

        Returns:
            Loaded SentenceTransformer model

        Raises:
            ModelNotAvailableError: If model cannot be loaded

        Example:
            >>> model = service.model
            >>> print(type(model))  # <class 'sentence_transformers.SentenceTransformer'>
        """
        if self._model is None:
            with self._model_lock:
                # Double-check after acquiring lock
                if self._model is None:
                    self._load_model()

        return self._model

    def _load_model(self) -> None:
        """Load the embedding model from cache or disk."""
        # Check if model is already loaded in class cache
        if self.model_name in self._model_cache:
            logger.info(f"Using cached model: {self.model_name}")
            self._model = self._model_cache[self.model_name]
            self._is_loaded = True
            return

        # Load new model
        logger.info(f"Loading model: {self.model_name}")
        start_time = time.time()

        try:
            # Determine device
            device = self._determine_device()

            # Load model
            self._model = SentenceTransformer(
                self.model_name,
                device=device,
            )

            # Verify model loaded successfully
            if self._model is None:
                raise ModelNotAvailableError(
                    model_name=self.model_name,
                    reason="Model loading returned None",
                )

            # Cache the model
            self._model_cache[self.model_name] = self._model
            self._is_loaded = True

            load_time = time.time() - start_time
            logger.info(
                f"Model loaded successfully in {load_time:.2f}s",
                extra={
                    "model": self.model_name,
                    "device": str(self._model.device),
                    "load_time": round(load_time, 2),
                },
            )

        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}", exc_info=True)
            raise ModelNotAvailableError(
                model_name=self.model_name,
                reason=str(e),
            )

    def _detect_device(self) -> str:
        """
        Detect the best available device for inference.

        Returns:
            Device string ("cuda", "mps", or "cpu")

        Example:
            >>> device = service._detect_device()
            >>> print(device)  # "cuda" or "cpu"
        """
        if self.device != "auto":
            return self.device

        # Try CUDA first
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass

        # Try MPS (Apple Silicon)
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass

        # Default to CPU
        return "cpu"

    def _determine_device(self) -> str:
        """
        Determine the actual device to use.

        Returns:
            Device string for PyTorch

        Example:
            >>> device = service._determine_device()
        """
        if self.device == "auto":
            return self._actual_device
        return self.device

    def unload_model(self) -> None:
        """
        Unload the model from memory.

        Useful for freeing memory when the service won't be used for a while.

        Example:
            >>> service.unload_model()
        """
        if self._model is not None:
            logger.info(f"Unloading model: {self.model_name}")
            del self._model
            self._model = None
            self._is_loaded = False

            # Force garbage collection
            gc.collect()

            # Try to clear CUDA cache
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    # ============================================================
    # Properties
    # ============================================================

    @property
    def embedding_dimension(self) -> int:
        """
        Return the dimension of embeddings.

        Returns:
            Embedding dimension (e.g., 384 for MiniLM-L6-v2)

        Example:
            >>> dim = service.embedding_dimension
            >>> print(dim)  # 384
        """
        return self._model_info.dimension

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded and self._model is not None

    @property
    def device(self) -> str:
        """Get the actual device being used."""
        if self._model is not None:
            return str(self._model.device)
        return self._actual_device

    # ============================================================
    # Embedding Methods
    # ============================================================

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts, handling batching automatically.

        Args:
            texts: List of text strings to embed

        Returns:
            NumPy array of shape (n_texts, embedding_dim)

        Raises:
            EmbeddingError: If embedding fails

        Example:
            >>> texts = ["Hello world", "Test document", "Another text"]
            >>> embeddings = service.embed_texts(texts)
            >>> print(embeddings.shape)  # (3, 384)
        """
        if not texts:
            return np.array([]).reshape(0, self.embedding_dimension)

        start_time = time.time()
        n_texts = len(texts)

        logger.debug(f"Embedding {n_texts} texts", extra={"batch_size": self.batch_size})

        try:
            # Process in batches
            all_embeddings = []

            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]

                # Check cache if enabled
                if self.enable_cache:
                    cached_embeddings, uncached_indices, uncached_texts = (
                        self._get_cached_embeddings(batch)
                    )
                else:
                    cached_embeddings = None
                    uncached_indices = list(range(len(batch)))
                    uncached_texts = batch

                # Embed uncached texts
                if uncached_texts:
                    new_embeddings = self._model.encode(
                        uncached_texts,
                        convert_to_numpy=True,
                        normalize_embeddings=self.normalize,
                        show_progress_bar=False,
                    )

                    # Update cache
                    if self.enable_cache:
                        self._cache_embeddings(
                            [uncached_texts[i] for i in uncached_indices],
                            new_embeddings,
                        )

                    # Merge with cached
                    if cached_embeddings is not None:
                        batch_embeddings = self._merge_cached_and_new(
                            cached_embeddings,
                            uncached_indices,
                            new_embeddings,
                        )
                    else:
                        batch_embeddings = new_embeddings
                else:
                    batch_embeddings = cached_embeddings

                all_embeddings.append(batch_embeddings)

            # Concatenate all batches
            embeddings = np.vstack(all_embeddings)

            # Update statistics
            elapsed = time.time() - start_time
            self._update_stats(
                texts_processed=n_texts,
                embeddings_generated=n_texts,
                elapsed=elapsed,
            )

            logger.debug(
                f"Embedded {n_texts} texts in {elapsed:.3f}s",
                extra={
                    "texts_per_second": round(n_texts / elapsed, 1),
                    "avg_time_per_text": round(elapsed / n_texts * 1000, 2),
                },
            )

            return embeddings

        except Exception as e:
            logger.error(f"Failed to embed texts: {str(e)}", exc_info=True)
            raise EmbeddingError(
                message=f"Text embedding failed: {str(e)}",
                model=self.model_name,
                text_length=len(texts),
            )

    def embed_single(self, text: str) -> np.ndarray:
        """
        Embed a single text.

        Args:
            text: Text string to embed

        Returns:
            NumPy array of shape (embedding_dim,)

        Example:
            >>> embedding = service.embed_single("Hello world")
            >>> print(embedding.shape)  # (384,)
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        embeddings = self.embed_texts([text])
        return embeddings[0]

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string.

        May use different prompt templates for query encoding
        depending on the model being used.

        Args:
            query: Query string to embed

        Returns:
            NumPy array of shape (embedding_dim,)

        Example:
            >>> query_emb = service.embed_query("What is the refund policy?")
        """
        # Some models benefit from instruction prefixes for queries
        if "e5-" in self.model_name.lower():
            # E5 models require "query:" prefix
            query = f"query: {query}"
        elif "bge-" in self.model_name.lower():
            # BGE models require "Represent this sentence for searching relevant passages:"
            query = f"Represent this sentence for searching relevant passages: {query}"

        return self.embed_single(query)

    # ============================================================
    # Caching Methods
    # ============================================================

    def _get_cached_embeddings(
        self,
        texts: list[str],
    ) -> tuple[Optional[np.ndarray], list[int], list[str]]:
        """
        Get cached embeddings for texts.

        Args:
            texts: List of texts to check in cache

        Returns:
            Tuple of (cached_embeddings, uncached_indices, uncached_texts)
        """
        cached_embeddings = np.zeros((len(texts), self.embedding_dimension))
        uncached_indices = []
        uncached_texts = []

        with self._stats_lock:
            for i, text in enumerate(texts):
                cache_key = self._make_cache_key(text)
                cached = self._cache_get(cache_key)
                if cached is not None:
                    cached_embeddings[i] = cached
                    self._stats["cache_hits"] += 1
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
                    self._stats["cache_misses"] += 1

        if len(uncached_indices) == len(texts):
            # No cache hits
            return None, uncached_indices, uncached_texts

        if len(uncached_indices) == 0:
            # All cached
            return cached_embeddings, [], []

        # Partial cache hit
        return cached_embeddings, uncached_indices, uncached_texts

    def _cache_embeddings(self, texts: list[str], embeddings: np.ndarray) -> None:
        """Cache embeddings for texts."""
        for text, embedding in zip(texts, embeddings):
            cache_key = self._make_cache_key(text)
            self._cache_put(cache_key, embedding)

    def _cache_get(self, cache_key: str) -> Optional[np.ndarray]:
        """Get embedding from cache with LRU tracking."""
        with self._cache_lock:
            if cache_key in self._embedding_cache:
                # Update access order
                if cache_key in self._cache_access_order:
                    self._cache_access_order.remove(cache_key)
                self._cache_access_order.append(cache_key)
                return self._embedding_cache[cache_key].copy()
            return None

    def _cache_put(self, cache_key: str, embedding: np.ndarray) -> None:
        """Put embedding in cache with LRU eviction."""
        with self._cache_lock:
            # Add to cache
            self._embedding_cache[cache_key] = embedding.copy()
            self._cache_access_order.append(cache_key)

            # Evict oldest if at capacity
            while len(self._embedding_cache) > self.cache_size:
                oldest = self._cache_access_order.pop(0)
                del self._embedding_cache[oldest]

    def _merge_cached_and_new(
        self,
        cached: np.ndarray,
        uncached_indices: list[int],
        new_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Merge cached and newly computed embeddings."""
        result = cached.copy()
        for idx, new_emb in zip(uncached_indices, new_embeddings):
            result[idx] = new_emb
        return result

    def _make_cache_key(self, text: str) -> str:
        """Create cache key from text."""
        # Use hash to avoid storing full strings in cache keys
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def clear_cache(self) -> None:
        """
        Clear the embedding cache.

        Example:
            >>> service.clear_cache()
        """
        with self._cache_lock:
            self._embedding_cache.clear()
            self._cache_access_order.clear()
        logger.info("Embedding cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats

        Example:
            >>> stats = service.get_cache_stats()
            >>> print(f"Hit rate: {stats['hit_rate']:.2%}")
        """
        with self._stats_lock:
            hits = self._stats["cache_hits"]
            misses = self._stats["cache_misses"]
            total = hits + misses

        with self._cache_lock:
            cache_size = len(self._embedding_cache)

        return {
            "hits": hits,
            "misses": misses,
            "total_lookups": total,
            "hit_rate": hits / total if total > 0 else 0.0,
            "cache_size": cache_size,
            "cache_max_size": self.cache_size,
        }

    # ============================================================
    # Similarity Methods
    # ============================================================

    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between -1 and 1 (typically 0 to 1 for normalized embeddings)

        Example:
            >>> sim = service.similarity(emb1, emb2)
            >>> print(f"Similarity: {sim:.3f}")  # 0.857
        """
        # Ensure embeddings are numpy arrays
        if not isinstance(embedding1, np.ndarray):
            embedding1 = np.array(embedding1)
        if not isinstance(embedding2, np.ndarray):
            embedding2 = np.array(embedding2)

        # Flatten if needed
        embedding1 = embedding1.flatten()
        embedding2 = embedding2.flatten()

        # Calculate cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def similarities(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate similarities between query and multiple documents.

        Args:
            query_embedding: Query embedding vector
            document_embeddings: Array of document embeddings

        Returns:
            Array of similarity scores

        Example:
            >>> scores = service.similarities(query_emb, doc_embs)
            >>> print(scores.shape)  # (n_docs,)
        """
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
        if not isinstance(document_embeddings, np.ndarray):
            document_embeddings = np.array(document_embeddings)

        # Flatten query
        query_embedding = query_embedding.flatten()

        # Calculate dot products
        if self.normalize:
            # For normalized embeddings, dot product = cosine similarity
            return np.dot(document_embeddings, query_embedding)
        else:
            # Manual cosine similarity
            norms = np.linalg.norm(document_embeddings, axis=1)
            dot_products = np.dot(document_embeddings, query_embedding)
            query_norm = np.linalg.norm(query_embedding)
            return dot_products / (norms * query_norm + 1e-8)

    # ============================================================
    # Statistics and Monitoring
    # ============================================================

    def get_stats(self) -> dict[str, Any]:
        """
        Get service statistics.

        Returns:
            Dictionary with performance stats

        Example:
            >>> stats = service.get_stats()
            >>> print(f"Processed {stats['texts_processed']} texts")
        """
        with self._stats_lock:
            total_time = self._stats["total_time"]
            texts_processed = self._stats["texts_processed"]

            return {
                "model": self.model_name,
                "device": self.device,
                "embedding_dimension": self.embedding_dimension,
                "is_loaded": self.is_loaded,
                "texts_processed": texts_processed,
                "embeddings_generated": self._stats["embeddings_generated"],
                "cache_hits": self._stats["cache_hits"],
                "cache_misses": self._stats["cache_misses"],
                "total_time_seconds": round(total_time, 3),
                "avg_texts_per_second": round(texts_processed / total_time, 1)
                if total_time > 0
                else 0,
            }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        with self._stats_lock:
            for key in self._stats:
                self._stats[key] = 0 if key != "total_time" else 0.0
        logger.info("Statistics reset")

    def _update_stats(
        self,
        texts_processed: int,
        embeddings_generated: int,
        elapsed: float,
    ) -> None:
        """Update internal statistics."""
        with self._stats_lock:
            self._stats["texts_processed"] += texts_processed
            self._stats["embeddings_generated"] += embeddings_generated
            self._stats["total_time"] += elapsed

    # ============================================================
    # Benchmarking
    # ============================================================

    def benchmark(
        self,
        n_texts: int = 100,
        text_length: int = 100,
        include_cache: bool = True,
    ) -> dict[str, Any]:
        """
        Benchmark embedding performance.

        Measures:
        - Embeddings per second
        - Memory usage
        - Device utilization
        - Cache effectiveness

        Args:
            n_texts: Number of texts to embed
            text_length: Length of synthetic texts
            include_cache: Whether to test cache performance

        Returns:
            Dictionary with benchmark results

        Example:
            >>> results = service.benchmark(n_texts=1000)
            >>> print(f"Speed: {results['embeddings_per_second']:.1f} emb/s")
        """
        logger.info(f"Starting benchmark with {n_texts} texts")

        # Generate synthetic texts
        import random
        import string

        words = [
            "embedding",
            "vector",
            "search",
            "retrieval",
            "document",
            "query",
            "similarity",
            "semantic",
        ]
        texts = [
            " ".join(random.choices(words, k=min(text_length // 6, 20))) for _ in range(n_texts)
        ]

        # Ensure model is loaded
        _ = self.model

        # Measure memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Benchmark embedding speed
        start_time = time.time()
        embeddings = self.embed_texts(texts)
        elapsed = time.time() - start_time

        # Measure memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before

        # Calculate metrics
        embeddings_per_second = n_texts / elapsed if elapsed > 0 else 0

        results = {
            "model": self.model_name,
            "device": self.device,
            "n_texts": n_texts,
            "elapsed_seconds": round(elapsed, 3),
            "embeddings_per_second": round(embeddings_per_second, 1),
            "avg_time_per_text_ms": round(elapsed / n_texts * 1000, 2),
            "memory_used_mb": round(memory_used, 2),
            "memory_after_mb": round(memory_after, 2),
            "batch_size": self.batch_size,
            "normalize": self.normalize,
        }

        # Add cache stats if enabled
        if include_cache and self.enable_cache:
            # Benchmark with repeated texts (cache hits)
            repeated_texts = texts[:10] * (n_texts // 10)
            start_time = time.time()
            _ = self.embed_texts(repeated_texts)
            cache_elapsed = time.time() - start_time

            results["cache_benchmark"] = {
                "n_texts": len(repeated_texts),
                "elapsed_seconds": round(cache_elapsed, 3),
                "embeddings_per_second": round(len(repeated_texts) / cache_elapsed, 1),
                "speedup_factor": round((elapsed / cache_elapsed), 2),
            }

            results["cache_stats"] = self.get_cache_stats()

        logger.info(
            f"Benchmark complete: {embeddings_per_second:.1f} emb/s",
            extra=results,
        )

        return results

    # ============================================================
    # Model Information
    # ============================================================

    def get_model_info(self) -> dict[str, Any]:
        """
        Return model information for debugging.

        Returns:
            Dictionary with model details

        Example:
            >>> info = service.get_model_info()
            >>> print(info['description'])
        """
        return {
            "name": self._model_info.name,
            "dimension": self._model_info.dimension,
            "max_seq_length": self._model_info.max_seq_length,
            "description": self._model_info.description,
            "download_size_mb": self._model_info.download_size_mb,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "normalize": self.normalize,
            "batch_size": self.batch_size,
        }

    def _get_model_info(self, model_name: str) -> ModelInfo:
        """Get model info from registry."""
        # Try exact match
        if model_name in MODEL_REGISTRY:
            return MODEL_REGISTRY[model_name]

        # Try partial match
        for key, info in MODEL_REGISTRY.items():
            if key in model_name or model_name in key:
                return info

        # Default to assuming standard sentence-transformer model
        logger.warning(
            f"Model {model_name} not in registry, using defaults. "
            "Dimension will be detected on first use."
        )

        # Try to get dimension from actual model
        try:
            temp_model = SentenceTransformer(model_name)
            dimension = temp_model.get_sentence_embedding_dimension()
            del temp_model

            return ModelInfo(
                name=model_name,
                dimension=dimension,
                max_seq_length=512,
                description="Custom model",
            )
        except Exception:
            return ModelInfo(
                name=model_name,
                dimension=768,  # Common default
                max_seq_length=512,
                description="Unknown model (using defaults)",
            )

    # ============================================================
    # Utility Methods
    # ============================================================

    @classmethod
    def clear_model_cache(cls) -> None:
        """
        Clear the class-level model cache.

        This will free memory used by all loaded models.

        Example:
            >>> EmbeddingService.clear_model_cache()
        """
        with cls._model_lock:
            for model_name in list(cls._model_cache.keys()):
                del cls._model_cache[model_name]
        logger.info("Model cache cleared")

    @classmethod
    def get_available_models(cls) -> list[str]:
        """
        Get list of available pre-configured models.

        Returns:
            List of model names

        Example:
            >>> models = EmbeddingService.get_available_models()
            >>> for model in models:
            ...     print(model)
        """
        return list(MODEL_REGISTRY.keys())

    @classmethod
    def get_model_description(cls, model_key: str) -> Optional[str]:
        """
        Get description of a model.

        Args:
            model_key: Key from MODEL_REGISTRY

        Returns:
            Model description or None

        Example:
            >>> desc = EmbeddingService.get_model_description("all-MiniLM-L6-v2")
            >>> print(desc)
        """
        info = MODEL_REGISTRY.get(model_key)
        return info.description if info else None

    def __repr__(self) -> str:
        """String representation of the service."""
        return (
            f"EmbeddingService("
            f"model={self.model_name}, "
            f"device={self.device}, "
            f"dim={self.embedding_dimension}, "
            f"loaded={self.is_loaded}"
            f")"
        )


# ============================================================
# Utility Functions
# ============================================================

def create_embedding_service(
    model_name: Optional[str] = None,
    device: str = "auto",
) -> EmbeddingService:
    """
    Create an embedding service configured from settings.

    Args:
        model_name: Override model from settings
        device: Override device from settings

    Returns:
        Configured EmbeddingService instance

    Example:
        >>> service = create_embedding_service()
        >>> emb = service.embed_single("Hello")
    """
    return EmbeddingService(
        model_name=model_name or settings.EMBEDDING_MODEL,
        device=device,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
    )


# ============================================================
# Context Manager for Optional Unloading
# ============================================================

@contextmanager
def temporary_unload(service: EmbeddingService):
    """
    Context manager that temporarily unloads the model.

    Useful for freeing memory during long operations that don't
    require embeddings.

    Args:
        service: EmbeddingService instance

    Example:
        >>> with temporary_unload(service):
        ...     # Do memory-intensive work
        ...     process_large_data()
        >>> # Model is automatically reloaded
        >>> emb = service.embed_single("text")
    """
    was_loaded = service.is_loaded
    if was_loaded:
        service.unload_model()

    try:
        yield
    finally:
        if was_loaded:
            # Model will be reloaded on next access
            pass


# Export public API
__all__ = [
    # Main class
    "EmbeddingService",
    # Model info
    "ModelInfo",
    "MODEL_REGISTRY",
    # Utilities
    "create_embedding_service",
    "temporary_unload",
]
