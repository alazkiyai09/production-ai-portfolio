"""
High-performance embedding service for StreamProcess-Pipeline.

Provides efficient batch embedding generation with GPU support,
model caching, and performance optimizations.
"""

import asyncio
import gc
import os
import threading
import time
from collections import OrderedDict
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel, Field, validator
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

class EmbeddingServiceConfig(BaseModel):
    """Configuration for embedding service."""
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Model name or path"
    )
    device: str = Field(
        default="auto",
        description="Device to use (auto, cpu, cuda, mps)"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Batch size for encoding"
    )
    normalize_embeddings: bool = Field(
        default=True,
        description="L2 normalize embeddings"
    )
    cache_size: int = Field(
        default=1000,
        ge=0,
        description="LRU cache size for recent embeddings"
    )
    show_progress: bool = Field(
        default=False,
        description="Show progress bar for large batches"
    )
    max_length: int = Field(
        default=512,
        ge=1,
        description="Max sequence length"
    )
    precision: str = Field(
        default="float32",
        description="Precision (float32, float16, bfloat16)"
    )

    @validator("device")
    def validate_device(cls, v):
        """Validate device setting."""
        allowed = ["auto", "cpu", "cuda", "mps"]
        if v not in allowed:
            raise ValueError(f"Device must be one of {allowed}, got '{v}'")
        return v

    @validator("precision")
    def validate_precision(cls, v):
        """Validate precision setting."""
        allowed = ["float32", "float16", "bfloat16"]
        if v not in allowed:
            raise ValueError(f"Precision must be one of {allowed}, got '{v}'")
        return v


# ============================================================================
# Model Registry
# ============================================================================

class ModelInfo(BaseModel):
    """Information about an embedding model."""
    name: str
    dimension: int
    max_length: int
    model_type: str
    download_size_mb: Optional[int] = None
    speed_tier: str  # fast, medium, slow
    quality_tier: str  # good, better, best


# Predefined model information
AVAILABLE_MODELS: Dict[str, ModelInfo] = {
    "sentence-transformers/all-MiniLM-L6-v2": ModelInfo(
        name="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384,
        max_length=512,
        model_type="MiniLM",
        download_size_mb=80,
        speed_tier="fast",
        quality_tier="good",
    ),
    "sentence-transformers/all-mpnet-base-v2": ModelInfo(
        name="sentence-transformers/all-mpnet-base-v2",
        dimension=768,
        max_length=512,
        model_type="MPNet",
        download_size_mb=400,
        speed_tier="medium",
        quality_tier="best",
    ),
    "sentence-transformers/all-distilroberta-v1": ModelInfo(
        name="sentence-transformers/all-distilroberta-v1",
        dimension=768,
        max_length=512,
        model_type="DistilRoBERTa",
        download_size_mb=300,
        speed_tier="medium",
        quality_tier="better",
    ),
    "sentence-transformers/paraphrase-albert-small-v2": ModelInfo(
        name="sentence-transformers/paraphrase-albert-small-v2",
        dimension=768,
        max_length=512,
        model_type="ALBERT",
        download_size_mb=70,
        speed_tier="fast",
        quality_tier="good",
    ),
}


# ============================================================================
# Metrics
# ============================================================================

embed_requests_total = Counter(
    "embed_requests_total",
    "Total embedding requests",
    ["model", "status"]
)

embed_duration_seconds = Histogram(
    "embed_duration_seconds",
    "Embedding generation duration",
    ["model", "batch_size"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

embed_cache_hits = Counter(
    "embed_cache_hits_total",
    "Total cache hits",
)

embed_cache_misses = Counter(
    "embed_cache_misses_total",
    "Total cache misses",
)

embed_memory_usage_bytes = Gauge(
    "embed_memory_usage_bytes",
    "Memory usage for embedding model",
    ["model"]
)

embed_batch_size = Histogram(
    "embed_batch_size",
    "Batch size distribution",
    buckets=[1, 8, 16, 32, 64, 128, 256, 512]
)


# ============================================================================
# LRU Cache for Embeddings
# ============================================================================

class EmbeddingCache:
    """
    Thread-safe LRU cache for embeddings.

    Stores text-to-embedding mappings to avoid recomputation.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of cached embeddings
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, text: str) -> Optional[np.ndarray]:
        """
        Get cached embedding.

        Args:
            text: Input text

        Returns:
            Cached embedding or None
        """
        with self._lock:
            if text in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(text)
                return self._cache[text]
            return None

    def set(self, text: str, embedding: np.ndarray) -> None:
        """
        Cache an embedding.

        Args:
            text: Input text
            embedding: Embedding vector
        """
        with self._lock:
            # Remove oldest if at capacity
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)

            self._cache[text] = embedding

    def clear(self) -> None:
        """Clear all cached embeddings."""
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


# ============================================================================
# Embedding Service
# ============================================================================

class EmbeddingService:
    """
    High-performance embedding service.

    Features:
    - GPU/CUDA/MPS support with automatic fallback
    - Model caching
    - LRU cache for recent embeddings
    - Batch optimization
    - Progress tracking
    - Memory-efficient processing
    """

    # Class-level model cache (shared across instances)
    _model_cache: Dict[str, Tuple[SentenceTransformer, str]] = {}
    _model_lock = threading.Lock()

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        cache_size: int = 1000,
        show_progress: bool = False,
    ):
        """
        Initialize embedding service.

        Args:
            model_name: Model name or path
            device: Device to use (auto, cpu, cuda, mps)
            batch_size: Batch size for encoding
            normalize_embeddings: Whether to L2 normalize embeddings
            cache_size: Size of LRU cache for embeddings
            show_progress: Show progress bar
        """
        self.config = EmbeddingServiceConfig(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            cache_size=cache_size,
            show_progress=show_progress,
        )

        self._model: Optional[SentenceTransformer] = None
        self._device: Optional[str] = None
        self._cache = EmbeddingCache(max_size=cache_size)

        # Progress callback
        self._progress_callback: Optional[Callable[[int, int], None]] = None

    @property
    def model(self) -> SentenceTransformer:
        """Get or load embedding model (lazy loading)."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def device(self) -> str:
        """Get device being used."""
        if self._device is None:
            self._load_model()
        return self._device

    def _load_model(self) -> None:
        """
        Load embedding model with caching.

        Uses class-level cache to avoid reloading across instances.
        """
        # Check cache first
        with self._model_lock:
            cache_key = f"{self.config.model_name}:{self.config.device}"

            if cache_key in self._model_cache:
                self._model, self._device = self._model_cache[cache_key]
                print(f"[EmbeddingService] Using cached model: {self.config.model_name} on {self._device}")
                return

            # Determine device
            self._device = self._determine_device()

            print(f"[EmbeddingService] Loading model: {self.config.model_name} on {self._device}")
            start_time = time.time()

            # Load model
            self._model = SentenceTransformer(
                self.config.model_name,
                device=self._device,
            )

            # Set precision
            if self.config.precision == "float16":
                self._model.half()
            elif self.config.precision == "bfloat16":
                if hasattr(torch, "bfloat16"):
                    self._model.bfloat16()

            load_time = time.time() - start_time
            print(f"[EmbeddingService] Model loaded in {load_time:.2f}s")

            # Cache model
            self._model_cache[cache_key] = (self._model, self._device)

            # Update metrics
            embed_memory_usage_bytes.labels(model=self.config.model_name).set(
                self._get_model_memory_size()
            )

    def _determine_device(self) -> str:
        """
        Determine the best available device.

        Returns:
            Device string
        """
        if self.config.device != "auto":
            return self.config.device

        # Auto-detect best device
        if torch.cuda.is_available():
            device = "cuda"
            print(f"[EmbeddingService] Using CUDA (GPU: {torch.cuda.get_device_name(0)})")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            print("[EmbeddingService] Using MPS (Apple Silicon)")
        else:
            device = "cpu"
            print("[EmbeddingService] Using CPU")

        return device

    def _get_model_memory_size(self) -> int:
        """
        Get model memory size in bytes.

        Returns:
            Memory size in bytes
        """
        if self._device == "cuda":
            return torch.cuda.memory_allocated()
        return 0  # CPU memory not easily measurable

    def set_progress_callback(self, callback: Callable[[int, int], None]) -> None:
        """
        Set progress callback for large batches.

        Args:
            callback: Function(current, total) called during processing
        """
        self._progress_callback = callback

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings

        Returns:
            numpy array of embeddings (shape: [len(texts), dimension])
        """
        if not texts:
            return np.array([]).reshape(0, self.get_dimension())

        start_time = time.time()

        try:
            # Check cache for each text
            embeddings = []
            uncached_texts = []
            uncached_indices = []

            for i, text in enumerate(texts):
                cached = self._cache.get(text)
                if cached is not None:
                    embeddings.append((i, cached))
                    embed_cache_hits.inc()
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    embed_cache_misses.inc()

            # Generate embeddings for uncached texts
            if uncached_texts:
                new_embeddings = self._generate_embeddings(uncached_texts)

                # Cache and collect results
                for text, idx, emb in zip(uncached_texts, uncached_indices, new_embeddings):
                    self._cache.set(text, emb)
                    embeddings.append((idx, emb))

            # Sort by original order
            embeddings.sort(key=lambda x: x[0])
            result = np.array([emb for _, emb in embeddings])

            # Record metrics
            duration = time.time() - start_time
            embed_requests_total.labels(model=self.config.model_name, status="success").inc()
            embed_duration_seconds.labels(
                model=self.config.model_name,
                batch_size=len(texts)
            ).observe(duration)
            embed_batch_size.observe(len(texts))

            return result

        except Exception as e:
            embed_requests_total.labels(model=self.config.model_name, status="error").inc()
            raise RuntimeError(f"Failed to generate embeddings: {e}") from e

    def _generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for texts (internal method).

        Handles batching and progress tracking.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        batch_size = self.config.batch_size

        # Create progress bar if needed
        if self.config.show_progress and len(texts) > batch_size:
            iterator = tqdm(range(0, len(texts), batch_size), desc="Generating embeddings")
        else:
            iterator = range(0, len(texts), batch_size)

        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]

            # Generate embeddings for batch
            with torch.no_grad():
                batch_embeddings = self.model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=self.config.normalize_embeddings,
                )

            all_embeddings.extend(batch_embeddings)

            # Progress callback
            if self._progress_callback:
                self._progress_callback(end_idx, len(texts))

            # Memory cleanup for large batches
            if len(texts) > 10000 and end_idx % (batch_size * 10) == 0:
                gc.collect()
                if self._device == "cuda":
                    torch.cuda.empty_cache()

        return all_embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector (1D numpy array)
        """
        embeddings = self.embed_texts([text])
        return embeddings[0] if len(embeddings) > 0 else np.array([])

    def get_dimension(self) -> int:
        """
        Get embedding dimension.

        Returns:
            Dimension of embeddings
        """
        # Try to get from model info
        if self.config.model_name in AVAILABLE_MODELS:
            return AVAILABLE_MODELS[self.config.model_name].dimension

        # Otherwise, encode a dummy text to get dimension
        if self._model is None:
            self._load_model()

        dummy = self.model.encode(["test"], convert_to_numpy=True)
        return dummy.shape[1]

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary with model information
        """
        info = AVAILABLE_MODELS.get(self.config.model_name)

        result = {
            "name": self.config.model_name,
            "device": self.device,
            "dimension": self.get_dimension(),
            "batch_size": self.config.batch_size,
            "normalize": self.config.normalize_embeddings,
            "cache_size": self._cache.size(),
            "precision": self.config.precision,
        }

        if info:
            result["model_type"] = info.model_type
            result["speed_tier"] = info.speed_tier
            result["quality_tier"] = info.quality_tier
            result["download_size_mb"] = info.download_size_mb

        return result

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        gc.collect()

    def preload_model(self) -> None:
        """Preload the model (useful for warm-up)."""
        _ = self.model
        _ = self.device


# ============================================================================
# Benchmarking
# ============================================================================

class EmbeddingBenchmark:
    """Benchmark embedding service performance."""

    @staticmethod
    def benchmark(
        service: EmbeddingService,
        texts: List[str],
        batch_sizes: List[int] = [8, 16, 32, 64, 128],
        warmup_runs: int = 3,
        benchmark_runs: int = 10,
    ) -> Dict[str, Any]:
        """
        Benchmark embedding service.

        Args:
            service: EmbeddingService instance
            texts: Test texts
            batch_sizes: List of batch sizes to test
            warmup_runs: Number of warmup runs
            benchmark_runs: Number of benchmark runs

        Returns:
            Benchmark results
        """
        results = {
            "model_info": service.get_model_info(),
            "num_texts": len(texts),
            "batch_sizes": {},
        }

        print(f"\n{'='*60}")
        print(f"Benchmarking {service.config.model_name}")
        print(f"{'='*60}")

        # Warmup
        print(f"\nWarming up ({warmup_runs} runs)...")
        for _ in range(warmup_runs):
            service.embed_texts(texts)

        # Benchmark different batch sizes
        for batch_size in batch_sizes:
            service.config.batch_size = batch_size

            times = []
            memory_usage = []

            print(f"\nBatch size: {batch_size}")
            print(f"  Running {benchmark_runs} benchmarks...")

            for run in range(benchmark_runs):
                # Measure memory before
                if service.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    mem_before = torch.cuda.memory_allocated()

                # Generate embeddings
                start_time = time.time()
                embeddings = service.embed_texts(texts)
                duration = time.time() - start_time

                # Measure memory after
                if service.device == "cuda":
                    mem_after = torch.cuda.memory_allocated()
                    memory_usage.append((mem_after - mem_before) / 1024 / 1024)  # MB

                times.append(duration)
                print(f"    Run {run+1}: {duration:.3f}s ({len(texts)/duration:.1f} embeddings/sec)")

            # Calculate statistics
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            embeddings_per_sec = len(texts) / avg_time

            batch_results = {
                "avg_time_seconds": avg_time,
                "std_time_seconds": std_time,
                "min_time_seconds": min_time,
                "max_time_seconds": max_time,
                "embeddings_per_second": embeddings_per_sec,
            }

            if memory_usage:
                batch_results["avg_memory_mb"] = np.mean(memory_usage)
                batch_results["max_memory_mb"] = np.max(memory_usage)

            results["batch_sizes"][str(batch_size)] = batch_results

            print(f"  Average: {avg_time:.3f}s ± {std_time:.3f}s")
            print(f"  Speed: {embeddings_per_sec:.1f} embeddings/sec")

            if memory_usage:
                print(f"  Memory: {np.mean(memory_usage):.1f} MB avg")

        return results

    @staticmethod
    def print_benchmark_report(results: Dict[str, Any]) -> None:
        """
        Print benchmark report.

        Args:
            results: Benchmark results from benchmark()
        """
        print(f"\n{'='*60}")
        print("BENCHMARK REPORT")
        print(f"{'='*60}\n")

        print(f"Model: {results['model_info']['name']}")
        print(f"Device: {results['model_info']['device']}")
        print(f"Dimension: {results['model_info']['dimension']}")
        print(f"Test texts: {results['num_texts']}\n")

        print(f"{'Batch Size':<12} {'Avg Time':<12} {'Speed':<20} {'Memory':<12}")
        print("-" * 60)

        for batch_size, stats in results["batch_sizes"].items():
            avg_time = stats["avg_time_seconds"]
            speed = stats["embeddings_per_second"]
            memory = stats.get("avg_memory_mb", 0)

            print(f"{batch_size:<12} {avg_time:<12.3f} {speed:<20.1f} {memory:<12.1f}")

        print("-" * 60)

        # Find optimal batch size
        best_batch = max(
            results["batch_sizes"].items(),
            key=lambda x: x[1]["embeddings_per_second"]
        )

        print(f"\nOptimal batch size: {best_batch[0]} ({best_batch[1]['embeddings_per_second']:.1f} embeddings/sec)")


# ============================================================================
# Convenience Functions
# ============================================================================

def create_embedding_service(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "auto",
    batch_size: int = 32,
) -> EmbeddingService:
    """
    Create an embedding service with default settings.

    Args:
        model_name: Model name
        device: Device to use
        batch_size: Batch size

    Returns:
        EmbeddingService instance
    """
    return EmbeddingService(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )


def quick_embed(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    """
    Quick embedding generation (one-shot).

    Args:
        texts: List of texts
        model_name: Model name

    Returns:
        Embedding array
    """
    service = create_embedding_service(model_name=model_name)
    return service.embed_texts(texts)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys

    # Benchmark if requested
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        # Generate test data
        test_texts = [
            "This is a test sentence for embedding generation.",
            "AdTech analytics require high-throughput processing.",
            "Machine learning models need efficient inference.",
        ] * 100  # 300 texts

        # Test different models
        models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
        ]

        for model_name in models:
            service = create_embedding_service(model_name=model_name)
            results = EmbeddingBenchmark.benchmark(service, test_texts)
            EmbeddingBenchmark.print_benchmark_report(results)
            print("\n")
