# ============================================================
# Enterprise-RAG: Cross-Encoder Reranker
# ============================================================
"""
Cross-encoder reranker for improved retrieval accuracy.

This module provides:
- Cross-encoder model for query-document relevance scoring
- Batch processing for efficiency
- Lazy model loading with caching
- Score normalization using sigmoid
- Performance benchmarking

Cross-encoders provide more accurate relevance scores than bi-encoders
(dense embeddings) because they process the query and document together,
allowing deeper interaction between terms. However, they're slower and
require scoring each query-document pair individually.

Best practice: Use hybrid retrieval (fast) → rerank top results (accurate)

Example:
    >>> from src.retrieval.reranker import CrossEncoderReranker
    >>> reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    >>> results = reranker.rerank(query="machine learning", results=hybrid_results, top_k=5)
    >>> for result in results:
    ...     print(f"{result.chunk_id}: {result.rerank_score:.3f}")
"""

import gc
import logging
import threading
import time
from dataclasses import replace
from typing import Any, Optional

import numpy as np

from src.config import settings
from src.logging_config import get_logger
from src.retrieval.hybrid_retriever import HybridSearchResult

# Initialize logger
logger = get_logger(__name__)


# ============================================================
# Reranked Result
# ============================================================

class RerankedSearchResult(HybridSearchResult):
    """
    Extended search result with reranking information.

    Inherits all fields from HybridSearchResult and adds:
    - rerank_score: Cross-encoder relevance score
    - original_rank: Rank before reranking
    - rank_change: Change in rank after reranking

    Example:
        >>> result = RerankedSearchResult(
        ...     **base_result_data,
        ...     rerank_score=0.92,
        ...     original_rank=5,
        ...     rank_change=-4  # Moved up 4 positions
        ... )
    """

    def __init__(
        self,
        base_result: HybridSearchResult,
        rerank_score: float,
        original_rank: int,
        rank_change: int,
    ):
        """
        Create reranked result from base hybrid result.

        Args:
            base_result: Original HybridSearchResult
            rerank_score: Cross-encoder relevance score
            original_rank: Rank before reranking (1-indexed)
            rank_change: Change in rank (negative = improved)
        """
        # Copy all fields from base result
        self._base_result = base_result
        self._rerank_score = rerank_score
        self._original_rank = original_rank
        self._rank_change = rank_change

    def __getattr__(self, name: str) -> Any:
        """Delegate to base result for all other attributes."""
        return getattr(self._base_result, name)

    @property
    def rerank_score(self) -> float:
        """Get cross-encoder relevance score."""
        return self._rerank_score

    @property
    def original_rank(self) -> int:
        """Get rank before reranking."""
        return self._original_rank

    @property
    def rank_change(self) -> int:
        """Get change in rank (negative = improved position)."""
        return self._rank_change

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with reranking info."""
        base_dict = self._base_result.to_dict()
        base_dict.update({
            "rerank_score": self._rerank_score,
            "original_rank": self._original_rank,
            "rank_change": self._rank_change,
        })
        return base_dict

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RerankedSearchResult("
            f"chunk_id={self.chunk_id}, "
            f"rerank_score={self._rerank_score:.3f}, "
            f"rank={self._original_rank}→{self._original_rank + self._rank_change}"
            f")"
        )


# ============================================================
# Cross-Encoder Reranker
# ============================================================

class CrossEncoderReranker:
    """
    Cross-encoder reranker for improved retrieval accuracy.

    Cross-encoders score query-document pairs by processing them together
    through a transformer model, allowing deeper interaction between query
    terms and document content. This produces more accurate relevance scores
    than bi-encoder embeddings alone.

    Usage Pattern:
        1. Retrieve candidates with hybrid search (fast, less accurate)
        2. Rerank top-N with cross-encoder (slower, more accurate)
        3. Return top-K reranked results

    Args:
        model_name: HuggingFace model name or path
        device: Device for inference ("auto", "cuda", "cpu", "mps")
        batch_size: Batch size for scoring
        normalize_scores: Whether to normalize scores to 0-1 range
        max_length: Maximum sequence length for model

    Example:
        >>> reranker = CrossEncoderReranker(
        ...     model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        ...     device="auto",
        ...     batch_size=16
        ... )
        >>> results = reranker.rerank(query, hybrid_results, top_k=5)
    """

    # Model registry with info
    MODELS = {
        "ms-marco-MiniLM-L-6-v2": {
            "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "description": "Fast, accurate model for general reranking",
            "max_length": 512,
            "expected_score_range": (-10, 10),
        },
        "ms-marco-MiniLM-L-12-v2": {
            "name": "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "description": "More accurate but slower (12 layers)",
            "max_length": 512,
            "expected_score_range": (-10, 10),
        },
        "ms-marco-electra-base": {
            "name": "cross-encoder/ms-marco-electra-base",
            "description": "ELECTRA-based model, good accuracy/speed",
            "max_length": 512,
            "expected_score_range": (-10, 10),
        },
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "auto",
        batch_size: Optional[int] = None,
        normalize_scores: bool = True,
        max_length: Optional[int] = None,
    ) -> None:
        """
        Initialize the cross-encoder reranker.

        Args:
            model_name: Model name (uses settings if None)
            device: Device for inference
            batch_size: Batch size for scoring
            normalize_scores: Whether to normalize scores
            max_length: Max sequence length (uses model default if None)
        """
        # Configuration
        self.model_name = model_name or settings.RERANKER_MODEL
        self.device_input = device
        self.batch_size = batch_size or 32
        self.normalize_scores = normalize_scores
        self.max_length = max_length

        # Resolve device
        self.device = self._resolve_device(device)

        # Lazy-loaded model
        self._model: Optional[Any] = None
        self._is_loaded = False
        self._model_lock = threading.Lock()

        # Statistics
        self._stats = {
            "queries_reranked": 0,
            "documents_scored": 0,
            "total_time": 0.0,
            "batches_processed": 0,
        }
        self._stats_lock = threading.Lock()

        logger.info(
            "CrossEncoderReranker initialized",
            extra={
                "model": self.model_name,
                "device": self.device,
                "batch_size": self.batch_size,
                "normalize": self.normalize_scores,
            },
        )

    @property
    def model(self) -> Any:
        """
        Lazy load the cross-encoder model.

        Returns:
            Loaded CrossEncoder model

        Example:
            >>> model = reranker.model
            >>> print(type(model))
        """
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._load_model()
        return self._model

    def _load_model(self) -> None:
        """Load the cross-encoder model."""
        from sentence_transformers import CrossEncoder

        logger.info(f"Loading cross-encoder model: {self.model_name}")
        start_time = time.time()

        try:
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length,
            )
            self._is_loaded = True

            load_time = time.time() - start_time
            logger.info(
                f"Cross-encoder model loaded in {load_time:.2f}s",
                extra={
                    "model": self.model_name,
                    "load_time": round(load_time, 2),
                    "device": str(self._model.device),
                },
            )

        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {str(e)}", exc_info=True)
            raise

    def unload_model(self) -> None:
        """
        Unload the model from memory.

        Example:
            >>> reranker.unload_model()
        """
        if self._model is not None:
            logger.info(f"Unloading cross-encoder model: {self.model_name}")
            del self._model
            self._model = None
            self._is_loaded = False

            # Force garbage collection
            gc.collect()

            # Clear CUDA cache if available
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    # ============================================================
    # Reranking
    # ============================================================

    def rerank(
        self,
        query: str,
        results: list[HybridSearchResult],
        top_k: Optional[int] = None,
        return_reranked_class: bool = True,
    ) -> list[HybridSearchResult]:
        """
        Rerank results using cross-encoder.

        Scores each query-document pair and reorders results by relevance.

        Args:
            query: Original search query
            results: Results from hybrid retrieval to rerank
            top_k: Number of top results to return (None = return all)
            return_reranked_class: If True, return RerankedSearchResult instances

        Returns:
            Reranked results, sorted by cross-encoder score (descending)

        Example:
            >>> reranked = reranker.rerank(query, results, top_k=5)
            >>> for r in reranked:
            ...     print(f"{r.chunk_id}: {r.rerank_score:.3f}")
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not results:
            logger.warning("No results to rerank")
            return []

        start_time = time.time()
        logger.info(
            f"Reranking {len(results)} results for query: {query[:100]}",
            extra={"num_results": len(results), "top_k": top_k},
        )

        # Store original rankings
        original_rankings = {r.chunk_id: i + 1 for i, r in enumerate(results)}

        # Extract document texts
        documents = [r.content for r in results]

        # Score all query-document pairs
        scores = self._batch_score(query, documents)

        # Update results with rerank scores
        reranked_results = []
        for result, score in zip(results, scores):
            if return_reranked_class:
                # Create RerankedSearchResult with extended info
                reranked_results.append(
                    RerankedSearchResult(
                        base_result=result,
                        rerank_score=score,
                        original_rank=original_rankings[result.chunk_id],
                        rank_change=0,  # Will update after sorting
                    )
                )
            else:
                # Update in-place (not recommended, loses original scores)
                reranked_results.append(result)

        # Sort by rerank score
        reranked_results.sort(key=lambda r: r.rerank_score, reverse=True)

        # Update rank changes
        if return_reranked_class:
            for new_rank, result in enumerate(reranked_results, start=1):
                result._rank_change = new_rank - result._original_rank

        # Trim to top_k
        if top_k is not None and top_k > 0:
            reranked_results = reranked_results[:top_k]

        # Update statistics
        elapsed = time.time() - start_time
        with self._stats_lock:
            self._stats["queries_reranked"] += 1
            self._stats["documents_scored"] += len(results)
            self._stats["total_time"] += elapsed

        logger.info(
            f"Reranked {len(results)} results in {elapsed:.3f}s",
            extra={
                "elapsed_ms": round(elapsed * 1000, 2),
                "docs_per_second": round(len(results) / elapsed, 1),
            },
        )

        return reranked_results

    def _batch_score(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        """
        Score query-document pairs in batches.

        Args:
            query: Search query
            documents: List of document texts

        Returns:
            List of relevance scores (same order as input)

        Example:
            >>> scores = reranker._batch_score("query", ["doc1", "doc2"])
            >>> print(scores)  # [0.85, 0.62]
        """
        if not documents:
            return []

        # Prepare query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Score in batches
        all_scores = []

        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]

            # Score batch
            raw_scores = self.model.predict(
                batch,
                apply_softmax=False,  # We'll normalize ourselves
                convert_to_numpy=True,
                convert_to_tensor=False,
            )

            # Convert to list if needed
            if hasattr(raw_scores, "tolist"):
                raw_scores = raw_scores.tolist()
            elif not isinstance(raw_scores, list):
                raw_scores = list(raw_scores)

            all_scores.extend(raw_scores)

            # Update batch stats
            with self._stats_lock:
                self._stats["batches_processed"] += 1

        # Normalize scores if requested
        if self.normalize_scores:
            all_scores = self._normalize_scores(all_scores)

        return all_scores

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        """
        Normalize scores to 0-1 range using sigmoid.

        Sigmoid: σ(x) = 1 / (1 + e^(-x))

        This handles both positive and negative raw scores and produces
        well-calibrated probabilities.

        Args:
            scores: Raw scores from cross-encoder

        Returns:
            Normalized scores in [0, 1] range

        Example:
            >>> normalized = reranker._normalize_scores([-5.2, 0.3, 3.8])
            >>> print(normalized)  # [0.005, 0.574, 0.978]
        """
        import numpy as np

        scores_array = np.array(scores)

        # Apply sigmoid normalization
        # Sigmoid: 1 / (1 + exp(-x))
        normalized = 1.0 / (1.0 + np.exp(-scores_array))

        # Clip to [0, 1] to handle any numerical issues
        normalized = np.clip(normalized, 0.0, 1.0)

        return normalized.tolist()

    # ============================================================
    # Device Resolution
    # ============================================================

    def _resolve_device(self, device: str) -> str:
        """
        Resolve 'auto' to actual device.

        Args:
            device: Device specification ("auto", "cuda", "cpu", "mps")

        Returns:
            Actual device to use

        Example:
            >>> device = reranker._resolve_device("auto")
            >>> print(device)  # "cuda" or "cpu"
        """
        if device != "auto":
            return device

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

    # ============================================================
    # Statistics
    # ============================================================

    def get_stats(self) -> dict[str, Any]:
        """
        Get reranker statistics.

        Returns:
            Dictionary with performance statistics

        Example:
            >>> stats = reranker.get_stats()
            >>> print(f"Queries: {stats['queries_reranked']}")
        """
        with self._stats_lock:
            total_time = self._stats["total_time"]
            queries = self._stats["queries_reranked"]

            return {
                "model": self.model_name,
                "device": self.device,
                "batch_size": self.batch_size,
                "queries_reranked": queries,
                "documents_scored": self._stats["documents_scored"],
                "batches_processed": self._stats["batches_processed"],
                "total_time_seconds": round(total_time, 3),
                "avg_time_per_query_ms": round(total_time / queries * 1000, 1)
                if queries > 0
                else 0,
                "avg_docs_per_second": round(self._stats["documents_scored"] / total_time, 1)
                if total_time > 0
                else 0,
            }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        with self._stats_lock:
            for key in self._stats:
                self._stats[key] = 0 if key != "total_time" else 0.0
        logger.info("Reranker statistics reset")

    # ============================================================
    # Benchmarking
    # ============================================================

    def benchmark(
        self,
        query: str,
        documents: list[str],
        num_runs: int = 3,
    ) -> dict[str, Any]:
        """
        Benchmark reranking performance.

        Args:
            query: Test query
            documents: List of documents to score
            num_runs: Number of benchmark runs

        Returns:
            Dictionary with benchmark results

        Example:
            >>> results = reranker.benchmark("test query", docs * 10)
            >>> print(f"Throughput: {results['docs_per_second']:.1f} docs/s")
        """
        logger.info(
            f"Running benchmark with {len(documents)} documents, {num_runs} runs"
        )

        # Ensure model is loaded
        _ = self.model

        # Warm-up run
        _ = self._batch_score(query, documents[:1])

        # Benchmark runs
        times = []
        for _ in range(num_runs):
            start = time.time()
            _ = self._batch_score(query, documents)
            times.append(time.time() - start)

        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        results = {
            "model": self.model_name,
            "device": self.device,
            "num_documents": len(documents),
            "batch_size": self.batch_size,
            "num_runs": num_runs,
            "avg_time_seconds": round(avg_time, 3),
            "std_time_seconds": round(std_time, 3),
            "min_time_seconds": round(min_time, 3),
            "max_time_seconds": round(max_time, 3),
            "docs_per_second": round(len(documents) / avg_time, 1),
            "avg_time_per_doc_ms": round(avg_time / len(documents) * 1000, 2),
        }

        # Add memory info if available
        try:
            import psutil

            process = psutil.Process()
            results["memory_mb"] = round(process.memory_info().rss / 1024 / 1024, 1)
        except Exception:
            pass

        logger.info(
            f"Benchmark complete: {results['docs_per_second']:.1f} docs/s",
            extra=results,
        )

        return results

    # ============================================================
    # Model Information
    # ============================================================

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dictionary with model details

        Example:
            >>> info = reranker.get_model_info()
            >>> print(info['description'])
        """
        # Check if model is in registry
        for key, info in self.MODELS.items():
            if key in self.model_name:
                return {
                    "name": self.model_name,
                    "description": info["description"],
                    "max_length": info["max_length"],
                    "device": self.device,
                    "is_loaded": self._is_loaded,
                    "batch_size": self.batch_size,
                }

        # Default info for unknown models
        return {
            "name": self.model_name,
            "description": "Custom cross-encoder model",
            "max_length": self.max_length or 512,
            "device": self.device,
            "is_loaded": self._is_loaded,
            "batch_size": self.batch_size,
        }

    @classmethod
    def get_available_models(cls) -> list[dict[str, Any]]:
        """
        Get list of available cross-encoder models.

        Returns:
            List of model information dictionaries

        Example:
            >>> models = CrossEncoderReranker.get_available_models()
            >>> for model in models:
            ...     print(f"{model['name']}: {model['description']}")
        """
        return [
            {
                "key": key,
                "name": info["name"],
                "description": info["description"],
                "max_length": info["max_length"],
            }
            for key, info in cls.MODELS.items()
        ]

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CrossEncoderReranker("
            f"model={self.model_name}, "
            f"device={self.device}, "
            f"loaded={self._is_loaded}"
            f")"
        )


# ============================================================
# Utility Functions
# ============================================================

def create_reranker(
    model_name: Optional[str] = None,
    device: str = "auto",
) -> CrossEncoderReranker:
    """
    Create a cross-encoder reranker configured from settings.

    Args:
        model_name: Override model from settings
        device: Device for inference

    Returns:
        Configured CrossEncoderReranker instance

    Example:
        >>> reranker = create_reranker()
        >>> results = reranker.rerank(query, hybrid_results, top_k=5)
    """
    return CrossEncoderReranker(
        model_name=model_name or settings.RERANKER_MODEL,
        device=device,
    )


# Export public API
__all__ = [
    # Classes
    "RerankedSearchResult",
    "CrossEncoderReranker",
    # Utilities
    "create_reranker",
]
