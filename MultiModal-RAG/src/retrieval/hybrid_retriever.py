# ============================================================
# Enterprise-RAG: Hybrid Retriever
# ============================================================
"""
Hybrid retrieval combining dense vector and sparse BM25 search.

This module provides:
- Dense vector search (semantic similarity)
- Sparse BM25 search (keyword matching)
- Reciprocal Rank Fusion (RRF) for result combination
- Configurable dense/sparse weights
- Automatic deduplication

Hybrid search is particularly effective for:
- Queries with both semantic and keyword components
- Domain-specific terminology
- Complementing semantic search with exact matches

Example:
    >>> from src.retrieval.hybrid_retriever import HybridRetriever
    >>> retriever = HybridRetriever(
    ...     vector_store=vector_store,
    ...     embedding_service=embedding_service,
    ...     bm25_retriever=bm25_retriever,
    ...     dense_weight=0.7,
    ...     sparse_weight=0.3
    ... )
    >>> results = retriever.retrieve("machine learning algorithms", top_k=10)
    >>> for result in results:
    ...     print(f"{result.chunk_id}: dense={result.dense_score:.2f} "
    ...           f"sparse={result.sparse_score:.2f} fused={result.fused_score:.2f}")
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from src.config import settings
from src.ingestion.document_processor import Document
from src.logging_config import get_logger
from src.retrieval.embedding_service import EmbeddingService
from src.retrieval.sparse_retriever import BM25Retriever, SparseSearchResult
from src.retrieval.vector_store import SearchResult, VectorStoreBase

# Initialize logger
logger = get_logger(__name__)


# ============================================================
# Data Classes
# ============================================================

@dataclass(frozen=True)
class HybridSearchResult:
    """
    Result from hybrid search combining dense and sparse retrieval.

    The fused_score combines both semantic similarity (dense) and keyword
    relevance (sparse) using Reciprocal Rank Fusion.

    Attributes:
        doc_id: Source document ID
        chunk_id: Unique chunk identifier
        content: Text content of the chunk
        metadata: Associated metadata (source, page, etc.)
        dense_score: Dense vector similarity score (0-1)
        sparse_score: BM25 keyword relevance score (0-1)
        fused_score: Combined RRF score
        dense_rank: Rank in dense results (0 if not in top-k)
        sparse_rank: Rank in sparse results (0 if not in top-k)

    Example:
        >>> result = HybridSearchResult(
        ...     doc_id="doc_123",
        ...     chunk_id="doc_123_chunk_0",
        ...     content="Machine learning is...",
        ...     metadata={"source": "ml_intro.pdf"},
        ...     dense_score=0.85,
        ...     sparse_score=0.62,
        ...     fused_score=0.76,
        ...     dense_rank=1,
        ...     sparse_rank=3
        ... )
    """

    doc_id: str
    chunk_id: str
    content: str
    metadata: dict[str, Any]
    dense_score: float
    sparse_score: float
    fused_score: float
    dense_rank: int = 0
    sparse_rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": self.metadata,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "fused_score": self.fused_score,
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HybridSearchResult":
        """Create HybridSearchResult from dictionary."""
        return cls(
            doc_id=data["doc_id"],
            chunk_id=data["chunk_id"],
            content=data["content"],
            metadata=data["metadata"],
            dense_score=data["dense_score"],
            sparse_score=data["sparse_score"],
            fused_score=data["fused_score"],
            dense_rank=data.get("dense_rank", 0),
            sparse_rank=data.get("sparse_rank", 0),
        )


@dataclass
class HybridRetrieverStats:
    """
    Statistics for the hybrid retriever.

    Attributes:
        total_queries: Total number of queries processed
        hybrid_queries: Number of queries using hybrid search
        dense_only_queries: Number of queries using dense only
        avg_dense_results: Average number of dense results
        avg_sparse_results: Average number of sparse results
        avg_fused_results: Average number of fused results
        avg_query_time_ms: Average query time in milliseconds
    """

    total_queries: int = 0
    hybrid_queries: int = 0
    dense_only_queries: int = 0
    avg_dense_results: float = 0.0
    avg_sparse_results: float = 0.0
    avg_fused_results: float = 0.0
    avg_query_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_queries": self.total_queries,
            "hybrid_queries": self.hybrid_queries,
            "dense_only_queries": self.dense_only_queries,
            "avg_dense_results": round(self.avg_dense_results, 2),
            "avg_sparse_results": round(self.avg_sparse_results, 2),
            "avg_fused_results": round(self.avg_fused_results, 2),
            "avg_query_time_ms": round(self.avg_query_time_ms, 2),
        }


# ============================================================
# Hybrid Retriever
# ============================================================

class HybridRetriever:
    """
    Hybrid retriever combining dense vector and sparse BM25 search.

    Uses Reciprocal Rank Fusion (RRF) to combine results from both retrieval
    methods. RRF is robust to differences in score scales and doesn't require
    score normalization.

    RRF Formula:
        fused_score(d) = Σ (weight_i / (k + rank_i))

    Where:
        - d is a document
        - weight_i is the weight for retrieval method i
        - rank_i is the rank of document d in method i
        - k is a constant (default: 60)

    Args:
        vector_store: Dense vector store (ChromaDB, Qdrant, etc.)
        embedding_service: Service for generating query embeddings
        bm25_retriever: BM25 sparse retriever
        dense_weight: Weight for dense results (default: 0.7)
        sparse_weight: Weight for sparse results (default: 0.3)
        rrf_k: RRF constant k (default: 60)
        top_k_multiplier: Multiplier for retrieving more candidates before fusion

    Example:
        >>> retriever = HybridRetriever(
        ...     vector_store=vector_store,
        ...     embedding_service=embedding_service,
        ...     bm25_retriever=bm25_retriever,
        ...     dense_weight=0.7,
        ...     sparse_weight=0.3
        ... )
        >>> results = retriever.retrieve("machine learning algorithms", top_k=10)
    """

    def __init__(
        self,
        vector_store: VectorStoreBase,
        embedding_service: EmbeddingService,
        bm25_retriever: BM25Retriever,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
        top_k_multiplier: int = 2,
    ) -> None:
        """
        Initialize the hybrid retriever.

        Args:
            vector_store: Dense vector store instance
            embedding_service: Embedding service instance
            bm25_retriever: BM25 retriever instance
            dense_weight: Weight for dense results (0-1)
            sparse_weight: Weight for sparse results (0-1)
            rrf_k: RRF constant (higher = more forgiving of rank differences)
            top_k_multiplier: Retrieve N * top_k candidates from each method
        """
        # Components
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.bm25_retriever = bm25_retriever

        # Configuration
        if not (0 <= dense_weight <= 1):
            raise ValueError(f"dense_weight must be between 0 and 1, got {dense_weight}")
        if not (0 <= sparse_weight <= 1):
            raise ValueError(f"sparse_weight must be between 0 and 1, got {sparse_weight}")
        if abs(dense_weight + sparse_weight - 1.0) > 0.01:
            logger.warning(
                f"dense_weight ({dense_weight}) + sparse_weight ({sparse_weight}) "
                f"!= 1.0. Weights will be normalized."
            )

        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        self.top_k_multiplier = top_k_multiplier

        # Statistics
        self._stats = HybridRetrieverStats()

        logger.info(
            "HybridRetriever initialized",
            extra={
                "dense_weight": dense_weight,
                "sparse_weight": sparse_weight,
                "rrf_k": rrf_k,
                "top_k_multiplier": top_k_multiplier,
            },
        )

    # ============================================================
    # Main Retrieval Method
    # ============================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_hybrid: bool = True,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[HybridSearchResult]:
        """
        Retrieve documents using hybrid search.

        Combines dense vector search (semantic similarity) with sparse BM25
        search (keyword matching) using Reciprocal Rank Fusion.

        Args:
            query: Search query string
            top_k: Number of final results to return
            use_hybrid: If True, use both dense and sparse; if False, dense only
            filters: Optional metadata filters for dense search

        Returns:
            List of HybridSearchResult objects, sorted by fused_score (descending)

        Example:
            >>> results = retriever.retrieve(
            ...     "machine learning algorithms",
            ...     top_k=10,
            ...     filters={"file_type": "pdf"}
            ... )
            >>> for result in results:
            ...     print(f"{result.chunk_id}: {result.fused_score:.3f}")
        """
        import time

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        start_time = time.time()
        logger.info(
            f"Hybrid retrieval for query: {query[:100]}",
            extra={"top_k": top_k, "use_hybrid": use_hybrid},
        )

        # Perform searches
        try:
            # Dense search (always performed)
            dense_results = self._dense_search(
                query,
                top_k * self.top_k_multiplier,
                filters,
            )

            # Sparse search (optional)
            sparse_results = []
            if use_hybrid:
                sparse_results = self._sparse_search(
                    query,
                    top_k * self.top_k_multiplier,
                )

            # Check if we have sparse results
            if use_hybrid and not sparse_results:
                logger.warning("Sparse search returned no results, using dense only")
                use_hybrid = False

            # Combine results
            if use_hybrid and sparse_results:
                # Use Reciprocal Rank Fusion
                fused_rankings = self._reciprocal_rank_fusion(
                    dense_results,
                    sparse_results,
                    k=self.rrf_k,
                )

                # Merge with original result data
                results = self._merge_results(
                    fused_rankings,
                    {r[0]: r for r in dense_results},
                    {r[0]: r for r in sparse_results},
                    top_k,
                )

                self._stats.hybrid_queries += 1
                self._stats.avg_sparse_results = (
                    (self._stats.avg_sparse_results * (self._stats.hybrid_queries - 1)
                     + len(sparse_results)) / self._stats.hybrid_queries
                )
            else:
                # Dense only - convert dense results to hybrid format
                results = []
                for rank, (chunk_id, dense_result) in enumerate(dense_results[:top_k], start=1):
                    result = HybridSearchResult(
                        doc_id=dense_result.doc_id,
                        chunk_id=chunk_id,
                        content=dense_result.content,
                        metadata=dense_result.metadata,
                        dense_score=dense_result.score,
                        sparse_score=0.0,
                        fused_score=dense_result.score,
                        dense_rank=rank,
                        sparse_rank=0,
                    )
                    results.append(result)

                self._stats.dense_only_queries += 1
                self._stats.avg_sparse_results = 0.0

            # Update statistics
            elapsed = time.time() - start_time
            self._stats.total_queries += 1
            self._stats.avg_dense_results = (
                (self._stats.avg_dense_results * (self._stats.total_queries - 1)
                 + len(dense_results)) / self._stats.total_queries
            )
            self._stats.avg_fused_results = (
                (self._stats.avg_fused_results * (self._stats.total_queries - 1)
                 + len(results)) / self._stats.total_queries
            )
            self._stats.avg_query_time_ms = (
                (self._stats.avg_query_time_ms * (self._stats.total_queries - 1)
                 + elapsed * 1000) / self._stats.total_queries
            )

            logger.info(
                f"Retrieved {len(results)} results in {elapsed:.3f}s",
                extra={
                    "results_count": len(results),
                    "dense_count": len(dense_results),
                    "sparse_count": len(sparse_results) if use_hybrid else 0,
                    "elapsed_ms": round(elapsed * 1000, 2),
                },
            )

            return results

        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {str(e)}", exc_info=True)
            raise

    # ============================================================
    # Dense Search
    # ============================================================

    def _dense_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[dict[str, Any]],
    ) -> list[tuple[str, SearchResult]]:
        """
        Perform dense vector search.

        Args:
            query: Search query
            top_k: Number of results to retrieve
            filters: Optional metadata filters

        Returns:
            List of (chunk_id, SearchResult) tuples

        Example:
            >>> results = retriever._dense_search("ml algorithms", top_k=20)
            >>> print([(chunk_id, r.score) for chunk_id, r in results[:3]])
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.embed_query(query)

            # Search vector store
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters,
            )

            # Convert to (chunk_id, result) tuples
            results = [(result.chunk_id, result) for result in search_results]

            logger.debug(
                f"Dense search returned {len(results)} results",
                extra={"query": query[:50], "top_k": top_k},
            )

            return results

        except Exception as e:
            logger.error(f"Dense search failed: {str(e)}", exc_info=True)
            return []

    # ============================================================
    # Sparse Search
    # ============================================================

    def _sparse_search(
        self,
        query: str,
        top_k: int,
    ) -> list[tuple[str, SparseSearchResult]]:
        """
        Perform sparse BM25 search.

        Args:
            query: Search query
            top_k: Number of results to retrieve

        Returns:
            List of (chunk_id, SparseSearchResult) tuples

        Example:
            >>> results = retriever._sparse_search("ml algorithms", top_k=20)
            >>> print([(chunk_id, r.score) for chunk_id, r in results[:3]])
        """
        try:
            # Search BM25 index
            search_results = self.bm25_retriever.search(
                query=query,
                top_k=top_k,
            )

            # Convert to (chunk_id, result) tuples
            results = [(result.chunk_id, result) for result in search_results]

            logger.debug(
                f"Sparse search returned {len(results)} results",
                extra={"query": query[:50], "top_k": top_k},
            )

            return results

        except Exception as e:
            logger.error(f"Sparse search failed: {str(e)}", exc_info=True)
            return []

    # ============================================================
    # Reciprocal Rank Fusion
    # ============================================================

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[tuple[str, SearchResult]],
        sparse_results: list[tuple[str, SparseSearchResult]],
        k: int = 60,
    ) -> list[tuple[str, float]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        RRF is robust to differences in score scales and doesn't require
        score normalization. It combines rankings rather than raw scores.

        RRF Formula:
            score(d) = dense_weight / (k + rank_dense) +
                       sparse_weight / (k + rank_sparse)

        Where rank starts at 1 (not 0).

        Args:
            dense_results: List of (chunk_id, SearchResult) from dense search
            sparse_results: List of (chunk_id, SparseSearchResult) from sparse search
            k: RRF constant (higher = more forgiving of rank differences)

        Returns:
            List of (chunk_id, fused_score) tuples, sorted by score (descending)

        Example:
            >>> fused = retriever._reciprocal_rank_fusion(
            ...     dense_results=[("id1", r1), ("id2", r2)],
            ...     sparse_results=[("id2", r3), ("id4", r4)],
            ...     k=60
            ... )
            >>> print([(id, score) for id, score in fused[:3]])
        """
        # Normalize weights if needed
        total_weight = self.dense_weight + self.sparse_weight
        norm_dense_weight = self.dense_weight / total_weight
        norm_sparse_weight = self.sparse_weight / total_weight

        # Build score dictionary
        scores: dict[str, float] = {}

        # Add dense scores
        for rank, (chunk_id, _) in enumerate(dense_results, start=1):
            rrf_score = norm_dense_weight / (k + rank)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_score

        # Add sparse scores
        for rank, (chunk_id, _) in enumerate(sparse_results, start=1):
            rrf_score = norm_sparse_weight / (k + rank)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_score

        # Sort by score (descending)
        fused_rankings = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        logger.debug(
            f"RRF combined {len(dense_results)} dense + {len(sparse_results)} sparse "
            f"into {len(fused_rankings)} unique results"
        )

        return fused_rankings

    # ============================================================
    # Result Merging
    # ============================================================

    def _merge_results(
        self,
        fused_rankings: list[tuple[str, float]],
        dense_results: dict[str, SearchResult],
        sparse_results: dict[str, SparseSearchResult],
        top_k: int,
    ) -> list[HybridSearchResult]:
        """
        Merge fused rankings with original result data.

        Args:
            fused_rankings: List of (chunk_id, fused_score) from RRF
            dense_results: Dictionary mapping chunk_id to SearchResult
            sparse_results: Dictionary mapping chunk_id to SparseSearchResult
            top_k: Number of results to return

        Returns:
            List of HybridSearchResult objects

        Example:
            >>> results = retriever._merge_results(
            ...     fused_rankings=[("id1", 0.8), ("id2", 0.6)],
            ...     dense_results={"id1": dense_r1, "id2": dense_r2},
            ...     sparse_results={"id2": sparse_r2},
            ...     top_k=10
            ... )
        """
        hybrid_results = []

        # Find ranks for each result
        dense_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(dense_results.values(), start=1)}
        sparse_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(sparse_results.values(), start=1)}

        # Build hybrid results
        for chunk_id, fused_score in fused_rankings[:top_k]:
            # Get data from available results
            dense_result = dense_results.get(chunk_id)
            sparse_result = sparse_results.get(chunk_id)

            # Use dense result as base (it has full metadata)
            if dense_result:
                base_result = dense_result
                base_score = dense_result.score
                base_rank = dense_ranks.get(chunk_id, 0)
                sparse_score = sparse_result.score if sparse_result else 0.0
                sparse_rank = sparse_ranks.get(chunk_id, 0)
            elif sparse_result:
                base_result = sparse_result
                base_score = 0.0
                base_rank = 0
                sparse_score = sparse_result.score
                sparse_rank = sparse_ranks.get(chunk_id, 0)
            else:
                # Shouldn't happen if RRF is working correctly
                logger.warning(f"Chunk {chunk_id} in fused results but not in source results")
                continue

            # Create hybrid result
            hybrid_result = HybridSearchResult(
                doc_id=base_result.doc_id,
                chunk_id=chunk_id,
                content=base_result.content,
                metadata=base_result.metadata,
                dense_score=base_score,
                sparse_score=sparse_score,
                fused_score=fused_score,
                dense_rank=base_rank,
                sparse_rank=sparse_rank,
            )
            hybrid_results.append(hybrid_result)

        return hybrid_results

    # ============================================================
    # Document Management
    # ============================================================

    def add_documents(self, documents: list[Document]) -> int:
        """
        Add documents to both dense and sparse indexes.

        Args:
            documents: List of Document objects to add

        Returns:
            Number of documents added

        Example:
            >>> count = retriever.add_documents(documents)
            >>> print(f"Added {count} documents to both indexes")
        """
        if not documents:
            return 0

        logger.info(f"Adding {len(documents)} documents to hybrid retriever")

        # Generate embeddings
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_service.embed_texts(texts)

        # Add to vector store (dense)
        self.vector_store.add_documents(documents, embeddings)

        # Add to BM25 (sparse)
        self.bm25_retriever.add_documents(documents)

        logger.info(f"Successfully added {len(documents)} documents to both indexes")

        return len(documents)

    def delete_documents(self, doc_ids: list[str]) -> int:
        """
        Delete documents from both indexes.

        Args:
            doc_ids: List of document IDs to delete

        Returns:
            Number of chunks deleted from dense index

        Example:
            >>> deleted = retriever.delete_documents(["doc_123", "doc_456"])
            >>> print(f"Deleted {deleted} chunks")
        """
        if not doc_ids:
            return 0

        logger.info(f"Deleting {len(doc_ids)} documents from hybrid retriever")

        # Delete from vector store
        dense_deleted = self.vector_store.delete(doc_ids)

        # Note: BM25 retriever doesn't have direct delete by doc_id
        # We would need to get chunks for each doc and delete those
        # For now, log a warning
        logger.warning(
            "BM25 index does not support efficient document deletion. "
            "Consider rebuilding the index if deletions are frequent."
        )

        logger.info(f"Deleted {dense_deleted} chunks from vector store")

        return dense_deleted

    # ============================================================
    # Statistics
    # ============================================================

    def get_stats(self) -> HybridRetrieverStats:
        """
        Get retriever statistics.

        Returns:
            HybridRetrieverStats object with current statistics

        Example:
            >>> stats = retriever.get_stats()
            >>> print(f"Total queries: {stats.total_queries}")
            >>> print(f"Avg query time: {stats.avg_query_time_ms:.1f}ms")
        """
        return self._stats

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = HybridRetrieverStats()
        logger.info("Hybrid retriever statistics reset")


# ============================================================
# Utility Functions
# ============================================================

def create_hybrid_retriever(
    vector_store: VectorStoreBase,
    embedding_service: EmbeddingService,
    bm25_retriever: BM25Retriever,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> HybridRetriever:
    """
    Create a hybrid retriever configured from settings.

    Args:
        vector_store: Dense vector store instance
        embedding_service: Embedding service instance
        bm25_retriever: BM25 retriever instance
        dense_weight: Weight for dense results
        sparse_weight: Weight for sparse results

    Returns:
        Configured HybridRetriever instance

    Example:
        >>> retriever = create_hybrid_retriever(
        ...     vector_store=vector_store,
        ...     embedding_service=embedding_service,
        ...     bm25_retriever=bm25_retriever
        ... )
    """
    return HybridRetriever(
        vector_store=vector_store,
        embedding_service=embedding_service,
        bm25_retriever=bm25_retriever,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        rrf_k=60,  # Default RRF constant
    )


# Export public API
__all__ = [
    # Data classes
    "HybridSearchResult",
    "HybridRetrieverStats",
    # Main class
    "HybridRetriever",
    # Utilities
    "create_hybrid_retriever",
]
