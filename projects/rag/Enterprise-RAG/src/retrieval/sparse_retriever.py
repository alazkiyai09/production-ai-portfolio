# ============================================================
# Enterprise-RAG: BM25 Sparse Retriever
# ============================================================
"""
BM25 sparse retrieval for hybrid search with dense vector retrieval.

This module provides:
- BM25 index building with tokenization
- Persistent index storage
- Sparse (keyword-based) search
- Score normalization for hybrid combination
- Easy integration with dense retrievers

BM25 is a ranking function that estimates relevance of documents to search
queries based on term frequency and inverse document frequency. It's
particularly effective for exact keyword matching.

Example:
    >>> from src.retrieval.sparse_retriever import BM25Retriever
    >>> retriever = BM25Retriever(index_path="./data/bm25_index.pkl")
    >>> retriever.build_index(documents)
    >>> results = retriever.search("refund policy", top_k=10)
    >>> for result in results:
    ...     print(f"{result.chunk_id}: {result.score:.3f}")
"""

import pickle
import re
import string
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import settings
from src.ingestion.document_processor import Document
from src.logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)


# ============================================================
# Data Classes
# ============================================================

@dataclass(frozen=True)
class SparseSearchResult:
    """
    Result from BM25 sparse search.

    BM25 scores are raw relevance scores that can be normalized to 0-1 range
    for combining with dense vector similarity scores.

    Attributes:
        doc_id: Source document ID
        chunk_id: Unique chunk identifier
        content: Text content of the chunk
        score: BM25 relevance score (higher = more relevant)

    Example:
        >>> result = SparseSearchResult(
        ...     doc_id="doc_123",
        ...     chunk_id="doc_123_chunk_0",
        ...     content="Refund policy...",
        ...     score=5.23
        ... )
    """

    doc_id: str
    chunk_id: str
    content: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SparseSearchResult":
        """Create SparseSearchResult from dictionary."""
        return cls(
            doc_id=data["doc_id"],
            chunk_id=data["chunk_id"],
            content=data["content"],
            score=data["score"],
        )


@dataclass
class BM25Stats:
    """
    Statistics for the BM25 retriever.

    Attributes:
        total_documents: Number of documents in the index
        total_chunks: Number of chunks in the index
        avg_doc_length: Average document length (tokens)
        vocabulary_size: Number of unique terms in vocabulary
        index_size_bytes: Size of index in memory (approximate)
        last_updated: Timestamp of last update
    """

    total_documents: int = 0
    total_chunks: int = 0
    avg_doc_length: float = 0.0
    vocabulary_size: int = 0
    index_size_bytes: int = 0
    last_updated: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_documents": self.total_documents,
            "total_chunks": self.total_chunks,
            "avg_doc_length": round(self.avg_doc_length, 2),
            "vocabulary_size": self.vocabulary_size,
            "index_size_mb": round(self.index_size_bytes / 1024 / 1024, 2),
            "last_updated": self.last_updated,
        }


# ============================================================
# BM25 Retriever
# ============================================================

class BM25Retriever:
    """
    BM25 sparse retriever for keyword-based search.

    BM25 (Best Matching 25) is a probabilistic retrieval function that ranks
    documents based on query terms appearing in each document. It's particularly
    effective for:
    - Exact keyword matching
    - Technical queries with specific terms
    - Complementing dense vector search in hybrid systems

    Features:
        - Persistent index storage
        - Configurable tokenization
        - Score normalization
        - Thread-safe operations
        - Index statistics

    Args:
        index_path: Path to save/load index (optional)
        k1: BM25 k1 parameter (term frequency saturation, default: 1.5)
        b: BM25 b parameter (length normalization, default: 0.75)
        normalize_scores: Whether to normalize scores to 0-1 range

    Example:
        >>> retriever = BM25Retriever(
        ...     index_path="./data/bm25_index.pkl",
        ...     k1=1.5,
        ...     b=0.75
        ... )
        >>> retriever.build_index(documents)
        >>> results = retriever.search("machine learning algorithms", top_k=10)
    """

    # Basic English stopwords
    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "will", "with", "the", "this", "but", "they", "have",
        "had", "what", "when", "where", "who", "which", "why", "how",
        "all", "each", "every", "both", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "can", "just", "should", "now", "i", "you",
        "your", "we", "our", "their", "them", "his", "her", "him", "me",
        "my", "your", "ours", "mine", "yours", "hers", "theirs",
    }

    def __init__(
        self,
        index_path: Optional[str] = None,
        k1: Optional[float] = None,
        b: Optional[float] = None,
        normalize_scores: bool = True,
    ) -> None:
        """
        Initialize the BM25 retriever.

        Args:
            index_path: Path to save/load index (default: None)
            k1: BM25 k1 parameter (uses settings if None)
            b: BM25 b parameter (uses settings if None)
            normalize_scores: Whether to normalize scores to 0-1 range
        """
        # Configuration
        self.index_path = index_path
        self.k1 = k1 or settings.BM25_K1
        self.b = b or settings.BM25_B
        self.normalize_scores = normalize_scores

        # Index data
        self.bm25: Optional[BM25Okapi] = None
        self.documents: list[Document] = []
        self.doc_ids: set[str] = set()
        self.tokenized_corpus: list[list[str]] = []

        # Statistics
        self._stats = BM25Stats()
        self._lock = threading.Lock()

        # Load existing index if path provided
        if self.index_path and Path(self.index_path).exists():
            self.load_index(self.index_path)

        logger.info(
            "BM25Retriever initialized",
            extra={
                "index_path": self.index_path,
                "k1": self.k1,
                "b": self.b,
                "normalize": self.normalize_scores,
            },
        )

    # ============================================================
    # Index Building
    # ============================================================

    def build_index(self, documents: list[Document]) -> None:
        """
        Build BM25 index from documents.

        This clears any existing index and builds a new one from scratch.

        Args:
            documents: List of Document objects to index

        Raises:
            ValueError: If documents list is empty

        Example:
            >>> retriever = BM25Retriever()
            >>> retriever.build_index(documents)
            >>> print(f"Indexed {len(documents)} documents")
        """
        if not documents:
            raise ValueError("Cannot build index from empty documents list")

        start_time = time.time()
        logger.info(f"Building BM25 index from {len(documents)} documents")

        with self._lock:
            # Clear existing index
            self.documents = []
            self.doc_ids = set()
            self.tokenized_corpus = []

            # Track document IDs for stats
            unique_doc_ids = set()

            # Tokenize all documents
            for doc in documents:
                self.documents.append(doc)
                self.doc_ids.add(doc.chunk_id)
                unique_doc_ids.add(doc.doc_id)

                # Tokenize content
                tokens = self._tokenize(doc.content)
                self.tokenized_corpus.append(tokens)

            # Build BM25 index
            self.bm25 = BM25Okapi(
                self.tokenized_corpus,
                k1=self.k1,
                b=self.b,
                epsilon=0.25,
            )

            # Update statistics
            elapsed = time.time() - start_time
            self._update_stats(unique_doc_ids)

            logger.info(
                f"BM25 index built in {elapsed:.2f}s",
                extra={
                    "documents": len(documents),
                    "unique_docs": len(unique_doc_ids),
                    "vocabulary_size": self._stats.vocabulary_size,
                    "avg_doc_length": round(self._stats.avg_doc_length, 2),
                },
            )

    def add_documents(self, documents: list[Document]) -> None:
        """
        Add documents to the existing index.

        Note: This rebuilds the entire index, which can be expensive for
        large collections. For frequent updates, consider batching adds
        and rebuilding less frequently.

        Args:
            documents: List of Document objects to add

        Example:
            >>> # Index initial documents
            >>> retriever.build_index(initial_docs)
            >>> # Add more documents later
            >>> retriever.add_documents(new_docs)
        """
        if not documents:
            logger.warning("No documents to add")
            return

        logger.info(f"Adding {len(documents)} documents to BM25 index")

        with self._lock:
            # Filter out documents that already exist
            new_docs = [doc for doc in documents if doc.chunk_id not in self.doc_ids]

            if not new_docs:
                logger.info("All documents already exist in index")
                return

            # Rebuild index with new documents
            all_documents = self.documents + new_docs
            self.build_index(all_documents)

            logger.info(
                f"Added {len(new_docs)} new documents (total: {len(self.documents)})"
            )

    def remove_documents(self, chunk_ids: list[str]) -> int:
        """
        Remove documents from the index.

        Note: This rebuilds the entire index without the specified documents.

        Args:
            chunk_ids: List of chunk IDs to remove

        Returns:
            Number of documents removed

        Example:
            >>> removed = retriever.remove_documents(["doc_123_chunk_0", "doc_123_chunk_1"])
            >>> print(f"Removed {removed} chunks")
        """
        if not chunk_ids:
            return 0

        chunk_ids_set = set(chunk_ids)
        logger.info(f"Removing {len(chunk_ids_set)} documents from BM25 index")

        with self._lock:
            # Filter out documents to remove
            old_count = len(self.documents)
            self.documents = [
                doc for doc in self.documents if doc.chunk_id not in chunk_ids_set
            ]
            self.doc_ids = {doc.chunk_id for doc in self.documents}

            removed_count = old_count - len(self.documents)

            if removed_count > 0:
                # Rebuild index
                if self.documents:
                    self.tokenized_corpus = [
                        self._tokenize(doc.content) for doc in self.documents
                    ]
                    self.bm25 = BM25Okapi(
                        self.tokenized_corpus,
                        k1=self.k1,
                        b=self.b,
                        epsilon=0.25,
                    )
                else:
                    self.bm25 = None
                    self.tokenized_corpus = []

                # Update stats
                self._stats.total_chunks = len(self.documents)
                self._stats.last_updated = time.strftime("%Y-%m-%d %H:%M:%S")

            logger.info(f"Removed {removed_count} documents from index")
            return removed_count

        return 0

    # ============================================================
    # Search
    # ============================================================

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SparseSearchResult]:
        """
        Search using BM25 sparse retrieval.

        Args:
            query: Search query string
            top_k: Number of results to return
            filters: Optional metadata filters (e.g., {"file_type": "pdf"})

        Returns:
            List of SparseSearchResult objects, sorted by score (descending)

        Raises:
            ValueError: If index hasn't been built yet

        Example:
            >>> results = retriever.search("machine learning algorithms", top_k=5)
            >>> for result in results:
            ...     print(f"{result.chunk_id}: {result.score:.3f}")
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build_index() first.")

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        start_time = time.time()

        # Tokenize query
        tokenized_query = self._tokenize(query)

        if not tokenized_query:
            logger.warning(f"Query produced no tokens after tokenization: {query}")
            return []

        logger.debug(
            f"BM25 search for query: {query[:100]}",
            extra={
                "query_length": len(query),
                "tokens": len(tokenized_query),
                "top_k": top_k,
            },
        )

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Normalize scores if requested
        if self.normalize_scores:
            scores = self._normalize_scores(scores)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Filter by score (ignore zero/negative scores)
        top_indices = [idx for idx in top_indices if scores[idx] > 0]

        # Apply metadata filters if provided
        if filters:
            top_indices = [
                idx
                for idx in top_indices
                if self._matches_filters(self.documents[idx].metadata, filters)
            ]

        # Build results
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            result = SparseSearchResult(
                doc_id=doc.doc_id,
                chunk_id=doc.chunk_id,
                content=doc.content,
                score=float(scores[idx]),
            )
            results.append(result)

        elapsed = time.time() - start_time
        logger.debug(
            f"BM25 search found {len(results)} results in {elapsed:.3f}s",
            extra={
                "results_count": len(results),
                "elapsed": round(elapsed, 3),
            },
        )

        return results

    # ============================================================
    # Tokenization
    # ============================================================

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text for BM25 indexing and search.

        Tokenization steps:
        1. Convert to lowercase
        2. Remove punctuation
        3. Split into words
        4. Remove stopwords
        5. Filter short tokens (< 2 chars)

        Args:
            text: Text to tokenize

        Returns:
            List of tokens

        Example:
            >>> tokens = retriever._tokenize("Hello, World! This is a test.")
            >>> print(tokens)  # ['hello', 'world', 'test']
        """
        if not text:
            return []

        # Convert to lowercase
        text = text.lower()

        # Remove punctuation (but keep alphanumeric and spaces)
        text = re.sub(r"[^\w\s]", " ", text)

        # Remove digits (optional, depending on use case)
        # text = re.sub(r"\d+", "", text)

        # Split into words
        tokens = text.split()

        # Remove stopwords
        tokens = [token for token in tokens if token not in self.STOPWORDS]

        # Filter short tokens
        tokens = [token for token in tokens if len(token) >= 2]

        return tokens

    # ============================================================
    # Score Normalization
    # ============================================================

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize BM25 scores to 0-1 range.

        Uses min-max normalization: (score - min) / (max - min)

        Args:
            scores: Raw BM25 scores

        Returns:
            Normalized scores in [0, 1] range

        Example:
            >>> normalized = retriever._normalize_scores(np.array([1.5, 3.2, 0.8]))
            >>> print(normalized)
        """
        if scores.size == 0:
            return scores

        min_score = scores.min()
        max_score = scores.max()

        if max_score == min_score:
            # All scores are the same
            return np.ones_like(scores) if max_score > 0 else np.zeros_like(scores)

        # Min-max normalization
        normalized = (scores - min_score) / (max_score - min_score)

        return normalized

    # ============================================================
# ============================================================
# Metadata Filtering
# ============================================================

    def _matches_filters(self, metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        """
        Check if document metadata matches filters.

        Args:
            metadata: Document metadata
            filters: Filter criteria

        Returns:
            True if all filters match

        Example:
            >>> matches = retriever._matches_filters(
            ...     {"file_type": "pdf", "page": 1},
            ...     {"file_type": "pdf"}
            ... )
        """
        for key, value in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True

    # ============================================================
    # Index Persistence
    # ============================================================

    def save_index(self, path: Optional[str] = None) -> None:
        """
        Persist index to disk.

        Args:
            path: Path to save index (uses self.index_path if None)

        Raises:
            ValueError: If no path is available

        Example:
            >>> retriever.save_index("./data/bm25_index.pkl")
        """
        save_path = path or self.index_path

        if not save_path:
            raise ValueError("No index path specified")

        logger.info(f"Saving BM25 index to {save_path}")

        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            try:
                index_data = {
                    "documents": self.documents,
                    "doc_ids": list(self.doc_ids),
                    "tokenized_corpus": self.tokenized_corpus,
                    "k1": self.k1,
                    "b": self.b,
                    "stats": self._stats.to_dict(),
                    "version": "1.0",
                }

                with open(save_path, "wb") as f:
                    pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)

                # Update index path if this was first save
                if self.index_path is None:
                    self.index_path = save_path

                logger.info(f"BM25 index saved successfully ({len(self.documents)} documents)")

            except Exception as e:
                logger.error(f"Failed to save index: {str(e)}", exc_info=True)
                raise

    def load_index(self, path: str) -> None:
        """
        Load index from disk.

        Args:
            path: Path to load index from

        Raises:
            FileNotFoundError: If index file doesn't exist
            ValueError: If index file is invalid

        Example:
            >>> retriever.load_index("./data/bm25_index.pkl")
            >>> print(f"Loaded {len(retriever.documents)} documents")
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        logger.info(f"Loading BM25 index from {path}")

        with self._lock:
            try:
                with open(path, "rb") as f:
                    index_data = pickle.load(f)

                # Validate index data
                if "documents" not in index_data or "tokenized_corpus" not in index_data:
                    raise ValueError("Invalid index file format")

                # Load data
                self.documents = index_data["documents"]
                self.doc_ids = set(index_data.get("doc_ids", []))
                self.tokenized_corpus = index_data["tokenized_corpus"]
                self.k1 = index_data.get("k1", self.k1)
                self.b = index_data.get("b", self.b)

                # Rebuild BM25 index
                if self.tokenized_corpus:
                    self.bm25 = BM25Okapi(
                        self.tokenized_corpus,
                        k1=self.k1,
                        b=self.b,
                        epsilon=0.25,
                    )
                else:
                    self.bm25 = None

                # Load stats
                if "stats" in index_data:
                    stats_dict = index_data["stats"]
                    self._stats = BM25Stats(
                        total_documents=stats_dict.get("total_documents", 0),
                        total_chunks=stats_dict.get("total_chunks", 0),
                        avg_doc_length=stats_dict.get("avg_doc_length", 0.0),
                        vocabulary_size=stats_dict.get("vocabulary_size", 0),
                        index_size_bytes=stats_dict.get("index_size_bytes", 0),
                        last_updated=stats_dict.get("last_updated"),
                    )
                else:
                    self._update_stats(set(doc.doc_id for doc in self.documents))

                # Update index path
                self.index_path = path

                logger.info(
                    f"BM25 index loaded successfully ({len(self.documents)} documents)"
                )

            except Exception as e:
                logger.error(f"Failed to load index: {str(e)}", exc_info=True)
                raise

    # ============================================================
    # Statistics
    # ============================================================

    def get_stats(self) -> BM25Stats:
        """
        Get index statistics.

        Returns:
            BM25Stats object with current statistics

        Example:
            >>> stats = retriever.get_stats()
            >>> print(f"Documents: {stats.total_chunks}")
            >>> print(f"Vocabulary: {stats.vocabulary_size}")
        """
        with self._lock:
            # Recalculate if stats are stale
            if self._stats.total_chunks != len(self.documents):
                unique_doc_ids = set(doc.doc_id for doc in self.documents)
                self._update_stats(unique_doc_ids)

            return self._stats

    def _update_stats(self, unique_doc_ids: set[str]) -> None:
        """Update index statistics."""
        self._stats.total_documents = len(unique_doc_ids)
        self._stats.total_chunks = len(self.documents)

        if self.tokenized_corpus:
            # Calculate average document length
            doc_lengths = [len(doc) for doc in self.tokenized_corpus]
            self._stats.avg_doc_length = np.mean(doc_lengths) if doc_lengths else 0.0

            # Calculate vocabulary size
            all_tokens = set()
            for doc_tokens in self.tokenized_corpus:
                all_tokens.update(doc_tokens)
            self._stats.vocabulary_size = len(all_tokens)
        else:
            self._stats.avg_doc_length = 0.0
            self._stats.vocabulary_size = 0.0

        # Estimate index size (rough approximation)
        self._stats.index_size_bytes = len(pickle.dumps(self.tokenized_corpus))

        self._stats.last_updated = time.strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# Utility Functions
# ============================================================

def create_bm25_retriever(
    index_path: Optional[str] = None,
) -> BM25Retriever:
    """
    Create a BM25 retriever configured from settings.

    Args:
        index_path: Path for index persistence

    Returns:
        Configured BM25Retriever instance

    Example:
        >>> retriever = create_bm25_retriever(index_path="./data/bm25.pkl")
        >>> retriever.build_index(documents)
    """
    return BM25Retriever(
        index_path=index_path,
        k1=settings.BM25_K1,
        b=settings.BM25_B,
        normalize_scores=True,
    )


# Export public API
__all__ = [
    # Data classes
    "SparseSearchResult",
    "BM25Stats",
    # Main class
    "BM25Retriever",
    # Utilities
    "create_bm25_retriever",
]
