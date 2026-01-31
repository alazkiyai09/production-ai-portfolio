# ============================================================
# Enterprise-RAG: Vector Store Abstraction Layer
# ============================================================
"""
Abstract vector store interface with ChromaDB implementation.

This module provides:
- Abstract base class for vector store implementations
- ChromaDB implementation for development
- Metadata filtering support
- Batch operations for efficiency
- Easy extensibility for Qdrant, Pinecone, etc.

Example:
    >>> from src.retrieval import create_vector_store, ChromaVectorStore
    >>> # Create store
    >>> store = ChromaVectorStore(collection_name="docs", persist_directory="./data/chroma")
    >>> # Add documents
    >>> store.add_documents(documents, embeddings)
    >>> # Search
    >>> results = store.search(query_embedding, top_k=5, filters={"file_type": "pdf"})
    >>> for result in results:
    ...     print(f"{result.chunk_id}: {result.score:.3f}")
"""

import hashlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import settings
from src.exceptions import RetrievalError
from src.ingestion.document_processor import Document
from src.logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)


# ============================================================
# Data Classes
# ============================================================

@dataclass(frozen=True)
class SearchResult:
    """
    Result from a vector similarity search.

    Attributes:
        doc_id: Source document ID
        chunk_id: Unique chunk identifier
        content: Text content of the chunk
        metadata: Associated metadata (source, page, etc.)
        score: Similarity score (higher = more similar)

    Example:
        >>> result = SearchResult(
        ...     doc_id="doc_123",
        ...     chunk_id="doc_123_chunk_0",
        ...     content="Sample text",
        ...     metadata={"source": "file.pdf"},
        ...     score=0.85
        ... )
    """

    doc_id: str
    chunk_id: str
    content: str
    metadata: dict[str, Any]
    score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchResult":
        """Create SearchResult from dictionary."""
        return cls(
            doc_id=data["doc_id"],
            chunk_id=data["chunk_id"],
            content=data["content"],
            metadata=data["metadata"],
            score=data["score"],
        )


@dataclass
class VectorStoreStats:
    """
    Statistics for a vector store.

    Attributes:
        total_documents: Number of documents in the store
        total_chunks: Number of chunks in the store
        collection_name: Name of the collection
        embedding_dimension: Dimension of embeddings
        last_updated: Timestamp of last update
    """

    total_documents: int = 0
    total_chunks: int = 0
    collection_name: str = ""
    embedding_dimension: Optional[int] = None
    last_updated: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_documents": self.total_documents,
            "total_chunks": self.total_chunks,
            "collection_name": self.collection_name,
            "embedding_dimension": self.embedding_dimension,
            "last_updated": self.last_updated,
        }


# ============================================================
# Abstract Base Class
# ============================================================

class VectorStoreBase(ABC):
    """
    Abstract base class for vector store implementations.

    Defines the interface that all vector store implementations must follow.
    This allows easy swapping between different vector databases (ChromaDB,
    Qdrant, Pinecone, Weaviate, etc.)

    Example:
        >>> class CustomVectorStore(VectorStoreBase):
        ...     def add_documents(self, documents, embeddings):
        ...         # Implementation
        ...         return len(documents)
        ...     # Implement other abstract methods...
    """

    @abstractmethod
    def add_documents(
        self,
        documents: list[Document],
        embeddings: Any,  # np.ndarray but avoid import
    ) -> int:
        """
        Add documents with their embeddings to the store.

        Args:
            documents: List of Document objects to add
            embeddings: Array of embeddings (shape: [n_docs, embedding_dim])

        Returns:
            Number of documents added

        Raises:
            RetrievalError: If add operation fails

        Example:
            >>> count = store.add_documents(documents, embeddings)
            >>> print(f"Added {count} documents")
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: Any,  # np.ndarray
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Optional metadata filters (e.g., {"file_type": "pdf"})

        Returns:
            List of SearchResult objects, sorted by score (descending)

        Raises:
            RetrievalError: If search operation fails

        Example:
            >>> results = store.search(query_emb, top_k=5, filters={"source": "docs/"})
            >>> for result in results:
            ...     print(f"{result.chunk_id}: {result.score:.3f}")
        """
        pass

    @abstractmethod
    def delete(self, doc_ids: list[str]) -> int:
        """
        Delete documents from the store.

        Args:
            doc_ids: List of document IDs to delete

        Returns:
            Number of documents deleted

        Raises:
            RetrievalError: If delete operation fails

        Example:
            >>> count = store.delete(["doc_123", "doc_456"])
            >>> print(f"Deleted {count} documents")
        """
        pass

    @abstractmethod
    def get_stats(self) -> VectorStoreStats:
        """
        Get statistics about the vector store.

        Returns:
            VectorStoreStats object with store information

        Example:
            >>> stats = store.get_stats()
            >>> print(f"Total chunks: {stats.total_chunks}")
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all documents from the store.

        Example:
            >>> store.clear()
            >>> print("Store cleared")
        """
        pass


# ============================================================
# ChromaDB Implementation
# ============================================================

class ChromaVectorStore(VectorStoreBase):
    """
    ChromaDB implementation of the vector store interface.

    ChromaDB is a lightweight, open-source vector database perfect for
    development and smaller production deployments.

    Features:
        - Automatic persistence to disk
        - Metadata filtering
        - No configuration required
        - Fast in-memory operations with disk backing

    Args:
        collection_name: Name of the collection
        persist_directory: Directory for persistent storage
        embedding_function: Optional custom embedding function (not used, we pre-embed)

    Example:
        >>> store = ChromaVectorStore(
        ...     collection_name="enterprise_rag",
        ...     persist_directory="./data/chroma"
        ... )
        >>> store.add_documents(documents, embeddings)
        >>> results = store.search(query_embedding, top_k=5)
    """

    # Class-level cache for ChromaDB clients (shared across instances)
    _client_cache: dict[str, chromadb.PersistentClient] = {}
    _collection_cache: dict[str, chromadb.Collection] = {}
    _lock = threading.Lock()

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        embedding_function: Optional[Any] = None,
    ) -> None:
        """
        Initialize the ChromaDB vector store.

        Args:
            collection_name: Name of the collection (uses settings if None)
            persist_directory: Path for persistent storage (uses settings if None)
            embedding_function: Optional (not used, embeddings provided directly)
        """
        # Configuration
        self.collection_name = collection_name or "enterprise_rag"
        self.persist_directory = persist_directory or str(settings.chroma_persist_path)

        # Ensure directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Statistics tracking
        self._stats = VectorStoreStats(collection_name=self.collection_name)
        self._stats_lock = threading.Lock()

        # Get or create client and collection
        self._client = self._get_client()
        self._collection = self._get_collection()

        logger.info(
            f"ChromaVectorStore initialized",
            extra={
                "collection": self.collection_name,
                "persist_dir": self.persist_directory,
            },
        )

    def _get_client(self) -> chromadb.PersistentClient:
        """Get or create ChromaDB client."""
        # Create cache key from persist directory
        cache_key = self.persist_directory

        with self._lock:
            if cache_key not in self._client_cache:
                logger.info(f"Creating ChromaDB client: {self.persist_directory}")
                self._client_cache[cache_key] = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=ChromaSettings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                    ),
                )
            return self._client_cache[cache_key]

    def _get_collection(self) -> chromadb.Collection:
        """Get or create collection."""
        with self._lock:
            # Try to get existing collection
            try:
                collection = self._client.get_collection(name=self.collection_name)
                logger.info(f"Using existing collection: {self.collection_name}")

                # Update stats from existing collection
                self._update_stats_from_collection(collection)

                return collection

            except Exception:
                # Collection doesn't exist, create it
                logger.info(f"Creating new collection: {self.collection_name}")
                collection = self._client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Enterprise RAG document store"},
                )

                # Cache the collection
                self._collection_cache[self.collection_name] = collection
                return collection

    def _update_stats_from_collection(self, collection: chromadb.Collection) -> None:
        """Update stats from existing collection."""
        try:
            count = collection.count()
            with self._stats_lock:
                self._stats.total_chunks = count

                # Try to get unique document count
                try:
                    # This is an approximation since ChromaDB doesn't have distinct count
                    results = collection.get(limit=1, include=["metadatas"])
                    if results and results["metadatas"]:
                        self._stats.embedding_dimension = len(
                            results.get("embeddings", [[]])[0] if results.get("embeddings") else []
                        )
                except Exception:
                    pass

                self._stats.last_updated = time.strftime("%Y-%m-%d %H:%M:%S")

        except Exception as e:
            logger.warning(f"Failed to update stats from collection: {e}")

    # ============================================================
    # CRUD Operations
    # ============================================================

    def add_documents(
        self,
        documents: list[Document],
        embeddings: Any,
    ) -> int:
        """
        Add documents with their embeddings to the store.

        Args:
            documents: List of Document objects
            embeddings: Array of embeddings (numpy array or list of lists)

        Returns:
            Number of documents added

        Raises:
            RetrievalError: If add operation fails

        Example:
            >>> count = store.add_documents(documents, embeddings)
            >>> print(f"Added {count} documents")
        """
        if not documents:
            logger.warning("No documents to add")
            return 0

        if len(documents) != len(embeddings):
            raise ValueError(
                f"Number of documents ({len(documents)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )

        start_time = time.time()
        logger.info(f"Adding {len(documents)} documents to collection {self.collection_name}")

        try:
            # Prepare data for ChromaDB
            ids = []
            texts = []
            metadatas = []
            embedding_list = []

            # Track unique documents
            doc_ids_set = set()

            for doc, emb in zip(documents, embeddings):
                ids.append(doc.chunk_id)
                texts.append(doc.content)
                metadatas.append(doc.metadata)
                embedding_list.append(emb.tolist() if hasattr(emb, "tolist") else list(emb))
                doc_ids_set.add(doc.doc_id)

            # Add to collection in batches (ChromaDB handles batching internally)
            self._collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embedding_list,
            )

            # Update stats
            elapsed = time.time() - start_time
            with self._stats_lock:
                self._stats.total_chunks += len(documents)
                self._stats.total_documents += len(doc_ids_set)
                self._stats.last_updated = time.strftime("%Y-%m-%d %H:%M:%S")

                # Set embedding dimension from first batch
                if self._stats.embedding_dimension is None and len(embedding_list) > 0:
                    self._stats.embedding_dimension = len(embedding_list[0])

            logger.info(
                f"Added {len(documents)} chunks from {len(doc_ids_set)} documents in {elapsed:.2f}s",
                extra={
                    "chunks_added": len(documents),
                    "documents_added": len(doc_ids_set),
                    "elapsed": round(elapsed, 2),
                },
            )

            return len(documents)

        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}", exc_info=True)
            raise RetrievalError(
                message=f"Failed to add documents to ChromaDB: {str(e)}",
                details={"collection": self.collection_name, "num_documents": len(documents)},
            )

    def search(
        self,
        query_embedding: Any,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector (numpy array or list)
            top_k: Number of results to return
            filters: Optional metadata filters

                Supported filters:
                - {"file_type": "pdf"}
                - {"source": "docs/"}
                - {"page": 1}
                - Combined: {"file_type": "pdf", "page": 1}

        Returns:
            List of SearchResult objects, sorted by score (descending)

        Raises:
            RetrievalError: If search operation fails

        Example:
            >>> results = store.search(query_emb, top_k=5, filters={"file_type": "pdf"})
            >>> for result in results:
            ...     print(f"{result.chunk_id}: {result.score:.3f} - {result.content[:50]}...")
        """
        start_time = time.time()

        # Convert embedding to list if needed
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()

        logger.debug(
            f"Searching collection {self.collection_name}",
            extra={"top_k": top_k, "filters": filters},
        )

        try:
            # Build where clause for filters
            where_clause = self._build_where_clause(filters)

            # Query the collection
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"],
            )

            # Parse results
            search_results = []
            if results and results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    # Convert distance to similarity score
                    # ChromaDB returns L2 distance, convert to cosine-like score
                    distance = results["distances"][0][i]
                    score = self._distance_to_score(distance)

                    # Get metadata
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}

                    # Get content
                    content = results["documents"][0][i] if results["documents"] else ""

                    # Extract doc_id from metadata
                    doc_id = metadata.get("doc_id", chunk_id.split("_chunk_")[0])

                    result = SearchResult(
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        content=content,
                        metadata=metadata,
                        score=score,
                    )
                    search_results.append(result)

            elapsed = time.time() - start_time
            logger.debug(
                f"Found {len(search_results)} results in {elapsed:.3f}s",
                extra={
                    "results_count": len(search_results),
                    "elapsed": round(elapsed, 3),
                },
            )

            return search_results

        except Exception as e:
            logger.error(f"Search failed: {str(e)}", exc_info=True)
            raise RetrievalError(
                message=f"ChromaDB search failed: {str(e)}",
                details={"collection": self.collection_name, "top_k": top_k, "filters": filters},
            )

    def delete(self, doc_ids: list[str]) -> int:
        """
        Delete documents from the store.

        This deletes all chunks belonging to the specified documents.

        Args:
            doc_ids: List of document IDs to delete

        Returns:
            Number of chunks deleted

        Raises:
            RetrievalError: If delete operation fails

        Example:
            >>> count = store.delete(["doc_abc123", "doc_def456"])
            >>> print(f"Deleted {count} chunks")
        """
        if not doc_ids:
            return 0

        start_time = time.time()
        logger.info(f"Deleting {len(doc_ids)} documents from collection {self.collection_name}")

        try:
            # ChromaDB doesn't support direct deletion by metadata
            # We need to query for chunks with matching doc_id and delete by chunk_id

            total_deleted = 0

            for doc_id in doc_ids:
                # Get all chunks for this document
                # Note: ChromaDB's where clause doesn't support all operators
                # We'll need to get chunks and filter manually

                # Get all documents with this doc_id in metadata
                try:
                    # Get all IDs (ChromaDB limitation: need to use get with where)
                    results = self._collection.get(
                        where={"doc_id": doc_id},
                        limit=10000,  # Large number to get all chunks
                    )

                    if results and results["ids"]:
                        # Delete by chunk IDs
                        self._collection.delete(ids=results["ids"])
                        total_deleted += len(results["ids"])
                        logger.debug(f"Deleted {len(results['ids'])} chunks for doc {doc_id}")

                except Exception as e:
                    logger.warning(f"Failed to delete chunks for doc {doc_id}: {e}")

            # Update stats
            elapsed = time.time() - start_time
            with self._stats_lock:
                self._stats.total_chunks = max(0, self._stats.total_chunks - total_deleted)
                # Can't easily update doc count without tracking
                self._stats.last_updated = time.strftime("%Y-%m-%d %H:%M:%S")

            logger.info(
                f"Deleted {total_deleted} chunks in {elapsed:.2f}s",
                extra={
                    "chunks_deleted": total_deleted,
                    "docs_requested": len(doc_ids),
                    "elapsed": round(elapsed, 2),
                },
            )

            return total_deleted

        except Exception as e:
            logger.error(f"Delete failed: {str(e)}", exc_info=True)
            raise RetrievalError(
                message=f"Failed to delete from ChromaDB: {str(e)}",
                details={"collection": self.collection_name, "doc_ids": doc_ids},
            )

    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> int:
        """
        Delete specific chunks by their chunk IDs.

        This is more efficient than delete() for removing specific chunks.

        Args:
            chunk_ids: List of chunk IDs to delete

        Returns:
            Number of chunks deleted

        Example:
            >>> count = store.delete_by_chunk_ids(["doc_123_chunk_0", "doc_123_chunk_1"])
        """
        if not chunk_ids:
            return 0

        logger.info(f"Deleting {len(chunk_ids)} chunks")

        try:
            self._collection.delete(ids=chunk_ids)

            # Update stats
            with self._stats_lock:
                self._stats.total_chunks = max(0, self._stats.total_chunks - len(chunk_ids))
                self._stats.last_updated = time.strftime("%Y-%m-%d %H:%M:%S")

            logger.info(f"Deleted {len(chunk_ids)} chunks")
            return len(chunk_ids)

        except Exception as e:
            logger.error(f"Delete by chunk IDs failed: {str(e)}", exc_info=True)
            raise RetrievalError(
                message=f"Failed to delete chunks: {str(e)}",
                details={"chunk_ids": chunk_ids},
            )

    # ============================================================
    # Statistics and Utilities
    # ============================================================

    def get_stats(self) -> VectorStoreStats:
        """
        Get statistics about the vector store.

        Returns:
            VectorStoreStats object with current statistics

        Example:
            >>> stats = store.get_stats()
            >>> print(f"Chunks: {stats.total_chunks}, Docs: {stats.total_documents}")
        """
        # Refresh from collection
        try:
            count = self._collection.count()
            with self._stats_lock:
                self._stats.total_chunks = count
        except Exception as e:
            logger.warning(f"Failed to get collection count: {e}")

        return self._stats

    def clear(self) -> None:
        """
        Clear all documents from the collection.

        Example:
            >>> store.clear()
            >>> print("Collection cleared")
        """
        logger.warning(f"Clearing all documents from collection {self.collection_name}")

        try:
            # Delete and recreate collection
            self._client.delete_collection(name=self.collection_name)
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"description": "Enterprise RAG document store"},
            )

            # Update cache
            self._collection_cache[self.collection_name] = self._collection

            # Reset stats
            with self._stats_lock:
                self._stats = VectorStoreStats(collection_name=self.collection_name)

            logger.info(f"Collection {self.collection_name} cleared")

        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}", exc_info=True)
            raise RetrievalError(
                message=f"Failed to clear collection: {str(e)}",
                details={"collection": self.collection_name},
            )

    def get_document_chunks(self, doc_id: str) -> list[SearchResult]:
        """
        Get all chunks for a specific document.

        Args:
            doc_id: Document ID to fetch chunks for

        Returns:
            List of all chunks for the document

        Example:
            >>> chunks = store.get_document_chunks("doc_abc123")
            >>> print(f"Document has {len(chunks)} chunks")
        """
        try:
            results = self._collection.get(
                where={"doc_id": doc_id},
                limit=10000,
                include=["documents", "metadatas"],
            )

            chunks = []
            if results and results["ids"]:
                for i, chunk_id in enumerate(results["ids"]):
                    chunks.append(
                        SearchResult(
                            doc_id=doc_id,
                            chunk_id=chunk_id,
                            content=results["documents"][i] if results["documents"] else "",
                            metadata=results["metadatas"][i] if results["metadatas"] else {},
                            score=1.0,  # Perfect score for exact match
                        )
                    )

            return chunks

        except Exception as e:
            logger.error(f"Failed to get document chunks: {str(e)}", exc_info=True)
            return []

    # ============================================================
    # Helper Methods
    # ============================================================

    def _build_where_clause(self, filters: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """
        Build ChromaDB where clause from filters.

        ChromaDB supports:
        - Simple equality: {"key": "value"}
        - Comparison operators: {"key": {"$eq": "value", "$ne": "value", "$gt": 0, etc.}}

        Args:
            filters: Filter dictionary

        Returns:
            ChromaDB-compatible where clause

        Example:
            >>> where = store._build_where_clause({"file_type": "pdf", "page": 1})
        """
        if not filters:
            return None

        # ChromaDB filters need specific structure
        # For now, support simple equality filters
        where = {}
        for key, value in filters.items():
            if isinstance(value, (str, int, float, bool)):
                where[key] = value
            else:
                # Complex filter, pass through as-is
                where[key] = value

        return where if where else None

    def _distance_to_score(self, distance: float) -> float:
        """
        Convert ChromaDB L2 distance to a similarity score.

        ChromaDB returns L2 distance (lower = more similar).
        Convert to a score where higher = more similar.

        Uses formula: score = 1 / (1 + distance)

        Args:
            distance: L2 distance from ChromaDB

        Returns:
            Similarity score (0-1, higher = better)
        """
        # Convert L2 distance to similarity score
        # Score ranges from 0 to 1, where 1 is identical
        return 1.0 / (1.0 + distance)

    def _score_to_distance(self, score: float) -> float:
        """
        Convert similarity score to ChromaDB L2 distance.

        Args:
            score: Similarity score (0-1)

        Returns:
            L2 distance
        """
        if score <= 0:
            return float("inf")
        return (1.0 / score) - 1.0


# ============================================================
# Factory Function
# ============================================================

def create_vector_store(
    store_type: str = "chroma",
    **kwargs: Any,
) -> VectorStoreBase:
    """
    Create a vector store instance.

    Args:
        store_type: Type of vector store ("chroma", "qdrant", etc.)
        **kwargs: Additional arguments passed to the store constructor

    Returns:
        VectorStoreBase instance

    Raises:
        ValueError: If store_type is not supported

    Example:
        >>> # ChromaDB (default)
        >>> store = create_vector_store(store_type="chroma", collection_name="docs")
        >>>
        >>> # Custom configuration
        >>> store = create_vector_store(
        ...     store_type="chroma",
        ...     collection_name="enterprise_rag",
        ...     persist_directory="./data/chroma"
        ... )
    """
    store_type = store_type.lower()

    if store_type == "chroma":
        return ChromaVectorStore(**kwargs)
    elif store_type == "qdrant":
        # Future: implement QdrantVectorStore
        raise NotImplementedError(
            "Qdrant store not yet implemented. Use 'chroma' or implement QdrantVectorStore."
        )
    else:
        raise ValueError(
            f"Unsupported store type: {store_type}. "
            f"Supported types: ['chroma']"
        )


def create_vector_store_from_settings() -> VectorStoreBase:
    """
    Create a vector store configured from application settings.

    Returns:
        Configured VectorStoreBase instance

    Example:
        >>> store = create_vector_store_from_settings()
        >>> results = store.search(query_embedding, top_k=10)
    """
    store_type = settings.VECTOR_STORE_TYPE

    if store_type == "chroma":
        return ChromaVectorStore(
            collection_name="enterprise_rag",
            persist_directory=str(settings.chroma_persist_path),
        )
    else:
        raise ValueError(f"Unsupported store type in settings: {store_type}")


# ============================================================
# Utility Functions
# ============================================================

def reset_collection(
    collection_name: str = "enterprise_rag",
    persist_directory: Optional[str] = None,
) -> None:
    """
    Reset a collection by deleting and recreating it.

    Warning: This deletes all data in the collection!

    Args:
        collection_name: Name of the collection to reset
        persist_directory: Path to ChromaDB storage

    Example:
        >>> reset_collection("enterprise_rag", "./data/chroma")
        >>> print("Collection reset")
    """
    logger.warning(f"Resetting collection: {collection_name}")

    persist_dir = persist_directory or str(settings.chroma_persist_path)

    try:
        client = chromadb.PersistentClient(path=persist_dir)

        # Delete if exists
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass  # Collection doesn't exist

        # Create new collection
        client.create_collection(
            name=collection_name,
            metadata={"description": "Enterprise RAG document store"},
        )

        logger.info(f"Collection {collection_name} reset successfully")

    except Exception as e:
        logger.error(f"Failed to reset collection: {str(e)}", exc_info=True)
        raise


# Export public API
__all__ = [
    # Data classes
    "SearchResult",
    "VectorStoreStats",
    # Abstract base
    "VectorStoreBase",
    # Implementations
    "ChromaVectorStore",
    # Factory functions
    "create_vector_store",
    "create_vector_store_from_settings",
    # Utilities
    "reset_collection",
]
