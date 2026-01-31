"""
Vector store operations for StreamProcess-Pipeline.

Supports ChromaDB and Qdrant for vector storage and similarity search.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings


# ============================================================================
# Vector Store Base
# ============================================================================

class VectorStore:
    """Base class for vector store operations."""

    async def add_texts(
        self,
        collection_name: str,
        texts: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add texts to vector store.

        Args:
            collection_name: Name of collection
            texts: List of text strings
            embeddings: Optional pre-computed embeddings
            metadatas: Optional metadata for each text
            ids: Optional IDs for each text

        Returns:
            List of inserted IDs
        """
        raise NotImplementedError

    async def search(
        self,
        collection_name: str,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Search for similar texts.

        Args:
            collection_name: Name of collection
            query: Query text
            n_results: Number of results to return
            where: Optional metadata filter

        Returns:
            Search results with scores
        """
        raise NotImplementedError

    async def delete(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Delete vectors from collection.

        Args:
            collection_name: Name of collection
            ids: Optional list of IDs to delete
            where: Optional metadata filter for deletion
        """
        raise NotImplementedError

    async def get_collection_size(self, collection_name: str) -> int:
        """
        Get number of vectors in collection.

        Args:
            collection_name: Name of collection

        Returns:
            Number of vectors
        """
        raise NotImplementedError


# ============================================================================
# ChromaDB Store
# ============================================================================

class ChromaStore(VectorStore):
    """
    ChromaDB vector store implementation.

    Provides vector storage and similarity search using ChromaDB.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8001,
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize ChromaDB client.

        Args:
            host: ChromaDB host
            port: ChromaDB port
            persist_directory: Optional persistence directory
        """
        self.host = host
        self.port = port
        self.persist_directory = persist_directory
        self._client: Optional[chromadb.Client] = None

    async def _get_client(self) -> chromadb.Client:
        """Get or create ChromaDB client."""
        if self._client is None:
            if self.persist_directory:
                # Persistent mode
                self._client = chromadb.Client(
                    settings=ChromaSettings(
                        chroma_db_impl="duckdb+parquet",
                        persist_directory=self.persist_directory,
                    )
                )
            else:
                # Client mode (connect to server)
                self._client = chromadb.HttpClient(
                    host=self.host,
                    port=self.port,
                )

        return self._client

    async def _get_or_create_collection(self, name: str) -> chromadb.Collection:
        """
        Get or create collection.

        Args:
            name: Collection name

        Returns:
            ChromaDB collection
        """
        client = await self._get_client()

        # Try to get existing collection
        try:
            collection = client.get_collection(name=name)
            return collection
        except Exception:
            # Create new collection
            collection = client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
            return collection

    async def add_texts(
        self,
        collection_name: str,
        texts: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add texts to ChromaDB.

        Args:
            collection_name: Name of collection
            texts: List of text strings
            embeddings: Optional pre-computed embeddings
            metadatas: Optional metadata for each text
            ids: Optional IDs for each text

        Returns:
            List of inserted IDs
        """
        collection = await self._get_or_create_collection(collection_name)

        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}_{hash(text) % 1000000}" for i, text in enumerate(texts)]

        # Add documents
        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        return ids

    async def search(
        self,
        collection_name: str,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Search ChromaDB for similar texts.

        Args:
            collection_name: Name of collection
            query: Query text
            n_results: Number of results to return
            where: Optional metadata filter
            embedding: Optional query embedding

        Returns:
            Search results with scores
        """
        collection = await self._get_or_create_collection(collection_name)

        results = collection.query(
            query_texts=[query] if embedding is None else None,
            query_embeddings=[embedding] if embedding else None,
            n_results=n_results,
            where=where,
        )

        return results

    async def delete(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Delete vectors from ChromaDB.

        Args:
            collection_name: Name of collection
            ids: Optional list of IDs to delete
            where: Optional metadata filter for deletion
        """
        collection = await self._get_or_create_collection(collection_name)

        collection.delete(
            ids=ids,
            where=where,
        )

    async def get_collection_size(self, collection_name: str) -> int:
        """
        Get number of vectors in ChromaDB collection.

        Args:
            collection_name: Name of collection

        Returns:
            Number of vectors
        """
        collection = await self._get_or_create_collection(collection_name)
        return collection.count()


# ============================================================================
# Qdrant Store (Optional)
# ============================================================================

class QdrantStore(VectorStore):
    """
    Qdrant vector store implementation.

    Provides vector storage and similarity search using Qdrant.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
    ):
        """
        Initialize Qdrant client.

        Args:
            host: Qdrant host
            port: Qdrant HTTP port
            grpc_port: Qdrant gRPC port
        """
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self._client = None

    async def _get_client(self):
        """Get or create Qdrant client."""
        if self._client is None:
            from qdrant_client import QdrantClient
            from qdrant_client.async_client import AsyncQdrantClient

            self._client = AsyncQdrantClient(
                host=self.host,
                port=self.grpc_port,
            )

        return self._client

    async def add_texts(
        self,
        collection_name: str,
        texts: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add texts to Qdrant.

        Note: This is a placeholder implementation.
        """
        raise NotImplementedError("Qdrant support not fully implemented")

    async def search(
        self,
        collection_name: str,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Search Qdrant for similar texts.

        Note: This is a placeholder implementation.
        """
        raise NotImplementedError("Qdrant support not fully implemented")

    async def delete(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Delete vectors from Qdrant.

        Note: This is a placeholder implementation.
        """
        raise NotImplementedError("Qdrant support not fully implemented")

    async def get_collection_size(self, collection_name: str) -> int:
        """
        Get number of vectors in Qdrant collection.

        Note: This is a placeholder implementation.
        """
        raise NotImplementedError("Qdrant support not fully implemented")


# ============================================================================
# Factory
# ============================================================================

_vector_store: Optional[VectorStore] = None


async def get_vector_store() -> VectorStore:
    """
    Get global vector store instance.

    Returns:
        VectorStore instance (ChromaDB or Qdrant)
    """
    global _vector_store

    if _vector_store is None:
        backend = os.getenv("VECTOR_STORE_BACKEND", "chroma").lower()

        if backend == "qdrant":
            _vector_store = QdrantStore(
                host=os.getenv("QDRANT_HOST", "localhost"),
                port=int(os.getenv("QDRANT_PORT", "6333")),
                grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
            )
        else:
            _vector_store = ChromaStore(
                host=os.getenv("CHROMA_HOST", "localhost"),
                port=int(os.getenv("CHROMA_PORT", "8001")),
                persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY"),
            )

    return _vector_store


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "VectorStore",
    "ChromaStore",
    "QdrantStore",
    "get_vector_store",
]
