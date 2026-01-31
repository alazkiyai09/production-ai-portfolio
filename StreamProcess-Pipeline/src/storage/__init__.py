"""
Storage layer for database and vector store operations.
"""

from src.storage.database import DatabaseManager, get_db, get_db_manager
from src.storage.models import Base, EventRecord, ProcessingStatus, EmbeddingCache
from src.storage.repositories import EventRepository, StatusRepository, EmbeddingCacheRepository
from src.storage.vector_store import VectorStore, ChromaStore, QdrantStore, get_vector_store

__all__ = [
    # Database
    "DatabaseManager",
    "get_db",
    "get_db_manager",
    # Models
    "Base",
    "EventRecord",
    "ProcessingStatus",
    "EmbeddingCache",
    # Repositories
    "EventRepository",
    "StatusRepository",
    "EmbeddingCacheRepository",
    # Vector Store
    "VectorStore",
    "ChromaStore",
    "QdrantStore",
    "get_vector_store",
]
