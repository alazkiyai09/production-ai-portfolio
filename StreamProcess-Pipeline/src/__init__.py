"""
StreamProcess-Pipeline - High-throughput data processing pipeline for LLM applications.

A scalable pipeline for processing AdTech data and generating embeddings in real-time.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from src.ingestion.consumer import MessageConsumer
from src.ingestion.producer import MessageProducer
from src.processing.worker import process_batch
from src.embedding.generator import EmbeddingGenerator
from src.storage.vector_store import VectorStore
from src.storage.database import DatabaseManager

__all__ = [
    "MessageConsumer",
    "MessageProducer",
    "process_batch",
    "EmbeddingGenerator",
    "VectorStore",
    "DatabaseManager",
    "__version__",
]
