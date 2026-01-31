"""
FraudDocs-RAG: Production-grade RAG system for financial fraud detection documents.

This package provides a complete Retrieval-Augmented Generation system specifically
designed for querying financial regulations and fraud detection documents.

Main modules:
- ingestion: Document loading, chunking, and vector store management
- retrieval: Query processing and reranking
- generation: LLM integration and response generation
- api: FastAPI REST endpoints
- utils: Shared utilities for logging, validation, etc.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from fraud_docs_rag.config import settings

__all__ = ["settings", "__version__"]
