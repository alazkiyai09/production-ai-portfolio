# ============================================================
# Enterprise-RAG: Production-Grade RAG System
# Main Package Initialization
# ============================================================
"""
Enterprise-RAG: A production-grade Retrieval-Augmented Generation system
with hybrid retrieval, cross-encoder reranking, and comprehensive evaluation.

This package provides:
- Hybrid retrieval combining dense vector search and sparse BM25
- Cross-encoder reranking for improved retrieval accuracy
- Multi-format document ingestion (PDF, DOCX, MD, TXT)
- RAGAS evaluation framework integration
- Production-ready FastAPI backend
- Streamlit demo interface

Example:
    >>> from src.core.rag_engine import RAGEngine
    >>> from src.config import settings
    >>> engine = RAGEngine(config=settings)
    >>> response = engine.query("What is the company's refund policy?")
    >>> print(response.answer)

Author: AI Engineer
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "AI Engineer"
__license__ = "MIT"

# Import key classes for convenient access
from src.core.rag_engine import RAGEngine
from src.core.document_processor import DocumentProcessor
from src.config import settings

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "RAGEngine",
    "DocumentProcessor",
    "settings",
]

# Package metadata
PYTHON_MIN_VERSION = (3, 11)
PACKAGE_NAME = "enterprise-rag"
DESCRIPTION = "Production-Grade RAG System with Hybrid Retrieval & Evaluation"

# Version info tuple for programmatic access
VERSION_INFO = tuple(int(x) for x in __version__.split(".") if x.isdigit())
