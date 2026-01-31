"""
API module for FraudDocs-RAG.

This module provides:
- FastAPI REST endpoints for document ingestion
- Query endpoints for the RAG system
- Health check and system status endpoints
- API documentation and validation
"""

from fraud_docs_rag.api.main import app

__all__ = ["app"]
