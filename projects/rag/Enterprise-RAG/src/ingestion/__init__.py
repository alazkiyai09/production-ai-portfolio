# ============================================================
# Enterprise-RAG: Document Ingestion Module
# ============================================================
"""
Document ingestion and processing pipeline.

This module handles multi-format document processing, intelligent chunking,
and metadata extraction for the RAG system.
"""

from src.ingestion.document_processor import (
    Document,
    DocumentProcessor,
    ProcessingResult,
)

__all__ = [
    "Document",
    "DocumentProcessor",
    "ProcessingResult",
]
