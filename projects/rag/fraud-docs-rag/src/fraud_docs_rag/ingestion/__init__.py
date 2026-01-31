"""
Document ingestion module for FraudDocs-RAG.

This module handles:
- Loading documents from various file formats (PDF, DOCX, PPTX, TXT, MD)
- Semantic chunking for optimal context preservation
- Vector store operations with ChromaDB
- Embedding generation using HuggingFace models
"""

from fraud_docs_rag.ingestion.document_processor import DocumentProcessor

__all__ = ["DocumentProcessor"]
