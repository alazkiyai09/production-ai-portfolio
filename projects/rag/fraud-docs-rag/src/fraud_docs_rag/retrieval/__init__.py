"""
Retrieval module for FraudDocs-RAG.

This module handles:
- Query processing and vector similarity search
- Cross-encoder reranking for improved accuracy
- Hybrid search capabilities (semantic + keyword)
- Retrieval configuration and optimization
"""

from fraud_docs_rag.retrieval.hybrid_retriever import HybridRetriever

__all__ = ["HybridRetriever"]
