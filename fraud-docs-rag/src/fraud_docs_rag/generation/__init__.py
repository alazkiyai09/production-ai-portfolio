"""
Generation module for FraudDocs-RAG.

This module handles:
- LLM integration (Ollama, GLM-4, OpenAI-compatible APIs)
- Response generation with retrieved context
- Streaming response support
- Citation generation and source attribution
"""

from fraud_docs_rag.generation.rag_chain import RAGChain

__all__ = ["RAGChain"]
