# ============================================================
# Enterprise-RAG: FastAPI Application
# ============================================================
"""
Production-grade FastAPI application for the RAG system.

This module provides:
- RESTful API with OpenAPI documentation
- Async endpoints for better performance
- Request validation with Pydantic
- Comprehensive error handling
- Health checks and monitoring

Example:
    >>> uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import settings
from src.exceptions import format_error_for_api
from src.logging_config import get_logger, set_request_context
from src.api.routes import query_router, documents_router, evaluation_router
from src.api import multimodal_endpoints

# Initialize logger
logger = get_logger(__name__)

# ============================================================
# Global Components (initialized on startup)
# ============================================================

rag_chain: Optional["RAGChain"] = None
document_processor: Optional["DocumentProcessor"] = None
vector_store: Optional["VectorStoreBase"] = None
rag_evaluator: Optional["RAGEvaluator"] = None


# ============================================================
# Lifespan Context Manager
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan.

    Initializes RAG components on startup and cleanup on shutdown.

    Yields:
        None
    """
    global rag_chain, document_processor, vector_store, rag_evaluator

    # Startup
    logger.info("=" * 60)
    logger.info("Enterprise-RAG API Starting")
    logger.info("=" * 60)

    try:
        # Import here to avoid circular imports
        from src.generation import RAGChain, create_rag_chain, LLMProvider
        from src.ingestion import DocumentProcessor, create_processor_from_settings
        from src.retrieval import create_vector_store, create_embedding_service, create_hybrid_retriever, CrossEncoderReranker
        from src.evaluation import RAGEvaluator, create_evaluator

        # Initialize embedding service
        logger.info("Initializing embedding service...")
        embedding_service = create_embedding_service()

        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = create_vector_store_from_settings()

        # Initialize BM25 retriever
        logger.info("Initializing BM25 retriever...")
        from src.retrieval.sparse_retriever import create_bm25_retriever
        bm25_retriever = create_bm25_retriever()

        # Initialize hybrid retriever
        logger.info("Initializing hybrid retriever...")
        hybrid_retriever = create_hybrid_retriever(
            vector_store=vector_store,
            embedding_service=embedding_service,
            bm25_retriever=bm25_retriever,
        )

        # Initialize reranker
        logger.info("Initializing cross-encoder reranker...")
        reranker = CrossEncoderReranker()

        # Initialize document processor
        logger.info("Initializing document processor...")
        document_processor = create_processor_from_settings()

        # Initialize RAG chain
        logger.info("Initializing RAG chain...")
        rag_chain = create_rag_chain(
            retriever=hybrid_retriever,
            reranker=reranker,
            llm_provider=settings.LLM_MODEL.split("-")[0] if "-" in settings.LLM_MODEL else "openai",
        )

        # Initialize evaluator
        if settings.ENABLE_EVALUATION:
            logger.info("Initializing RAG evaluator...")
            rag_evaluator = create_evaluator(rag_chain)
        else:
            logger.info("RAG evaluation disabled")
            rag_evaluator = None

        # Store components in app state for access in routes
        app.state.rag_chain = rag_chain
        app.state.document_processor = document_processor
        app.state.vector_store = vector_store
        app.state.rag_evaluator = rag_evaluator

        logger.info("=" * 60)
        logger.info("Enterprise-RAG API Ready")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed to initialize RAG components: {str(e)}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Enterprise-RAG API Shutting down...")

    # Cleanup
    if rag_chain:
        logger.info("Cleaning up RAG chain...")

    logger.info("Shutdown complete")


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Enterprise-RAG API",
    description="""
## Production-Grade RAG System with Hybrid Retrieval & Evaluation

A complete Retrieval-Augmented Generation system featuring:
- **Hybrid Retrieval**: Dense vector search + sparse BM25
- **Cross-Encoder Reranking**: Improved accuracy with MS-MARCO
- **Multi-Format Ingestion**: PDF, DOCX, MD, TXT support
- **RAGAS Evaluation**: Comprehensive metrics (faithfulness, relevancy, etc.)
- **Multiple LLM Providers**: OpenAI, Anthropic, Ollama, GLM

### Features
- Semantic search with keyword fallback
- Automatic citation extraction
- Streaming responses
- Batch document ingestion
- Evaluation and benchmarking

### Quick Start
1. **Query**: POST /query with your question
2. **Ingest**: POST /ingest to upload documents
3. **Evaluate**: POST /evaluate to run RAGAS metrics

### Authentication
Set `OPENAI_API_KEY` and optionally `ANTHROPIC_API_KEY` in your environment.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ============================================================
# CORS Middleware
# ============================================================

origins = [origin.strip() for origin in settings.CORS_ORIGINS.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ============================================================
# Exception Handlers
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    """Handle all uncaught exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)

    error_response = format_error_for_api(exc)

    return JSONResponse(
        status_code=error_response.get("http_status", 500),
        content=error_response,
    )


# ============================================================
# Request ID Middleware
# ============================================================

@app.middleware("http")
async def request_id_middleware(request, call_next):
    """Add request ID to context for logging."""
    import uuid

    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    set_request_context(request_id)

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id

    return response


# ============================================================
# Routes
# ============================================================

# Include routers
app.include_router(query_router, prefix="/api/v1", tags=["Query"])
app.include_router(documents_router, prefix="/api/v1/documents", tags=["Documents"])
app.include_router(evaluation_router, prefix="/api/v1/evaluation", tags=["Evaluation"])
app.include_router(multimodal_endpoints.router, tags=["Multi-Modal"])


# ============================================================
# Root Endpoints
# ============================================================

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint.

    Returns API information and available endpoints.
    """
    return {
        "name": "Enterprise-RAG API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "query": "/api/v1/query",
            "ingest": "/api/v1/documents/ingest",
            "evaluate": "/api/v1/evaluation",
            "health": "/health",
        },
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Checks the health of all RAG components.

    Returns:
        Health status with component-level details
    """
    components = {
        "api": "healthy",
        "rag_chain": "unknown",
        "vector_store": "unknown",
        "embedding_service": "unknown",
        "reranker": "unknown",
        "evaluator": "unknown",
    }

    # Check RAG chain
    try:
        if app.state.rag_chain:
            components["rag_chain"] = "healthy"
    except Exception as e:
        components["rag_chain"] = f"unhealthy: {str(e)[:50]}"

    # Check vector store
    try:
        if app.state.vector_store:
            stats = app.state.vector_store.get_stats()
            components["vector_store"] = f"healthy ({stats.total_chunks} chunks)"
    except Exception as e:
        components["vector_store"] = f"unhealthy: {str(e)[:50]}"

    # Check embedding service
    try:
        if app.state.rag_chain and app.state.rag_chain.retriever:
            components["embedding_service"] = "healthy"
    except Exception as e:
        components["embedding_service"] = f"unhealthy: {str(e)[:50]}"

    # Check reranker
    try:
        if app.state.rag_chain and app.state.rag_chain.reranker:
            if app.state.rag_chain.reranker._is_loaded:
                components["reranker"] = "healthy"
            else:
                components["reranker"] = "healthy (not loaded)"
    except Exception as e:
        components["reranker"] = f"unhealthy: {str(e)[:50]}"

    # Check evaluator
    try:
        if app.state.rag_evaluator:
            components["evaluator"] = "healthy" if app.state.rag_evaluator.enable_evaluation else "disabled"
        else:
            components["evaluator"] = "disabled"
    except Exception as e:
        components["evaluator"] = f"unhealthy: {str(e)[:50]}"

    # Determine overall status
    all_healthy = all(
        "unhealthy" not in str(v) and "unknown" not in str(v)
        for v in components.values()
    )

    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "components": components,
    }


@app.get("/stats", tags=["Health"])
async def get_stats():
    """
    Get statistics about the RAG system.

    Returns component-level statistics.
    """
    stats = {
        "vector_store": {},
        "rag_chain": {},
        "retriever": {},
    }

    # Vector store stats
    try:
        if app.state.vector_store:
            vs_stats = app.state.vector_store.get_stats()
            stats["vector_store"] = vs_stats.to_dict()
    except Exception as e:
        stats["vector_store"] = {"error": str(e)}

    # RAG chain stats
    try:
        if app.state.rag_chain:
            chain_stats = app.state.rag_chain.get_stats()
            stats["rag_chain"] = chain_stats
    except Exception as e:
        stats["rag_chain"] = {"error": str(e)}

    # Retriever stats
    try:
        if app.state.rag_chain and app.state.rag_chain.retriever:
            retriever_stats = app.state.rag_chain.retriever.get_stats()
            stats["retriever"] = retriever_stats.to_dict()
    except Exception as e:
        stats["retriever"] = {"error": str(e)}

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "stats": stats,
    }


# ============================================================
# Main Entry Point
# ============================================================

def main():
    """
    Run the FastAPI application.

    Usage:
        python -m src.api.main
    """
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
