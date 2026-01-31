"""
FastAPI Application for FraudDocs-RAG.

This module provides a REST API for the RAG system including:
- Query endpoint for asking questions
- Document ingestion endpoint
- Health check and status endpoints
- CORS support for frontend integration
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fraud_docs_rag.generation.rag_chain import RAGChain
from fraud_docs_rag.ingestion.document_processor import DocumentProcessor
from fraud_docs_rag.retrieval.hybrid_retriever import HybridRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Environment configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
APP_VERSION = "1.0.0"
APP_NAME = "FraudDocs-RAG API"

# ============================================================================
# Pydantic Models for Request/Response Validation
# ============================================================================


class Source(BaseModel):
    """
    Source document information.

    Attributes:
        index: Citation number [1], [2], [3], etc.
        source: File name of the source document
        doc_type: Document category (aml, kyc, fraud, regulation, general)
        score: Relevance score from retrieval
        preview: First 200 characters of the content
        title: Document title if available
    """

    index: int = Field(..., description="Citation index number", ge=1)
    source: str = Field(..., description="Source file name")
    doc_type: str = Field(..., description="Document category")
    score: float = Field(..., description="Relevance score", ge=0.0, le=1.0)
    preview: str = Field(..., description="Content preview (first 200 chars)")
    title: str = Field(default="", description="Document title")

    model_config = {"json_schema_extra": {"example": {
        "index": 1,
        "source": "aml_sar_requirements.pdf",
        "doc_type": "aml",
        "score": 0.924,
        "preview": "Suspicious Activity Reports (SAR) must be filed...",
        "title": "SAR Filing Requirements"
    }}}


class QueryRequest(BaseModel):
    """
    Request model for querying the RAG system.

    Attributes:
        question: The question to ask
        doc_type_filter: Optional filter by document category
        use_rerank: Whether to apply cross-encoder reranking
        top_k: Number of documents to retrieve
        stream: Whether to stream the response (not yet implemented)
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to ask the RAG system"
    )
    doc_type_filter: str | None = Field(
        None,
        description="Filter by document category (aml, kyc, fraud, regulation, general)"
    )
    use_rerank: bool = Field(
        default=True,
        description="Whether to apply cross-encoder reranking"
    )
    top_k: int | None = Field(
        None,
        ge=1,
        le=50,
        description="Number of documents to retrieve (overrides default)"
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response (future feature)"
    )

    @field_validator("doc_type_filter")
    @classmethod
    def validate_doc_type(cls, v: str | None) -> str | None:
        """Validate document type filter."""
        if v is None:
            return v

        valid_types = {"aml", "kyc", "fraud", "regulation", "general"}
        if v.lower() not in valid_types:
            raise ValueError(
                f"Invalid doc_type_filter: {v}. Must be one of: {valid_types}"
            )
        return v.lower()

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        """Validate and clean question."""
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()

    model_config = {"json_schema_extra": {"example": {
        "question": "What are the SAR filing requirements?",
        "doc_type_filter": "aml",
        "use_rerank": True,
        "top_k": 10,
        "stream": False
    }}}


class QueryResponse(BaseModel):
    """
    Response model for query results.

    Attributes:
        answer: The generated answer with citations
        sources: List of source documents used
        query: The original question
        processing_time: Time taken to process the query (seconds)
        environment: Current environment (development/demo/production)
    """

    answer: str = Field(..., description="Generated answer with source citations")
    sources: list[Source] = Field(..., description="List of source documents")
    query: str = Field(..., description="Original question")
    processing_time: float = Field(..., description="Processing time in seconds", ge=0.0)
    environment: str = Field(..., description="Current environment")

    model_config = {"json_schema_extra": {"example": {
        "answer": "Based on the retrieved documents, SAR must be filed within 30 days [1]...",
        "sources": [
            {
                "index": 1,
                "source": "aml_sar_requirements.pdf",
                "doc_type": "aml",
                "score": 0.924,
                "preview": "Suspicious Activity Reports (SAR) must be filed...",
                "title": "SAR Filing Requirements"
            }
        ],
        "query": "What are the SAR filing requirements?",
        "processing_time": 2.34,
        "environment": "demo"
    }}}


class IngestResponse(BaseModel):
    """
    Response model for document ingestion.

    Attributes:
        status: Status of the ingestion operation
        documents_processed: Number of documents successfully processed
        chunks_created: Number of text chunks created
        categories: Breakdown by document category
        processing_time: Time taken to process documents (seconds)
        errors: List of any errors that occurred
    """

    status: str = Field(..., description="Status of the ingestion")
    documents_processed: int = Field(..., ge=0, description="Number of documents processed")
    chunks_created: int = Field(..., ge=0, description="Number of chunks created")
    categories: dict[str, int] = Field(..., description="Chunks by category")
    processing_time: float = Field(..., ge=0.0, description="Processing time in seconds")
    errors: list[str] = Field(default_factory=list, description="Any errors that occurred")

    model_config = {"json_schema_extra": {"example": {
        "status": "success",
        "documents_processed": 3,
        "chunks_created": 15,
        "categories": {"aml": 8, "kyc": 5, "fraud": 2},
        "processing_time": 12.5,
        "errors": []
    }}}


class HealthResponse(BaseModel):
    """
    Response model for health check.

    Attributes:
        status: Health status (healthy/unhealthy)
        version: API version
        environment: Current environment
        retriever_status: Status of the retriever component
        chain_status: Status of the RAG chain
        uptime: Server uptime in seconds
    """

    status: Literal["healthy", "unhealthy", "degraded"] = Field(
        ...,
        description="Overall health status"
    )
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Current environment")
    retriever_status: Literal["loaded", "not_loaded", "error"] = Field(
        ...,
        description="Retriever status"
    )
    chain_status: Literal["ready", "not_ready", "error"] = Field(
        ...,
        description="RAG chain status"
    )
    uptime: float = Field(..., ge=0.0, description="Server uptime in seconds")
    collection_stats: dict[str, Any] | None = Field(
        None,
        description="Collection statistics if available"
    )

    model_config = {"json_schema_extra": {"example": {
        "status": "healthy",
        "version": "1.0.0",
        "environment": "demo",
        "retriever_status": "loaded",
        "chain_status": "ready",
        "uptime": 3600.5,
        "collection_stats": {
            "collection_name": "financial_documents",
            "total_docs": 150
        }
    }}}


class CollectionsResponse(BaseModel):
    """
    Response model for listing collections.

    Attributes:
        collections: List of available collections
        total: Total number of collections
    """

    collections: list[dict[str, Any]] = Field(..., description="List of collections")
    total: int = Field(..., ge=0, description="Total number of collections")

    model_config = {"json_schema_extra": {"example": {
        "collections": [
            {
                "name": "financial_documents",
                "doc_count": 150,
                "categories": {"aml": 50, "kyc": 40, "fraud": 35, "regulation": 25}
            }
        ],
        "total": 1
    }}}


class ErrorResponse(BaseModel):
    """
    Response model for errors.

    Attributes:
        error: Error message
        detail: Detailed error information
        status_code: HTTP status code
        path: Request path that caused the error
    """

    error: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")
    path: str = Field(..., description="Request path")

    model_config = {"json_schema_extra": {"example": {
        "error": "Validation Error",
        "detail": "Question cannot be empty",
        "status_code": 422,
        "path": "/api/v1/query"
    }}}


# ============================================================================
# Global State
# ============================================================================

# Global variables for RAG components
rag_chain: RAGChain | None = None
retriever: HybridRetriever | None = None
document_processor: DocumentProcessor | None = None
app_start_time: float = time.time()


# ============================================================================
# Lifespan Management
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan events.

    Handles startup initialization and shutdown cleanup.
    """
    global rag_chain, retriever, document_processor

    logger.info("=" * 70)
    logger.info(f"Starting {APP_NAME} v{APP_VERSION}")
    logger.info(f"Environment: {ENVIRONMENT}")
    logger.info("=" * 70)

    try:
        # Startup: Initialize RAG components
        logger.info("Initializing RAG components...")

        # Path configurations
        chroma_path = Path(os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db"))

        # Initialize retriever
        logger.info("Initializing retriever...")
        retriever = HybridRetriever(
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", "financial_documents"),
            chroma_path=chroma_path,
            top_k=int(os.getenv("TOP_K_RETRIEVAL", "10")),
            rerank_top_n=int(os.getenv("RERANK_TOP_N", "5")),
        )

        # Try to load existing index
        if retriever.load_index():
            logger.info("✓ Loaded existing vector index")
        else:
            logger.warning(
                "⚠ No existing index found. "
                "Use the /ingest endpoint to add documents."
            )

        # Initialize document processor
        logger.info("Initializing document processor...")
        document_processor = DocumentProcessor(
            embed_model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        )

        # Initialize RAG chain
        logger.info("Initializing RAG chain...")
        rag_chain = RAGChain(
            retriever=retriever,
            environment=ENVIRONMENT,
        )

        logger.info(f"✓ RAG chain initialized (provider: {rag_chain.config['provider']})")
        logger.info("=" * 70)
        logger.info("API startup complete!")
        logger.info("=" * 70)

        # Yield control to the application
        yield

    except Exception as e:
        logger.error(f"Failed to initialize RAG components: {e}", exc_info=True)
        logger.error("API startup failed!")
        raise

    finally:
        # Shutdown: Cleanup
        logger.info("=" * 70)
        logger.info("Shutting down API...")
        logger.info("=" * 70)


# ============================================================================
# FastAPI Application
# ============================================================================

# Create FastAPI app with lifespan
app = FastAPI(
    title=APP_NAME,
    description="RAG system for financial fraud detection documents",
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ============================================================================
# Middleware Configuration
# ============================================================================

# CORS Middleware
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Rate Limiting
# ============================================================================

# TODO: Add rate limiting middleware
# Recommended libraries:
# - slowapi (for async rate limiting)
# - fastapi-limiter (Redis-based)
#
# Example using slowapi:
# from slowapi import Limiter
# from slowapi.util import get_remote_address
#
# limiter = Limiter(key_func=get_remote_address)
# app.state.limiter = limiter
#
# Then add @limiter.limit("100/minute") to endpoint functions


# ============================================================================
# Request Logging Middleware
# ============================================================================


@app.middleware("http")
async def log_requests(request, call_next):
    """
    Log all incoming requests.
    """
    start_time = time.time()

    # Log request
    logger.info(f"→ {request.method} {request.url.path}")

    # Process request
    response = await call_next(request)

    # Calculate processing time
    process_time = time.time() - start_time

    # Log response
    logger.info(
        f"← {request.method} {request.url.path} "
        f"Status: {response.status_code} "
        f"Time: {process_time:.3f}s"
    )

    # Add processing time to response headers
    response.headers["X-Process-Time"] = str(process_time)

    return response


# ============================================================================
# Exception Handlers
# ============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    Handle HTTP exceptions with custom error response.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=None,
            status_code=exc.status_code,
            path=request.url.path,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    Handle general exceptions with custom error response.
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            detail=str(exc) if ENVIRONMENT == "development" else "An unexpected error occurred",
            status_code=500,
            path=request.url.path,
        ).model_dump(),
    )


# ============================================================================
# API Endpoints
# ============================================================================


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the API and its components",
    tags=["System"],
)
async def health_check():
    """
    Health check endpoint.

    Returns the health status of the API and its components.
    """
    uptime = time.time() - app_start_time

    # Determine component statuses
    retriever_status = "not_loaded"
    chain_status = "not_ready"

    if retriever is not None:
        retriever_status = "loaded" if retriever.index else "not_loaded"

    if rag_chain is not None:
        chain_status = "ready"

    # Get collection stats if available
    collection_stats = None
    if retriever is not None:
        try:
            collection_stats = retriever.get_collection_stats()
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")

    # Determine overall health
    if retriever_status == "loaded" and chain_status == "ready":
        health_status = "healthy"
    elif retriever is not None or rag_chain is not None:
        health_status = "degraded"
    else:
        health_status = "unhealthy"

    return HealthResponse(
        status=health_status,
        version=APP_VERSION,
        environment=ENVIRONMENT,
        retriever_status=retriever_status,
        chain_status=chain_status,
        uptime=uptime,
        collection_stats=collection_stats,
    )


@app.get(
    "/collections",
    response_model=CollectionsResponse,
    summary="List document collections",
    description="List all available document collections in the vector store",
    tags=["System"],
)
async def list_collections():
    """
    List available document collections.

    Returns information about all collections in the vector store.
    """
    if retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retriever not initialized",
        )

    try:
        stats = retriever.get_collection_stats()

        collections_list = [{
            "name": stats.get("collection_name", "financial_documents"),
            "doc_count": stats.get("total_docs", 0),
            "status": stats.get("status", "unknown"),
        }]

        return CollectionsResponse(
            collections=collections_list,
            total=len(collections_list),
        )

    except Exception as e:
        logger.error(f"Failed to list collections: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list collections: {str(e)}",
        )


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the knowledge base",
    description="Ask a question and get an answer with source citations",
    tags=["Query"],
)
async def query_knowledge_base(request: QueryRequest):
    """
    Query the RAG knowledge base.

    Takes a question and returns an answer with source citations.
    """
    if rag_chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG chain not initialized",
        )

    start_time = time.time()

    try:
        logger.info(f"Query: {request.question[:100]}...")

        # Process query
        answer, sources_data = rag_chain.query(
            question=request.question,
            doc_type_filter=request.doc_type_filter,
            use_rerank=request.use_rerank,
            top_k=request.top_k,
        )

        # Format sources for response
        sources = []
        for i, source_data in enumerate(sources_data, start=1):
            sources.append(Source(
                index=i,
                source=source_data.get("file_name", "Unknown"),
                doc_type=source_data.get("category", "general"),
                score=source_data.get("score", 0.0),
                preview=source_data.get("text_preview", ""),
                title=source_data.get("title", ""),
            ))

        processing_time = time.time() - start_time

        logger.info(
            f"Query complete: {len(sources)} sources, "
            f"{processing_time:.2f}s"
        )

        return QueryResponse(
            answer=answer,
            sources=sources,
            query=request.question,
            processing_time=processing_time,
            environment=ENVIRONMENT,
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}",
        )


@app.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest documents",
    description="Upload and process documents into the knowledge base",
    tags=["Ingestion"],
)
async def ingest_documents(
    file: UploadFile = File(..., description="Document file to ingest"),
    doc_type: str | None = Form(None, description="Document type override (aml, kyc, fraud, regulation, general)"),
):
    """
    Ingest a document into the knowledge base.

    Uploads a file, processes it into chunks, and adds it to the vector store.
    Supported formats: PDF, DOCX, TXT, HTML.
    """
    if document_processor is None or retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document processor or retriever not initialized",
        )

    start_time = time.time()

    try:
        # Validate file type
        file_extension = file.filename.split(".")[-1].lower() if file.filename else ""
        supported_formats = {"pdf", "docx", "doc", "txt", "html", "htm"}

        if file_extension not in supported_formats:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unsupported file format: .{file_extension}. "
                      f"Supported formats: {', '.join(supported_formats)}",
            )

        logger.info(f"Ingesting file: {file.filename} ({file_extension})")

        # Save uploaded file temporarily
        temp_dir = Path("./data/temp_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)

        temp_file_path = temp_dir / file.filename

        try:
            # Write uploaded file
            with temp_file_path.open("wb") as f:
                content = await file.read()
                f.write(content)

            logger.info(f"Saved temporary file: {temp_file_path}")

            # Process document
            nodes = document_processor.process_document(
                temp_file_path,
                add_context=True,
            )

            if nodes is None or len(nodes) == 0:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Failed to process document: {file.filename}",
                )

            logger.info(f"Created {len(nodes)} chunks from document")

            # Get existing index or create new one
            if retriever.index is None:
                logger.info("No existing index, creating new one...")
                retriever.build_index(nodes)
            else:
                # Add to existing index
                logger.info("Adding to existing index...")
                # Note: In a real implementation, you'd want to use
                # retriever.index.insert(nodes) or similar method
                # For now, we rebuild the index
                logger.warning(
                    "Rebuilding entire index (not optimal for production)"
                )
                retriever.build_index(nodes)

            # Calculate category breakdown
            categories: dict[str, int] = {}
            for node in nodes:
                cat = node.metadata.get("category", "general")
                categories[cat] = categories.get(cat, 0) + 1

            processing_time = time.time() - start_time

            logger.info(
                f"Ingestion complete: {len(nodes)} chunks, "
                f"{processing_time:.2f}s"
            )

            return IngestResponse(
                status="success",
                documents_processed=1,
                chunks_created=len(nodes),
                categories=categories,
                processing_time=processing_time,
                errors=[],
            )

        finally:
            # Cleanup temporary file
            if temp_file_path.exists():
                temp_file_path.unlink()
                logger.debug(f"Cleaned up temporary file: {temp_file_path}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        return IngestResponse(
            status="error",
            documents_processed=0,
            chunks_created=0,
            categories={},
            processing_time=time.time() - start_time,
            errors=[str(e)],
        )


@app.get(
    "/",
    summary="API root",
    description="Welcome message and API information",
    tags=["System"],
)
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "environment": ENVIRONMENT,
        "status": "running",
        "uptime": time.time() - app_start_time,
        "endpoints": {
            "health": "/health",
            "query": "/query",
            "ingest": "/ingest",
            "collections": "/collections",
            "docs": "/docs",
        },
    }


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """
    Run the FastAPI application.
    """
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    reload = ENVIRONMENT == "development"

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        "fraud_docs_rag.api.main:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
