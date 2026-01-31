# ============================================================
# Enterprise-RAG: Query Routes
# ============================================================
"""
Query endpoints for the RAG API.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.config import settings
from src.exceptions import format_error_for_api
from src.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


# ============================================================
# Request/Response Models
# ============================================================

class Citation(BaseModel):
    """Citation model."""

    source: str
    chunk_id: str
    content_preview: str
    relevance_score: float
    page_number: Optional[int] = None


class QueryRequest(BaseModel):
    """Request model for RAG query."""

    question: str = Field(..., min_length=1, max_length=1000, description="Question to ask the RAG system")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    use_reranking: bool = Field(default=True, description="Whether to use cross-encoder reranking")
    filters: Optional[dict] = Field(default=None, description="Optional metadata filters")
    include_history: bool = Field(default=False, description="Include conversation history")


class QueryResponse(BaseModel):
    """Response model for RAG query."""

    answer: str
    citations: list[Citation]
    processing_time: float
    retrieval_time: float = 0.0
    rerank_time: float = 0.0
    generation_time: float = 0.0
    model_used: str
    provider_used: str
    token_usage: Optional[dict] = None


class StreamQueryRequest(BaseModel):
    """Request model for streaming query."""

    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranking: bool = Field(default=True)


# ============================================================
# Routes
# ============================================================

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system with a question.

    This endpoint:
    1. Retrieves relevant documents using hybrid search (dense + sparse)
    2. Reranks results using cross-encoder (optional)
    3. Generates answer using LLM
    4. Extracts citations from retrieved contexts

    Args:
        request: Query request with question and parameters

    Returns:
        Generated answer with citations and metadata

    Example:
        POST /api/v1/query
        {
            "question": "What is the refund policy?",
            "top_k": 5,
            "use_reranking": true
        }
    """
    from fastapi import Request

    # Get RAG chain from app state
    request_obj = Request.scope()["app"]
    rag_chain = getattr(request_obj.state, "rag_chain", None)

    if rag_chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized",
        )

    try:
        logger.info(f"Query received: {request.question[:100]}")

        # Execute query
        response = rag_chain.query(
            question=request.question,
            top_k_retrieve=request.top_k * 2,  # Retrieve more for reranking
            top_k_rerank=request.top_k,
            use_reranking=request.use_reranking,
            filters=request.filters,
            include_history=request.include_history,
        )

        # Convert citations
        citations = [
            Citation(
                source=cit.source,
                chunk_id=cit.chunk_id,
                content_preview=cit.content_preview,
                relevance_score=cit.relevance_score,
                page_number=cit.page_number,
            )
            for cit in response.citations
        ]

        return QueryResponse(
            answer=response.answer,
            citations=citations,
            processing_time=response.processing_time,
            retrieval_time=response.retrieval_time,
            rerank_time=response.rerank_time,
            generation_time=response.generation_time,
            model_used=response.model_used,
            provider_used=response.provider_used.value,
            token_usage=response.token_usage,
        )

    except Exception as e:
        logger.error(f"Query failed: {str(e)}", exc_info=True)
        error_response = format_error_for_api(e)
        raise HTTPException(
            status_code=error_response.get("http_status", 500),
            detail=error_response,
        )


@router.post("/query/stream")
async def query_stream(request: StreamQueryRequest):
    """
    Stream a query response in real-time.

    This endpoint streams the generated answer token by token.

    Args:
        request: Query request

    Returns:
        Streaming response with answer chunks

    Example:
        POST /api/v1/query/stream
        {
            "question": "Explain the privacy policy"
        }
    """
    from fastapi import Request
    from fastapi.responses import StreamingResponse

    # Get RAG chain
    request_obj = Request.scope()["app"]
    rag_chain = getattr(request_obj.state, "rag_chain", None)

    if rag_chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized",
        )

    async def generate():
        """Generate streaming response."""
        try:
            async for chunk in rag_chain.query_stream(
                question=request.question,
                top_k_retrieve=request.top_k * 2,
                top_k_rerank=request.top_k,
                use_reranking=request.use_reranking,
            ):
                yield chunk

        except Exception as e:
            logger.error(f"Streaming query failed: {str(e)}", exc_info=True)
            yield f"\n\n[Error: {str(e)}]"

    return StreamingResponse(generate(), media_type="text/plain")


@router.post("/conversation/clear")
async def clear_conversation():
    """
    Clear the conversation history.

    Example:
        POST /api/v1/conversation/clear
    """
    from fastapi import Request

    request_obj = Request.scope()["app"]
    rag_chain = getattr(request_obj.state, "rag_chain", None)

    if rag_chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized",
        )

    rag_chain.clear_history()

    return {
        "status": "success",
        "message": "Conversation history cleared",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/conversation")
async def get_conversation():
    """
    Get the conversation history.

    Example:
        GET /api/v1/conversation
    """
    from fastapi import Request

    request_obj = Request.scope()["app"]
    rag_chain = getattr(request_obj.state, "rag_chain", None)

    if rag_chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized",
        )

    history = rag_chain.get_history()

    return {
        "history": [
            {"role": msg.role, "content": msg.content, "timestamp": msg.timestamp}
            for msg in history
        ],
        "length": len(history),
    }
