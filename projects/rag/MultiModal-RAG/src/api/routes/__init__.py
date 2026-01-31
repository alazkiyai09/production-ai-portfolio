# ============================================================
# Enterprise-RAG: API Routes
# ============================================================
"""API route handlers."""

from src.api.routes.query import router as query_router
from src.api.routes.documents import router as documents_router
from src.api.routes.evaluation import router as evaluation_router

__all__ = ["query_router", "documents_router", "evaluation_router"]
