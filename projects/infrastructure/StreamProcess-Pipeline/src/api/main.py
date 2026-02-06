# ============================================================
# StreamProcess-Pipeline: FastAPI Application
# ============================================================
"""
FastAPI application for the Stream Processing Pipeline.

This module provides:
- RESTful API for pipeline management
- Metrics and monitoring endpoints
- Health checks and status
"""

import logging
import os
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, status, Request, Depends, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Shared modules
from shared.security import SensitiveDataFilter, install_security_filter
from shared.rate_limit import limiter, rate_limit_exception_handler, RateLimitExceeded
from shared.auth import (
    get_current_user,
    require_role,
    require_admin,
    create_access_token,
    create_refresh_token,
    verify_token,
    User,
    UserCreate,
    Token,
    TokenData,
    Role,
    InMemoryUserStore,
)
from shared.errors import (
    register_error_handlers,
    AuthenticationError,
    AuthorizationError,
    ValidationError,
    DatabaseError,
    ExternalAPIError,
)
from shared.secrets import get_settings

from src.api.metrics_endpoint import router as metrics_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="StreamProcess-Pipeline API",
    description="""
## Stream Processing Pipeline for Real-Time Data

A high-performance stream processing pipeline featuring:
- Real-time data ingestion and processing
- Vector embeddings generation
- Async task queue with Celery
- Prometheus metrics and monitoring
- Health checks and status endpoints
""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ============================================================
# CORS Middleware
# ============================================================

origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8501").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Install security filter for logs
install_security_filter()

# Register error handlers
register_error_handlers(app)

# Register rate limit exception handler
app.add_exception_handler(RateLimitExceeded, rate_limit_exception_handler)

# Get settings
settings = get_settings()

# ============================================================
# Authentication
# ============================================================

# In-memory user store for demo (replace with database in production)
user_store = InMemoryUserStore()


@app.post("/auth/register", tags=["Authentication"])
@limiter.limit("10/hour")
async def register(
    user_data: UserCreate,
    request: Request,
):
    """Register a new user."""
    try:
        user = await user_store.create_user(user_data)
        return {
            "message": "User registered successfully",
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@app.post("/auth/login", tags=["Authentication"])
@limiter.limit("20/minute")
async def login(
    username: str = Form(...),
    password: str = Form(...),
    request: Request = None,
):
    """Login and receive access token."""
    from shared.auth import authenticate_user, login_user

    user = await authenticate_user(username, password, user_store)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_data = await login_user(username, password, user_store)
    return token_data


@app.post("/auth/refresh", tags=["Authentication"])
@limiter.limit("30/minute")
async def refresh(
    refresh_token: str = Body(..., embed=True),
    request: Request = None,
):
    """Refresh access token."""
    from shared.auth import refresh_user_token

    token_data = await refresh_user_token(refresh_token, user_store)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token_data


@app.get("/auth/me", tags=["Authentication"])
@limiter.limit("60/minute")
async def get_current_user_info(current_user: TokenData = Depends(get_current_user)):
    """Get current user information."""
    user = await user_store.get_user_by_id(current_user.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role": user.role.value,
        "is_active": user.is_active,
    }


# ============================================================
# Include Routers
# ============================================================

app.include_router(metrics_router)


# ============================================================
# Root Endpoints
# ============================================================

@app.get("/", tags=["Root"])
@limiter.limit("60/minute")
async def root(request: Request):
    """
    Root endpoint.

    Returns API information and available endpoints.
    """
    return {
        "name": "StreamProcess-Pipeline API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/metrics/health",
        "endpoints": {
            "metrics": "/metrics",
            "health": "/metrics/health",
            "liveness": "/metrics/health/live",
            "readiness": "/metrics/health/ready",
        },
    }


@app.get("/health", tags=["Health"])
@limiter.limit("60/minute")
async def health_check(request: Request):
    """
    Health check endpoint.

    Returns the health status of the API and its components.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "api": "healthy",
            "metrics": "healthy",
        },
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
