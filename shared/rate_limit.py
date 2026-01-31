# ============================================================
# Rate Limiting for All Projects
# ============================================================
"""
Rate limiting utilities using slowapi.

This module provides rate limiting functionality for FastAPI applications
to prevent DoS attacks and abuse.

Example:
    >>> from shared.rate_limit import limiter, get_rate_limit
    >>> from fastapi import FastAPI, Request
    >>>
    >>> app = FastAPI()
    >>> app.state.limiter = limiter
    >>>
    >>> @app.get("/api/endpoint")
    >>> @limiter.limit("10/minute")
    >>> async def endpoint(request: Request):
    ...     return {"status": "ok"}
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)


# ============================================================
# Global Rate Limiter
# ============================================================

def get_remote_address_or_user_id(request: Request) -> str:
    """
    Get client identifier for rate limiting.

    Priority:
    1. X-User-ID header (for authenticated users)
    2. Remote address (IP-based limiting)
    3. API key header (for API authentication)

    Args:
        request: FastAPI Request object

    Returns:
        Client identifier string
    """
    # Check for user ID header (for authenticated requests)
    user_id = request.headers.get("X-User-ID")
    if user_id:
        return f"user:{user_id}"

    # Check for API key header (for API authentication)
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api_key:{api_key[:10]}"  # Use partial key for privacy

    # Fall back to IP address
    return f"ip:{get_remote_address(request)}"


# Create global limiter instance
limiter = Limiter(key_func=get_remote_address_or_user_id)


# ============================================================
# Exception Handler
# ============================================================

async def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """
    Handle rate limit exceeded errors.

    Args:
        request: FastAPI Request object
        exc: Rate limit exceeded exception

    Returns:
        JSON response with 429 status code
    """
    logger.warning(
        f"Rate limit exceeded for {get_remote_address_or_user_id(request)}: {exc.detail}"
    )

    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": str(exc.detail),
            "retry_after": getattr(exc, "retry_after", 60),
        },
    )


# ============================================================
# Pre-configured Rate Limits
# ============================================================

RATE_LIMITS = {
    "default": "60/minute",
    "strict": "10/minute",
    "moderate": "30/minute",
    "lenient": "120/minute",
    "upload": "5/minute",
    "expensive": "5/hour",
}


def get_rate_limit(level: str = "default") -> str:
    """
    Get pre-configured rate limit string.

    Args:
        level: Rate limit level (default, strict, moderate, lenient, upload, expensive)

    Returns:
        Rate limit string for slowapi

    Example:
        >>> get_rate_limit("strict")
        '10/minute'
    """
    return RATE_LIMITS.get(level, RATE_LIMITS["default"])


# ============================================================
# Decorator Helpers
# ============================================================

def rate_limit(level: str = "default"):
    """
    Rate limiting decorator helper.

    Args:
        level: Rate limit level (default, strict, moderate, lenient, upload, expensive)

    Example:
        >>> @app.post("/api/upload")
        >>> @rate_limit("upload")
        >>> async def upload_file(request: Request, file: UploadFile):
        ...     ...
    """
    return limiter.limit(get_rate_limit(level))


# ============================================================
# Export
# ============================================================

__all__ = [
    "limiter",
    "RateLimitExceeded",
    "get_remote_address_or_user_id",
    "get_rate_limit",
    "rate_limit",
    "rate_limit_exception_handler",
]

# Re-export for convenience
RateLimitExceeded = RateLimitExceeded
