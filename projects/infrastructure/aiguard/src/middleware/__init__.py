"""
FastAPI middleware for easy AIGuard integration.
"""

from src.middleware.aiguard_middleware import AIGuardMiddleware, AIGuardConfig

__all__ = ["AIGuardMiddleware", "AIGuardConfig"]
