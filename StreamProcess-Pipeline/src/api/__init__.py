"""
FastAPI application and models.
"""

from src.api.main import (
    app,
    main,
    get_settings,
)
from src.api.metrics_endpoint import (
    router as metrics_router,
    setup_metrics_middleware,
)

# Future imports - these will be created
# from src.api.routes import (
#     ingestion,
#     processing,
#     health,
#     metrics,
#     admin,
# )

__all__ = [
    # App
    "app",
    "main",
    # Utilities
    "get_settings",
    # Metrics
    "metrics_router",
    "setup_metrics_middleware",
]
