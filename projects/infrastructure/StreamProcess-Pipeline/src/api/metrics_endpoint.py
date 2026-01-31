"""
FastAPI metrics endpoint for StreamProcess-Pipeline.

Exposes Prometheus metrics and health status via HTTP.
"""

import os
from typing import Dict, Any

from fastapi import APIRouter, Response, status
from fastapi.responses import PlainTextResponse

from src.monitoring.metrics import (
    get_metrics,
    generate_metrics,
    ComponentStatus,
)


# ============================================================================
# Router
# ============================================================================

router = APIRouter(prefix="/metrics", tags=["monitoring"])


# ============================================================================
# Metrics Endpoint
# ============================================================================

@router.get(
    "",
    response_class=PlainTextResponse,
    status_code=status.HTTP_200_OK,
    summary="Prometheus metrics",
    description="Returns all metrics in Prometheus text format",
)
async def get_prometheus_metrics() -> PlainTextResponse:
    """
    Get Prometheus metrics.

    Returns all registered metrics in Prometheus text exposition format.
    Can be scraped by Prometheus server.
    """
    metrics_text = generate_metrics()
    return PlainTextResponse(
        content=metrics_text,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# ============================================================================
# Health Endpoint
# ============================================================================

@router.get(
    "/health",
    summary="Health check",
    description="Returns overall system health status",
)
async def get_health_status() -> Dict[str, Any]:
    """
    Get system health status.

    Returns health status for all components including:
    - Overall status (healthy, degraded, unhealthy, unknown)
    - Component-level status
    - Uptime
    - Timestamp
    """
    collector = get_metrics()
    return collector.get_health_status()


@router.get(
    "/health/live",
    summary="Liveness probe",
    description="Kubernetes liveness probe endpoint",
)
async def liveness_probe() -> Dict[str, str]:
    """
    Liveness probe for Kubernetes.

    Returns 200 if the service is alive.
    """
    return {"status": "alive"}


@router.get(
    "/health/ready",
    summary="Readiness probe",
    description="Kubernetes readiness probe endpoint",
)
async def readiness_probe() -> Dict[str, Any]:
    """
    Readiness probe for Kubernetes.

    Returns 200 if the service is ready to accept traffic.
    Returns 503 if the service is not ready.
    """
    collector = get_metrics()
    health = collector.get_health_status()

    # Check if all critical components are healthy
    overall_status = health.get("status", "unknown")

    if overall_status == "healthy":
        return {"status": "ready"}
    else:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not_ready", "health": health}
        )


# ============================================================================
# Component Status Endpoints
# ============================================================================

@router.get(
    "/components",
    summary="Component status",
    description="Returns status of all components",
)
async def get_component_status() -> Dict[str, Any]:
    """
    Get status of all components.

    Returns detailed status for each component:
    - Name
    - Status (healthy, degraded, unhealthy, unknown)
    - Message
    - Metrics
    - Last check timestamp
    """
    collector = get_metrics()
    health = collector.get_health_status()
    return health.get("components", {})


@router.get(
    "/components/{component_name}",
    summary="Component status",
    description="Returns status of a specific component",
)
async def get_single_component_status(component_name: str) -> Dict[str, Any]:
    """
    Get status of a specific component.

    Args:
        component_name: Name of the component

    Returns:
        Component status details
    """
    collector = get_metrics()
    health = collector.get_health_status()
    components = health.get("components", {})

    if component_name not in components:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Component '{component_name}' not found"
        )

    return components[component_name]


# ============================================================================
# Metrics Summary Endpoints
# ============================================================================

@router.get(
    "/summary",
    summary="Metrics summary",
    description="Returns a summary of key metrics",
)
async def get_metrics_summary() -> Dict[str, Any]:
    """
    Get metrics summary.

    Returns aggregated metrics for monitoring dashboards:
    - Total records ingested
    - Total records processed
    - Total embeddings generated
    - Current queue depths
    - Average latencies
    - Error rates
    """
    from prometheus_client import REGISTRY

    # This would require querying the metrics
    # For now, return a placeholder
    collector = get_metrics()
    health = collector.get_health_status()

    return {
        "status": health.get("status"),
        "uptime_seconds": health.get("uptime_seconds"),
        "message": "Use /metrics for full Prometheus metrics",
    }


# ============================================================================
# Alert Status Endpoint
# ============================================================================

@router.get(
    "/alerts",
    summary="Alert status",
    description="Returns current alert status based on thresholds",
)
async def get_alert_status() -> Dict[str, Any]:
    """
    Get alert status.

    Returns alerts that would fire based on current metrics and thresholds.
    """
    from src.monitoring.metrics import AlertThresholds

    collector = get_metrics()
    alerts = []
    warnings = []

    # Check ingestion metrics
    # Note: This would require actually querying the metric values
    # For now, this is a placeholder showing the structure

    return {
        "alerts": alerts,
        "warnings": warnings,
        "thresholds": {
            "ingestion_rate_low": AlertThresholds.INGESTION_RATE_LOW_THRESHOLD,
            "ingestion_latency_high": AlertThresholds.INGESTION_LATENCY_HIGH_THRESHOLD,
            "processing_latency_high": AlertThresholds.PROCESSING_LATENCY_HIGH_THRESHOLD,
            "queue_depth_high": AlertThresholds.QUEUE_DEPTH_HIGH_THRESHOLD,
            "embedding_latency_high": AlertThresholds.EMBEDDING_LATENCY_HIGH_THRESHOLD,
            "memory_usage_high": AlertThresholds.MEMORY_USAGE_HIGH_THRESHOLD,
            "cpu_usage_high": AlertThresholds.CPU_USAGE_HIGH_THRESHOLD,
        }
    }


# ============================================================================
# Debug Endpoints
# ============================================================================

@router.get(
    "/debug",
    summary="Debug metrics",
    description="Returns detailed debug information (not for production)",
    include_in_schema=False,
)
async def get_debug_metrics() -> Dict[str, Any]:
    """
    Get debug metrics.

    Returns detailed metrics for debugging.
    WARNING: This endpoint may expose sensitive information.
    """
    import sys

    return {
        "python_version": sys.version,
        "environment": os.getenv("ENVIRONMENT", "unknown"),
        "message": "Detailed metrics available at /metrics endpoint",
    }


# ============================================================================
# Integration with Main App
# ============================================================================

def setup_metrics_middleware(app):
    """
    Setup metrics middleware for FastAPI app.

    Tracks HTTP requests, latency, and errors.

    Args:
        app: FastAPI application instance
    """
    from fastapi import Request
    import time

    from src.monitoring.metrics import (
        http_requests_total,
        http_request_duration_seconds,
        http_requests_in_progress,
    )

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        """Middleware to track HTTP metrics."""
        # Get method and path
        method = request.method
        path = request.url.path

        # Skip metrics endpoint to avoid recursion
        if path.startswith("/metrics"):
            return await call_next(request)

        # Track in-progress requests
        http_requests_in_progress.labels(method=method, endpoint=path).inc()

        # Track start time
        start_time = time.time()

        try:
            # Process request
            response = await call_next(request)

            # Record metrics
            status = str(response.status_code)
            http_requests_total.labels(
                method=method,
                endpoint=path,
                status=status,
            ).inc()

            return response

        except Exception as e:
            # Record error
            http_requests_total.labels(
                method=method,
                endpoint=path,
                status="500",
            ).inc()

            raise

        finally:
            # Record latency
            duration = time.time() - start_time
            http_request_duration_seconds.labels(
                method=method,
                endpoint=path,
            ).observe(duration)

            # Decrement in-progress
            http_requests_in_progress.labels(method=method, endpoint=path).dec()

    return app


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "router",
    "get_prometheus_metrics",
    "get_health_status",
    "setup_metrics_middleware",
]
