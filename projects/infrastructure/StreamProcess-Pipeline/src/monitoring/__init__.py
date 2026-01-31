"""
Monitoring and metrics collection layer.
"""

from src.monitoring.metrics import (
    MetricsCollector,
    get_metrics,
    start_metrics_server,
    set_app_info,
    set_build_info,
    generate_metrics,
    ComponentStatus,
    ComponentHealth,
    HealthChecker,
    AlertThresholds,
)

# Future imports
# from src.monitoring.tracing import init_tracing
# from src.monitoring.health import HealthChecker

__all__ = [
    # Core
    "MetricsCollector",
    "get_metrics",
    "start_metrics_server",
    "generate_metrics",
    # App Info
    "set_app_info",
    "set_build_info",
    # Health
    "ComponentStatus",
    "ComponentHealth",
    "HealthChecker",
    "AlertThresholds",
    # Future
    # "init_tracing",
]
