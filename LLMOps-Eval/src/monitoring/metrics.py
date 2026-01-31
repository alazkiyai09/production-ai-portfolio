"""
Prometheus metrics collection and monitoring.

Provides centralized metrics tracking for the LLMOps-Eval system.
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from typing import Optional, Dict, Any
import logging
import time

logger = logging.getLogger(__name__)


# ============================================================================
# Metrics Registry
# ============================================================================

# Custom registry for our metrics
registry = CollectorRegistry()


# ============================================================================
# HTTP Metrics
# ============================================================================

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
    registry=registry,
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    registry=registry,
)

http_requests_in_progress = Gauge(
    "http_requests_in_progress",
    "HTTP requests currently in progress",
    ["method", "endpoint"],
    registry=registry,
)


# ============================================================================
# Evaluation Metrics
# ============================================================================

evaluation_total = Counter(
    "evaluation_total",
    "Total evaluations run",
    ["dataset", "status"],
    registry=registry,
)

evaluation_duration_seconds = Histogram(
    "evaluation_duration_seconds",
    "Evaluation duration in seconds",
    ["dataset"],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600],
    registry=registry,
)

evaluation_cost_usd_total = Gauge(
    "evaluation_cost_usd_total",
    "Total cost of all evaluations in USD",
    registry=registry,
)

evaluation_tests_total = Gauge(
    "evaluation_tests_total",
    "Total number of tests run",
    registry=registry,
)

evaluation_tests_passed_total = Gauge(
    "evaluation_tests_passed_total",
    "Total number of tests passed",
    registry=registry,
)

evaluation_tests_failed_total = Gauge(
    "evaluation_tests_failed_total",
    "Total number of tests failed",
    registry=registry,
)

active_evaluations = Gauge(
    "active_evaluations",
    "Number of currently running evaluations",
    registry=registry,
)


# ============================================================================
# Model Metrics
# ============================================================================

model_requests_total = Counter(
    "model_requests_total",
    "Total model API requests",
    ["provider", "model", "status"],
    registry=registry,
)

model_latency_seconds = Histogram(
    "model_latency_seconds",
    "Model request latency in seconds",
    ["provider", "model"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    registry=registry,
)

model_tokens_total = Counter(
    "model_tokens_total",
    "Total tokens processed",
    ["provider", "model", "token_type"],  # token_type: input, output
    registry=registry,
)

model_cost_usd_total = Counter(
    "model_cost_usd_total",
    "Total cost in USD",
    ["provider", "model"],
    registry=registry,
)

model_errors_total = Counter(
    "model_errors_total",
    "Total model errors",
    ["provider", "model", "error_type"],
    registry=registry,
)


# ============================================================================
# Metric Metrics
# ============================================================================

metric_evaluations_total = Counter(
    "metric_evaluations_total",
    "Total metric evaluations",
    ["metric_name", "status"],
    registry=registry,
)

metric_duration_seconds = Histogram(
    "metric_duration_seconds",
    "Metric evaluation duration",
    ["metric_name"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    registry=registry,
)


# ============================================================================
# Dataset Metrics
# ============================================================================

dataset_loads_total = Counter(
    "dataset_loads_total",
    "Total dataset loads",
    ["dataset_name", "status"],
    registry=registry,
)

dataset_test_cases_total = Gauge(
    "dataset_test_cases_total",
    "Number of test cases in loaded datasets",
    ["dataset_name"],
    registry=registry,
)


# ============================================================================
# System Metrics
# ============================================================================

system_info = Gauge(
    "system_info",
    "System information",
    ["version", "python_version"],
    registry=registry,
)


# ============================================================================
# Metric Trackers
# ============================================================================

class MetricsTracker:
    """Context manager for tracking request metrics."""

    def __init__(
        self,
        metric_type: str,
        labels: Dict[str, str],
        latency_histogram: Optional[Histogram] = None,
        in_progress_gauge: Optional[Gauge] = None,
    ):
        """
        Initialize metrics tracker.

        Args:
            metric_type: Type of metric (http, model, evaluation, etc.)
            labels: Labels for the metrics
            latency_histogram: Histogram to track latency
            in_progress_gauge: Gauge to track in-progress requests
        """
        self.metric_type = metric_type
        self.labels = labels
        self.latency_histogram = latency_histogram
        self.in_progress_gauge = in_progress_gauge
        self.start_time = None

    def __enter__(self):
        """Start tracking."""
        self.start_time = time.time()
        if self.in_progress_gauge:
            self.in_progress_gauge.labels(**self.labels).inc()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop tracking."""
        duration = time.time() - self.start_time

        if self.in_progress_gauge:
            self.in_progress_gauge.labels(**self.labels).dec()

        if self.latency_histogram:
            self.latency_histogram.labels(**self.labels).observe(duration)

        # Track request total
        status = "success" if exc_type is None else "error"

        if self.metric_type == "http":
            http_requests_total.labels(**self.labels, status=status).inc()
        elif self.metric_type == "model":
            model_requests_total.labels(**self.labels, status=status).inc()
        elif self.metric_type == "evaluation":
            evaluation_total.labels(**self.labels, status=status).inc()

        return False


# ============================================================================
# Convenience Functions
# ============================================================================

def track_http_request(method: str, endpoint: str):
    """Track an HTTP request."""
    return MetricsTracker(
        metric_type="http",
        labels={"method": method, "endpoint": endpoint},
        latency_histogram=http_request_duration_seconds,
        in_progress_gauge=http_requests_in_progress,
    )


def track_model_request(provider: str, model: str):
    """Track a model request."""
    return MetricsTracker(
        metric_type="model",
        labels={"provider": provider, "model": model},
        latency_histogram=model_latency_seconds,
    )


def track_evaluation(dataset: str):
    """Track an evaluation."""
    return MetricsTracker(
        metric_type="evaluation",
        labels={"dataset": dataset},
        latency_histogram=evaluation_duration_seconds,
    )


def record_model_tokens(provider: str, model: str, input_tokens: int, output_tokens: int):
    """Record token usage."""
    model_tokens_total.labels(
        provider=provider,
        model=model,
        token_type="input",
    ).inc(input_tokens)

    model_tokens_total.labels(
        provider=provider,
        model=model,
        token_type="output",
    ).inc(output_tokens)


def record_model_cost(provider: str, model: str, cost_usd: float):
    """Record model cost."""
    model_cost_usd_total.labels(
        provider=provider,
        model=model,
    ).inc(cost_usd)


def record_evaluation_results(
    total_tests: int,
    passed_tests: int,
    failed_tests: int,
    cost_usd: float,
):
    """Record evaluation results."""
    evaluation_tests_total.inc(total_tests)
    evaluation_tests_passed_total.inc(passed_tests)
    evaluation_tests_failed_total.inc(failed_tests)
    evaluation_cost_usd_total.inc(cost_usd)


def increment_active_evaluations():
    """Increment active evaluations count."""
    active_evaluations.inc()


def decrement_active_evaluations():
    """Decrement active evaluations count."""
    active_evaluations.dec()


def get_metrics_text() -> bytes:
    """
    Get metrics in Prometheus text format.

    Returns:
        Metrics as bytes in Prometheus exposition format
    """
    return generate_latest(registry)


def get_content_type() -> str:
    """Get Prometheus content type."""
    return CONTENT_TYPE_LATEST


# Export metrics
__all__ = [
    # HTTP
    "http_requests_total",
    "http_request_duration_seconds",
    "http_requests_in_progress",
    # Evaluation
    "evaluation_total",
    "evaluation_duration_seconds",
    "evaluation_cost_usd_total",
    "evaluation_tests_total",
    "evaluation_tests_passed_total",
    "evaluation_tests_failed_total",
    "active_evaluations",
    # Model
    "model_requests_total",
    "model_latency_seconds",
    "model_tokens_total",
    "model_cost_usd_total",
    "model_errors_total",
    # Metric
    "metric_evaluations_total",
    "metric_duration_seconds",
    # Dataset
    "dataset_loads_total",
    "dataset_test_cases_total",
    # System
    "system_info",
    # Trackers
    "MetricsTracker",
    "track_http_request",
    "track_model_request",
    "track_evaluation",
    "record_model_tokens",
    "record_model_cost",
    "record_evaluation_results",
    "increment_active_evaluations",
    "decrement_active_evaluations",
    "get_metrics_text",
    "get_content_type",
]
