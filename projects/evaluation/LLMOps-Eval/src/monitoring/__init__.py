"""
Prometheus metrics collection and monitoring.

Provides centralized metrics tracking for the LLMOps-Eval system.
"""

from src.monitoring.metrics import *

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
