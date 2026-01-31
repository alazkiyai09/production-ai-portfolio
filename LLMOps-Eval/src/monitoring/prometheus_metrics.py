"""
Prometheus metrics collector for LLMOps-Eval.

Provides centralized metrics tracking for evaluations, LLM requests,
and system performance with Prometheus exposition format.
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
)
from typing import Optional, Dict, Any
import logging
import time
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger(__name__)


# ============================================================================
# Metrics Registry
# ============================================================================

# Custom registry for isolation
registry = CollectorRegistry()


# ============================================================================
# Evaluation Metrics
# ============================================================================

evaluations_total = Counter(
    "evaluations_total",
    "Total number of evaluations run",
    ["dataset", "status"],  # status: started, completed, failed
    registry=registry,
)

evaluation_duration_seconds = Histogram(
    "evaluation_duration_seconds",
    "Evaluation duration in seconds",
    ["dataset"],
    buckets=[10, 30, 60, 120, 300, 600, 1800, 3600],
    registry=registry,
)

test_cases_evaluated_total = Counter(
    "test_cases_evaluated_total",
    "Total number of test cases evaluated",
    ["dataset", "status"],  # status: passed, failed
    registry=registry,
)

test_cases_passed_total = Counter(
    "test_cases_passed_total",
    "Total number of test cases that passed",
    ["dataset"],
    registry=registry,
)

test_cases_failed_total = Counter(
    "test_cases_failed_total",
    "Total number of test cases that failed",
    ["dataset"],
    registry=registry,
)

active_evaluations = Gauge(
    "active_evaluations",
    "Number of currently active evaluations",
    registry=registry,
)


# ============================================================================
# LLM Request Metrics
# ============================================================================

llm_requests_total = Counter(
    "llm_requests_total",
    "Total number of LLM API requests",
    ["provider", "model", "status"],  # status: success, error, timeout
    registry=registry,
)

llm_request_duration_seconds = Histogram(
    "llm_request_duration_seconds",
    "LLM request duration in seconds",
    ["provider", "model"],
    buckets=[0.5, 1, 2, 5, 10, 20, 30, 60, 120],
    registry=registry,
)

llm_tokens_used_total = Counter(
    "llm_tokens_used_total",
    "Total number of tokens used",
    ["provider", "model", "token_type"],  # token_type: input, output
    registry=registry,
)

llm_cost_usd_total = Counter(
    "llm_cost_usd_total",
    "Total cost in USD for LLM requests",
    ["provider", "model"],
    registry=registry,
)

llm_time_to_first_token_seconds = Histogram(
    "llm_time_to_first_token_seconds",
    "Time to first token in seconds (TTFT)",
    ["provider", "model"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
    registry=registry,
)


# ============================================================================
# Metric Evaluation Metrics
# ============================================================================

metric_evaluations_total = Counter(
    "metric_evaluations_total",
    "Total number of metric evaluations",
    ["metric_name", "status"],  # status: success, error
    registry=registry,
)

metric_evaluation_duration_seconds = Histogram(
    "metric_evaluation_duration_seconds",
    "Duration of metric evaluation in seconds",
    ["metric_name"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    registry=registry,
)


# ============================================================================
# System Metrics
# ============================================================================

system_errors_total = Counter(
    "system_errors_total",
    "Total system errors",
    ["error_type", "component"],
    registry=registry,
)

cache_hits_total = Counter(
    "cache_hits_total",
    "Total cache hits",
    ["cache_type"],  # cache_type: response, embedding, dataset
    registry=registry,
)

cache_misses_total = Counter(
    "cache_misses_total",
    "Total cache misses",
    ["cache_type"],
    registry=registry,
)


# ============================================================================
# Metrics Collector Class
# ============================================================================

class MetricsCollector:
    """
    Prometheus metrics collector for LLMOps-Eval.

    Provides high-level methods for recording evaluation events,
    LLM requests, and system metrics.
    """

    def __init__(self):
        """Initialize the metrics collector."""
        self._active_evaluations: Dict[str, float] = {}

    # ========================================================================
    # Evaluation Metrics
    # ========================================================================

    def record_evaluation_start(self, evaluation_id: str, dataset: str) -> None:
        """
        Record the start of an evaluation.

        Args:
            evaluation_id: Unique evaluation identifier
            dataset: Dataset name being evaluated
        """
        evaluations_total.labels(dataset=dataset, status="started").inc()
        active_evaluations.inc()
        self._active_evaluations[evaluation_id] = time.time()
        logger.debug(f"Recorded evaluation start: {evaluation_id}")

    def record_evaluation_complete(
        self,
        evaluation_id: str,
        dataset: str,
        status: str,
        total_tests: int = 0,
        passed_tests: int = 0,
        failed_tests: int = 0,
    ) -> None:
        """
        Record the completion of an evaluation.

        Args:
            evaluation_id: Unique evaluation identifier
            dataset: Dataset name
            status: Completion status (completed, failed, cancelled)
            total_tests: Total number of test cases
            passed_tests: Number of passed test cases
            failed_tests: Number of failed test cases
        """
        # Calculate duration
        start_time = self._active_evaluations.pop(evaluation_id, time.time())
        duration = time.time() - start_time

        # Record metrics
        evaluations_total.labels(dataset=dataset, status=status).inc()
        evaluation_duration_seconds.labels(dataset=dataset).observe(duration)
        active_evaluations.dec()

        # Record test case results
        test_cases_evaluated_total.labels(dataset=dataset, status="passed").inc(passed_tests)
        test_cases_evaluated_total.labels(dataset=dataset, status="failed").inc(failed_tests)
        test_cases_passed_total.labels(dataset=dataset).inc(passed_tests)
        test_cases_failed_total.labels(dataset=dataset).inc(failed_tests)

        logger.info(
            f"Recorded evaluation complete: {evaluation_id} "
            f"(duration: {duration:.2f}s, status: {status})"
        )

    def record_evaluation_failure(
        self,
        evaluation_id: str,
        dataset: str,
        error_type: str,
    ) -> None:
        """
        Record an evaluation failure.

        Args:
            evaluation_id: Unique evaluation identifier
            dataset: Dataset name
            error_type: Type of error that occurred
        """
        start_time = self._active_evaluations.pop(evaluation_id, None)
        if start_time:
            duration = time.time() - start_time
            evaluation_duration_seconds.labels(dataset=dataset).observe(duration)
            active_evaluations.dec()

        evaluations_total.labels(dataset=dataset, status="failed").inc()
        system_errors_total.labels(error_type=error_type, component="evaluation").inc()

        logger.error(f"Recorded evaluation failure: {evaluation_id} ({error_type})")

    # ========================================================================
    # LLM Request Metrics
    # ========================================================================

    def record_llm_request(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        duration_ms: float,
        ttft_ms: Optional[float] = None,
        status: str = "success",
    ) -> None:
        """
        Record an LLM API request.

        Args:
            provider: Provider name (openai, anthropic, ollama)
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost_usd: Cost in USD
            duration_ms: Request duration in milliseconds
            ttft_ms: Time to first token in milliseconds
            status: Request status (success, error, timeout)
        """
        # Convert to seconds
        duration_sec = duration_ms / 1000

        # Record metrics
        llm_requests_total.labels(provider=provider, model=model, status=status).inc()
        llm_request_duration_seconds.labels(provider=provider, model=model).observe(duration_sec)
        llm_tokens_used_total.labels(provider=provider, model=model, token_type="input").inc(input_tokens)
        llm_tokens_used_total.labels(provider=provider, model=model, token_type="output").inc(output_tokens)
        llm_cost_usd_total.labels(provider=provider, model=model).inc(cost_usd)

        # Record TTFT if available
        if ttft_ms is not None:
            ttft_sec = ttft_ms / 1000
            llm_time_to_first_token_seconds.labels(provider=provider, model=model).observe(ttft_sec)

        logger.debug(
            f"Recorded LLM request: {provider}:{model} "
            f"(tokens: {input_tokens + output_tokens}, cost: ${cost_usd:.6f})"
        )

    def record_llm_error(
        self,
        provider: str,
        model: str,
        error_type: str,
    ) -> None:
        """
        Record an LLM request error.

        Args:
            provider: Provider name
            model: Model name
            error_type: Type of error
        """
        llm_requests_total.labels(provider=provider, model=model, status="error").inc()
        system_errors_total.labels(error_type=error_type, component="llm").inc()

        logger.warning(f"Recorded LLM error: {provider}:{model} ({error_type})")

    # ========================================================================
    # Metric Evaluation Metrics
    # ========================================================================

    def record_metric_evaluation(
        self,
        metric_name: str,
        duration_ms: float,
        status: str = "success",
    ) -> None:
        """
        Record a metric evaluation.

        Args:
            metric_name: Name of the metric evaluated
            duration_ms: Evaluation duration in milliseconds
            status: Evaluation status (success, error)
        """
        duration_sec = duration_ms / 1000

        metric_evaluations_total.labels(metric_name=metric_name, status=status).inc()
        metric_evaluation_duration_seconds.labels(metric_name=metric_name).observe(duration_sec)

        logger.debug(f"Recorded metric evaluation: {metric_name} ({duration_sec:.4f}s)")

    # ========================================================================
    # Cache Metrics
    # ========================================================================

    def record_cache_hit(self, cache_type: str) -> None:
        """
        Record a cache hit.

        Args:
            cache_type: Type of cache (response, embedding, dataset)
        """
        cache_hits_total.labels(cache_type=cache_type).inc()

    def record_cache_miss(self, cache_type: str) -> None:
        """
        Record a cache miss.

        Args:
            cache_type: Type of cache (response, embedding, dataset)
        """
        cache_misses_total.labels(cache_type=cache_type).inc()

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_metrics_text(self) -> bytes:
        """
        Get all metrics in Prometheus exposition format.

        Returns:
            Metrics as bytes in Prometheus text format
        """
        return generate_latest(registry)

    def get_content_type(self) -> str:
        """
        Get the content type for metrics output.

        Returns:
            Content type string
        """
        return CONTENT_TYPE_LATEST

    @contextmanager
    def track_evaluation(self, evaluation_id: str, dataset: str):
        """
        Context manager for tracking an evaluation.

        Usage:
            with metrics.track_evaluation(eval_id, dataset):
                # ... run evaluation
                pass

        Args:
            evaluation_id: Unique evaluation identifier
            dataset: Dataset name
        """
        self.record_evaluation_start(evaluation_id, dataset)
        try:
            yield
            self.record_evaluation_complete(
                evaluation_id=evaluation_id,
                dataset=dataset,
                status="completed",
            )
        except Exception as e:
            self.record_evaluation_failure(
                evaluation_id=evaluation_id,
                dataset=dataset,
                error_type=type(e).__name__,
            )
            raise

    @contextmanager
    def track_llm_request(self, provider: str, model: str):
        """
        Context manager for tracking an LLM request.

        Usage:
            with metrics.track_llm_request("openai", "gpt-4o-mini") as tracker:
                response = await model.generate()
                tracker(
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    cost_usd=response.cost_usd,
                    ttft_ms=response.time_to_first_token_ms,
                )

        Args:
            provider: Provider name
            model: Model name

        Yields:
            Callable to record request details
        """
        start_time = time.time()
        ttft_time = None
        ttft_recorded = False

        def record_details(
            input_tokens: int,
            output_tokens: int,
            cost_usd: float,
            ttft_ms: Optional[float] = None,
        ):
            """Record the request details."""
            nonlocal ttft_time, ttft_recorded
            duration_ms = (time.time() - start_time) * 1000
            self.record_llm_request(
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                duration_ms=duration_ms,
                ttft_ms=ttft_ms,
            )
            ttft_recorded = True

        try:
            yield record_details
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.record_llm_request(
                provider=provider,
                model=model,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                duration_ms=duration_ms,
                status="error",
            )
            self.record_llm_error(provider, model, type(e).__name__)
            raise

    @contextmanager
    def track_metric(self, metric_name: str):
        """
        Context manager for tracking a metric evaluation.

        Usage:
            with metrics.track_metric("semantic_similarity"):
                result = await metric.evaluate(...)
        """
        start_time = time.time()
        try:
            yield
            duration_ms = (time.time() - start_time) * 1000
            self.record_metric_evaluation(metric_name, duration_ms, "success")
        except Exception:
            duration_ms = (time.time() - start_time) * 1000
            self.record_metric_evaluation(metric_name, duration_ms, "error")
            raise


# ============================================================================
# Singleton Instance
# ============================================================================

metrics = MetricsCollector()


# ============================================================================
# Decorators
# ============================================================================

def track_llm_call(provider: str, model: str):
    """
    Decorator to track LLM calls.

    Usage:
        @track_llm_call("openai", "gpt-4o-mini")
        async def generate(prompt: str) -> LLMResponse:
            # ... generation logic
            return response
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with metrics.track_llm_request(provider, model) as record:
                response = await func(*args, **kwargs)
                record(
                    input_tokens=getattr(response, "input_tokens", 0),
                    output_tokens=getattr(response, "output_tokens", 0),
                    cost_usd=getattr(response, "cost_usd", 0),
                    ttft_ms=getattr(response, "time_to_first_token_ms", None),
                )
                return response

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with metrics.track_llm_request(provider, model) as record:
                response = func(*args, **kwargs)
                record(
                    input_tokens=getattr(response, "input_tokens", 0),
                    output_tokens=getattr(response, "output_tokens", 0),
                    cost_usd=getattr(response, "cost_usd", 0),
                    ttft_ms=getattr(response, "time_to_first_token_ms", None),
                )
                return response

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def track_metric_evaluation(metric_name: str):
    """
    Decorator to track metric evaluations.

    Usage:
        @track_metric_evaluation("semantic_similarity")
        async def evaluate(response: str, expected: str) -> MetricResult:
            # ... evaluation logic
            return result
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with metrics.track_metric(metric_name):
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with metrics.track_metric(metric_name):
                return func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# ============================================================================
# FastAPI Integration
# ============================================================================

def setup_prometheus_endpoint(app):
    """
    Add Prometheus metrics endpoint to FastAPI app.

    Usage:
        from fastapi import FastAPI
        from src.monitoring.prometheus_metrics import setup_prometheus_endpoint

        app = FastAPI()
        setup_prometheus_endpoint(app)
    """
    from fastapi import Response
    from fastapi.responses import Response as FastAPIResponse

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics():
        """Prometheus metrics endpoint."""
        return FastAPIResponse(
            content=metrics.get_metrics_text(),
            media_type=metrics.get_content_type(),
        )

    @app.get("/prometheus", include_in_schema=False)
    async def prometheus_metrics_alternate():
        """Alternative Prometheus metrics endpoint."""
        return FastAPIResponse(
            content=metrics.get_metrics_text(),
            media_type=metrics.get_content_type(),
        )


# ============================================================================
# Export
# ============================================================================

__all__ = [
    # Metrics
    "evaluations_total",
    "evaluation_duration_seconds",
    "test_cases_evaluated_total",
    "test_cases_passed_total",
    "test_cases_failed_total",
    "active_evaluations",
    "llm_requests_total",
    "llm_request_duration_seconds",
    "llm_tokens_used_total",
    "llm_cost_usd_total",
    "llm_time_to_first_token_seconds",
    "metric_evaluations_total",
    "metric_evaluation_duration_seconds",
    "system_errors_total",
    "cache_hits_total",
    "cache_misses_total",
    # Collector
    "MetricsCollector",
    "metrics",
    # Decorators
    "track_llm_call",
    "track_metric_evaluation",
    # FastAPI
    "setup_prometheus_endpoint",
]
