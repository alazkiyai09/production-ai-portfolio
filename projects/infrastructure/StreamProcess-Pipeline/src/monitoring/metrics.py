"""
Comprehensive monitoring and metrics collection for StreamProcess-Pipeline.

Provides Prometheus metrics for all pipeline components including ingestion,
processing, embeddings, and vector store operations.
"""

import os
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    Summary,
    start_http_server,
    exposition,
)


# ============================================================================
# Registry
# ============================================================================

# Default registry for all metrics
registry = CollectorRegistry()


# ============================================================================
# Application Info
# ============================================================================

app_info = Info(
    "streamprocess",
    "StreamProcess-Pipeline application information",
    registry=registry,
)

build_info = Info(
    "streamprocess_build",
    "StreamProcess-Pipeline build information",
    registry=registry,
)


def set_app_info(
    name: str,
    version: str,
    environment: str,
) -> None:
    """Set application information."""
    app_info.info({
        "name": name,
        "version": version,
        "environment": environment,
    })


def set_build_info(
    commit: str,
    build_time: str,
) -> None:
    """Set build information."""
    build_info.info({
        "commit": commit,
        "build_time": build_time,
    })


# ============================================================================
# Component Status Enums
# ============================================================================

class ComponentStatus(str, Enum):
    """Component health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


# ============================================================================
# Ingestion Metrics
# ============================================================================

# Counters
records_ingested_total = Counter(
    "streamprocess_ingestion_records_total",
    "Total number of records ingested",
    ["event_type", "status"],  # status: success, failed
    registry=registry,
)

ingestion_requests_total = Counter(
    "streamprocess_ingestion_requests_total",
    "Total number of ingestion requests",
    ["endpoint", "status"],
    registry=registry,
)

validation_failures_total = Counter(
    "streamprocess_ingestion_validation_failures_total",
    "Total number of validation failures",
    ["reason"],  # reason: duplicate, invalid_schema, rate_limited, etc.
    registry=registry,
)

batches_ingested_total = Counter(
    "streamprocess_ingestion_batches_total",
    "Total number of batches ingested",
    ["status"],
    registry=registry,
)

# Histograms
ingestion_latency_seconds = Histogram(
    "streamprocess_ingestion_latency_seconds",
    "Ingestion request latency",
    ["endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=registry,
)

batch_size_histogram = Histogram(
    "streamprocess_ingestion_batch_size",
    "Batch size distribution",
    buckets=[1, 10, 50, 100, 500, 1000],
    registry=registry,
)

# Gauges
ingestion_queue_depth = Gauge(
    "streamprocess_ingestion_queue_depth",
    "Current ingestion queue depth",
    registry=registry,
)

ingestion_rate = Gauge(
    "streamprocess_ingestion_rate",
    "Current ingestion rate (records/second)",
    registry=registry,
)


# ============================================================================
# Processing Metrics
# ============================================================================

# Counters
records_processed_total = Counter(
    "streamprocess_processing_records_total",
    "Total number of records processed",
    ["status"],  # status: success, failed
    registry=registry,
)

processing_jobs_total = Counter(
    "streamprocess_processing_jobs_total",
    "Total number of processing jobs",
    ["status"],
    registry=registry,
)

processing_errors_total = Counter(
    "streamprocess_processing_errors_total",
    "Total number of processing errors",
    ["error_type"],  # error_type: transform, embedding, vector_db, metadata
    registry=registry,
)

# Histograms
processing_latency_seconds = Histogram(
    "streamprocess_processing_latency_seconds",
    "Processing job latency",
    ["job_type"],  # job_type: batch, single, transform, embed, etc.
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0],
    registry=registry,
)

# Gauges
processing_queue_depth = Gauge(
    "streamprocess_processing_queue_depth",
    "Current processing queue depth",
    ["queue_name"],  # queue_name: processing, embedding, vector_db, database
    registry=registry,
)

active_workers = Gauge(
    "streamprocess_processing_active_workers",
    "Number of active processing workers",
    registry=registry,
)

worker_utilization_percent = Gauge(
    "streamprocess_processing_worker_utilization",
    "Worker utilization percentage",
    registry=registry,
)


# ============================================================================
# Embedding Metrics
# ============================================================================

# Counters
embeddings_generated_total = Counter(
    "streamprocess_embedding_generated_total",
    "Total number of embeddings generated",
    ["model", "status"],
    registry=registry,
)

embedding_cache_hits_total = Counter(
    "streamprocess_embedding_cache_hits_total",
    "Total number of embedding cache hits",
    ["cache_level"],  # cache_level: l1, l2
    registry=registry,
)

embedding_cache_misses_total = Counter(
    "streamprocess_embedding_cache_misses_total",
    "Total number of embedding cache misses",
    registry=registry,
)

model_loads_total = Counter(
    "streamprocess_embedding_model_loads_total",
    "Total number of model loads",
    ["model"],
    registry=registry,
)

# Histograms
embedding_latency_seconds = Histogram(
    "streamprocess_embedding_latency_seconds",
    "Embedding generation latency",
    ["model", "batch_size"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=registry,
)

embedding_batch_size = Histogram(
    "streamprocess_embedding_batch_size",
    "Embedding batch size distribution",
    buckets=[1, 8, 16, 32, 64, 128, 256, 512],
    registry=registry,
)

# Gauges
embedding_model_load_time_seconds = Gauge(
    "streamprocess_embedding_model_load_time_seconds",
    "Time taken to load embedding model",
    ["model"],
    registry=registry,
)

embedding_model_memory_bytes = Gauge(
    "streamprocess_embedding_model_memory_bytes",
    "Memory used by embedding model",
    ["model"],
    registry=registry,
)

embedding_rate = Gauge(
    "streamprocess_embedding_rate",
    "Current embedding generation rate (embeddings/second)",
    registry=registry,
)


# ============================================================================
# Vector Store Metrics
# ============================================================================

# Counters
vector_store_operations_total = Counter(
    "streamprocess_vector_store_operations_total",
    "Total number of vector store operations",
    ["operation", "status"],  # operation: upsert, query, delete
    registry=registry,
)

# Histograms
vector_store_upsert_latency_seconds = Histogram(
    "streamprocess_vector_store_upsert_latency_seconds",
    "Vector store upsert latency",
    ["store_type"],  # store_type: chroma, qdrant
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=registry,
)

vector_store_query_latency_seconds = Histogram(
    "streamprocess_vector_store_query_latency_seconds",
    "Vector store query latency",
    ["store_type"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=registry,
)

# Gauges
vector_store_size = Gauge(
    "streamprocess_vector_store_size",
    "Total number of vectors in store",
    ["collection_name", "store_type"],
    registry=registry,
)

vector_store_memory_bytes = Gauge(
    "streamprocess_vector_store_memory_bytes",
    "Memory used by vector store",
    ["store_type"],
    registry=registry,
)

vector_store_query_results_count = Histogram(
    "streamprocess_vector_store_query_results",
    "Number of results returned from vector queries",
    buckets=[1, 5, 10, 20, 50, 100, 200, 500],
    registry=registry,
)


# ============================================================================
# Database Metrics
# ============================================================================

# Counters
database_queries_total = Counter(
    "streamprocess_database_queries_total",
    "Total number of database queries",
    ["operation", "table", "status"],
    registry=registry,
)

database_connections_total = Counter(
    "streamprocess_database_connections_total",
    "Total number of database connections",
    ["state"],  # state: created, closed, reused
    registry=registry,
)

# Histograms
database_query_latency_seconds = Histogram(
    "streamprocess_database_query_latency_seconds",
    "Database query latency",
    ["operation", "table"],
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=registry,
)

# Gauges
database_pool_size = Gauge(
    "streamprocess_database_pool_size",
    "Database connection pool size",
    ["state"],  # state: active, idle
    registry=registry,
)

database_transaction_duration_seconds = Gauge(
    "streamprocess_database_transaction_duration_seconds",
    "Database transaction duration",
    registry=registry,
)


# ============================================================================
# System Metrics
# ============================================================================

# Gauges
system_memory_usage_bytes = Gauge(
    "streamprocess_system_memory_usage_bytes",
    "System memory usage",
    ["type"],  # type: rss, vms, shared
    registry=registry,
)

system_cpu_percent = Gauge(
    "streamprocess_system_cpu_percent",
    "System CPU usage percentage",
    registry=registry,
)

system_disk_usage_bytes = Gauge(
    "streamprocess_system_disk_usage_bytes",
    "System disk usage",
    ["mount_point"],
    registry=registry,
)

process_uptime_seconds = Gauge(
    "streamprocess_process_uptime_seconds",
    "Process uptime in seconds",
    registry=registry,
)

process_open_fds = Gauge(
    "streamprocess_process_open_fds",
    "Number of open file descriptors",
    registry=registry,
)


# ============================================================================
# Celery Metrics
# ============================================================================

# Counters
celery_tasks_total = Counter(
    "streamprocess_celery_tasks_total",
    "Total number of Celery tasks",
    ["task_name", "status"],
    registry=registry,
)

celery_worker_restarts_total = Counter(
    "streamprocess_celery_worker_restarts_total",
    "Total number of worker restarts",
    registry=registry,
)

# Histograms
celery_task_runtime_seconds = Histogram(
    "streamprocess_celery_task_runtime_seconds",
    "Celery task runtime",
    ["task_name"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 300.0],
    registry=registry,
)

# Gauges
celery_queue_length = Gauge(
    "streamprocess_celery_queue_length",
    "Celery queue length",
    ["queue_name"],
    registry=registry,
)

celery_active_tasks = Gauge(
    "streamprocess_celery_active_tasks",
    "Number of active Celery tasks",
    registry=registry,
)

celery_worker_heartbeat = Gauge(
    "streamprocess_celery_worker_heartbeat",
    "Last worker heartbeat timestamp",
    ["worker_name"],
    registry=registry,
)


# ============================================================================
# HTTP Metrics
# ============================================================================

http_requests_total = Counter(
    "streamprocess_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
    registry=registry,
)

http_request_duration_seconds = Histogram(
    "streamprocess_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=registry,
)

http_requests_in_progress = Gauge(
    "streamprocess_http_requests_in_progress",
    "HTTP requests currently in progress",
    ["method", "endpoint"],
    registry=registry,
)

http_response_size_bytes = Histogram(
    "streamprocess_http_response_size_bytes",
    "HTTP response size",
    ["endpoint"],
    buckets=[100, 1000, 10000, 100000, 1000000, 10000000],
    registry=registry,
)


# ============================================================================
# Alert Thresholds
# ============================================================================

class AlertThresholds:
    """Alert threshold configuration."""

    # Ingestion alerts
    INGESTION_RATE_LOW_THRESHOLD = 10  # records/sec
    INGESTION_LATENCY_HIGH_THRESHOLD = 5.0  # seconds
    VALIDATION_FAILURE_RATE_THRESHOLD = 0.1  # 10%

    # Processing alerts
    PROCESSING_LATENCY_HIGH_THRESHOLD = 60.0  # seconds
    QUEUE_DEPTH_HIGH_THRESHOLD = 10000  # records
    ERROR_RATE_THRESHOLD = 0.05  # 5%

    # Embedding alerts
    EMBEDDING_LATENCY_HIGH_THRESHOLD = 2.0  # seconds
    CACHE_HIT_RATE_LOW_THRESHOLD = 0.5  # 50%

    # Vector store alerts
    VECTOR_QUERY_LATENCY_HIGH_THRESHOLD = 1.0  # seconds
    VECTOR_STORE_SIZE_LOW_THRESHOLD = 100  # vectors

    # Database alerts
    DB_QUERY_LATENCY_HIGH_THRESHOLD = 1.0  # seconds
    DB_POOL_EXHAUSTED_THRESHOLD = 0.9  # 90% utilization

    # System alerts
    MEMORY_USAGE_HIGH_THRESHOLD = 0.9  # 90%
    CPU_USAGE_HIGH_THRESHOLD = 0.9  # 90%
    DISK_USAGE_HIGH_THRESHOLD = 0.9  # 90%


# ============================================================================
# Health Status
# ============================================================================

class ComponentHealth:
    """Health status for a component."""

    def __init__(
        self,
        name: str,
        status: ComponentStatus = ComponentStatus.UNKNOWN,
        message: str = "",
        metrics: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.status = status
        self.message = message
        self.metrics = metrics or {}
        self.last_check = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "metrics": self.metrics,
            "last_check": self.last_check.isoformat(),
        }


class HealthChecker:
    """Aggregates health status from all components."""

    def __init__(self):
        """Initialize health checker."""
        self.components: Dict[str, ComponentHealth] = {}
        self.start_time = time.time()

    def set_component_status(
        self,
        name: str,
        status: ComponentStatus,
        message: str = "",
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update component health status."""
        self.components[name] = ComponentHealth(name, status, message, metrics)

    def get_overall_status(self) -> ComponentStatus:
        """Get overall system health status."""
        if not self.components:
            return ComponentStatus.UNKNOWN

        statuses = [c.status for c in self.components.values()]

        if ComponentStatus.UNHEALTHY in statuses:
            return ComponentStatus.UNHEALTHY
        elif ComponentStatus.DEGRADED in statuses:
            return ComponentStatus.DEGRADED
        elif all(s == ComponentStatus.HEALTHY for s in statuses):
            return ComponentStatus.HEALTHY
        else:
            return ComponentStatus.UNKNOWN

    def get_health_report(self) -> Dict[str, Any]:
        """Get complete health report."""
        overall = self.get_overall_status()
        uptime = time.time() - self.start_time

        return {
            "status": overall.value,
            "uptime_seconds": uptime,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {name: c.to_dict() for name, c in self.components.items()},
        }


# ============================================================================
# Metrics Collector
# ============================================================================

class MetricsCollector:
    """
    Centralized metrics collector for StreamProcess-Pipeline.

    Provides convenient methods for recording metrics across all components.
    """

    def __init__(self):
        """Initialize metrics collector."""
        self._start_time = time.time()
        self._health_checker = HealthChecker()

        # Update process uptime
        process_uptime_seconds.set(0)

        # Update periodically
        self._update_system_metrics()

    def _update_system_metrics(self) -> None:
        """Update system metrics."""
        try:
            import psutil

            # Memory
            process = psutil.Process()
            memory_info = process.memory_info()
            system_memory_usage_bytes.labels(type="rss").set(memory_info.rss)
            system_memory_usage_bytes.labels(type="vms").set(memory_info.vms)

            # CPU
            cpu_percent = process.cpu_percent()
            system_cpu_percent.set(cpu_percent)

            # Open file descriptors
            try:
                system_memory_usage_bytes.labels(type="fds").set(process.num_fds())
            except:
                pass

            # Disk
            for mount in psutil.disk_partitions():
                if mount.fstype:
                    try:
                        usage = psutil.disk_usage(mount.mountpoint)
                        system_disk_usage_bytes.labels(mount_point=mount.mountpoint).set(usage.used)
                    except:
                        pass

        except ImportError:
            # psutil not available
            pass

        # Update uptime
        uptime = time.time() - self._start_time
        process_uptime_seconds.set(uptime)

    # ========================================================================
    # Ingestion Metrics
    # ========================================================================

    def record_ingestion(
        self,
        batch_size: int,
        latency: float,
        success: bool,
        endpoint: str = "/ingest/batch",
        event_type: str = "unknown",
    ) -> None:
        """
        Record ingestion metrics.

        Args:
            batch_size: Number of records in batch
            latency: Request latency in seconds
            success: Whether ingestion succeeded
            endpoint: API endpoint
            event_type: Event type
        """
        status = "success" if success else "failed"

        # Counters
        ingestion_requests_total.labels(endpoint=endpoint, status=status).inc()
        records_ingested_total.labels(event_type=event_type, status=status).inc(batch_size)
        batches_ingested_total.labels(status=status).inc()

        # Histograms
        ingestion_latency_seconds.labels(endpoint=endpoint).observe(latency)
        batch_size_histogram.observe(batch_size)

        # Gauges
        ingestion_rate.set(batch_size / latency if latency > 0 else 0)

    def record_validation_failure(self, reason: str, count: int = 1) -> None:
        """
        Record validation failure.

        Args:
            reason: Failure reason
            count: Number of failures
        """
        validation_failures_total.labels(reason=reason).inc(count)

    def set_ingestion_queue_depth(self, depth: int) -> None:
        """Set ingestion queue depth."""
        ingestion_queue_depth.set(depth)

    # ========================================================================
    # Processing Metrics
    # ========================================================================

    def record_processing(
        self,
        records: int,
        latency: float,
        success: bool,
        job_type: str = "batch",
        error_type: Optional[str] = None,
    ) -> None:
        """
        Record processing metrics.

        Args:
            records: Number of records processed
            latency: Processing latency in seconds
            success: Whether processing succeeded
            job_type: Type of job
            error_type: Type of error if failed
        """
        status = "success" if success else "failed"

        # Counters
        records_processed_total.labels(status=status).inc(records)
        processing_jobs_total.labels(status=status).inc()

        if not success and error_type:
            processing_errors_total.labels(error_type=error_type).inc()

        # Histograms
        processing_latency_seconds.labels(job_type=job_type).observe(latency)

    def set_processing_queue_depth(self, queue_name: str, depth: int) -> None:
        """Set processing queue depth."""
        processing_queue_depth.labels(queue_name=queue_name).set(depth)

    def set_active_workers(self, count: int) -> None:
        """Set number of active workers."""
        active_workers.set(count)

    # ========================================================================
    # Embedding Metrics
    # ========================================================================

    def record_embedding(
        self,
        batch_size: int,
        latency: float,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_hit: bool = False,
    ) -> None:
        """
        Record embedding generation metrics.

        Args:
            batch_size: Batch size
            latency: Generation latency in seconds
            model: Model name
            cache_hit: Whether cache was hit
        """
        # Counters
        if cache_hit:
            embedding_cache_hits_total.labels(cache_level="l1").inc(batch_size)
        else:
            embeddings_generated_total.labels(model=model, status="success").inc(batch_size)
            embedding_cache_misses_total.inc()

        # Histograms
        embedding_latency_seconds.labels(
            model=model,
            batch_size=str(batch_size),
        ).observe(latency)
        embedding_batch_size.observe(batch_size)

        # Gauges
        embedding_rate.set(batch_size / latency if latency > 0 else 0)

    def record_model_load(self, model: str, load_time: float, memory_bytes: int = 0) -> None:
        """
        Record model load metrics.

        Args:
            model: Model name
            load_time: Load time in seconds
            memory_bytes: Memory used
        """
        model_loads_total.labels(model=model).inc()
        embedding_model_load_time_seconds.labels(model=model).set(load_time)

        if memory_bytes > 0:
            embedding_model_memory_bytes.labels(model=model).set(memory_bytes)

    # ========================================================================
    # Vector Store Metrics
    # ========================================================================

    def record_vector_op(
        self,
        operation: str,
        latency: float,
        store_type: str = "chroma",
        success: bool = True,
        result_count: int = 0,
    ) -> None:
        """
        Record vector store operation.

        Args:
            operation: Operation type (upsert, query, delete)
            latency: Operation latency in seconds
            store_type: Store type (chroma, qdrant)
            success: Whether operation succeeded
            result_count: Number of results (for queries)
        """
        status = "success" if success else "failed"

        # Counters
        vector_store_operations_total.labels(operation=operation, status=status).inc()

        # Histograms
        if operation == "upsert":
            vector_store_upsert_latency_seconds.labels(store_type=store_type).observe(latency)
        elif operation == "query":
            vector_store_query_latency_seconds.labels(store_type=store_type).observe(latency)
            vector_store_query_results_count.observe(result_count)

    def set_vector_store_size(self, collection_name: str, size: int, store_type: str = "chroma") -> None:
        """Set vector store size."""
        vector_store_size.labels(collection_name=collection_name, store_type=store_type).set(size)

    # ========================================================================
    # Database Metrics
    # ========================================================================

    def record_db_query(
        self,
        operation: str,
        table: str,
        latency: float,
        success: bool = True,
    ) -> None:
        """
        Record database query.

        Args:
            operation: Query operation (select, insert, update, delete)
            table: Table name
            latency: Query latency in seconds
            success: Whether query succeeded
        """
        status = "success" if success else "failed"
        database_queries_total.labels(operation=operation, table=table, status=status).inc()
        database_query_latency_seconds.labels(operation=operation, table=table).observe(latency)

    def set_db_pool_size(self, active: int, idle: int) -> None:
        """Set database pool sizes."""
        database_pool_size.labels(state="active").set(active)
        database_pool_size.labels(state="idle").set(idle)

    # ========================================================================
    # Health Status
    # ========================================================================

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get overall health status.

        Returns:
            Health status dictionary
        """
        self._update_system_metrics()
        return self._health_checker.get_health_report()

    def update_component_health(
        self,
        name: str,
        status: ComponentStatus,
        message: str = "",
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update component health status.

        Args:
            name: Component name
            status: Health status
            message: Status message
            metrics: Optional metrics
        """
        self._health_checker.set_component_status(name, status, message, metrics)


# ============================================================================
# Global Collector Instance
# ============================================================================

_collector: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """
    Get global metrics collector instance.

    Returns:
        MetricsCollector
    """
    global _collector

    if _collector is None:
        _collector = MetricsCollector()

    return _collector


# ============================================================================
# Start Metrics Server
# ============================================================================

def start_metrics_server(port: int = 9090, addr: str = "0.0.0.0") -> None:
    """
    Start Prometheus metrics HTTP server.

    Args:
        port: Port to listen on
        addr: Address to bind to
    """
    start_http_server(port, addr, registry=registry)


# ============================================================================
# Metrics Export
# ============================================================================

def generate_metrics() -> str:
    """
    Generate Prometheus metrics text format.

    Returns:
        Metrics in Prometheus text format
    """
    return exposition.generate_latest(registry).decode("utf-8")


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Info
    "app_info",
    "build_info",
    "set_app_info",
    "set_build_info",
    # Enums
    "ComponentStatus",
    "AlertThresholds",
    # Health
    "ComponentHealth",
    "HealthChecker",
    # Collector
    "MetricsCollector",
    "get_metrics",
    # Server
    "start_metrics_server",
    "generate_metrics",
    # Component exports
    "records_ingested_total",
    "ingestion_latency_seconds",
    "batch_size_histogram",
    "records_processed_total",
    "processing_latency_seconds",
    "embeddings_generated_total",
    "embedding_latency_seconds",
    "vector_store_size",
    "vector_store_upsert_latency_seconds",
]
