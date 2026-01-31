"""
FastAPI application for LLMOps-Eval.

Provides REST API for running evaluations, managing datasets,
generating reports, and monitoring system health.
"""

from fastapi import (
    FastAPI,
    HTTPException,
    BackgroundTasks,
    UploadFile,
    File,
    Query,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, List
from datetime import datetime
from pathlib import Path
import uuid
import asyncio
import logging
from contextlib import asynccontextmanager

# Prometheus metrics
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from fastapi.responses import Response

# Import our modules
from src.runners.eval_runner import (
    EvaluationRunner,
    EvaluationConfig,
    ModelConfig,
    EvaluationResult,
    TestResult,
)
from src.datasets.dataset_manager import (
    DatasetManager,
    TestDataset,
    TestCase,
)
from src.reporting.report_generator import (
    ReportGenerator,
    generate_report,
)
from src.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# Prometheus Metrics
# ============================================================================

# Request metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

http_request_duration = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["method", "endpoint"],
)

# Evaluation metrics
evaluation_total = Counter(
    "evaluation_total",
    "Total evaluations run",
    ["dataset", "status"],
)

evaluation_duration = Histogram(
    "evaluation_duration_seconds",
    "Evaluation duration",
    ["dataset"],
)

evaluation_cost_usd = Gauge(
    "evaluation_cost_usd",
    "Total cost of all evaluations in USD",
)

active_evaluations = Gauge(
    "active_evaluations",
    "Number of currently running evaluations",
)

# Model metrics
model_requests_total = Counter(
    "model_requests_total",
    "Total model requests",
    ["provider", "model"],
)

model_latency = Histogram(
    "model_latency_seconds",
    "Model request latency",
    ["provider", "model"],
)


# ============================================================================
# Pydantic Models
# ============================================================================

class EvaluateRequest(BaseModel):
    """Request to start an evaluation."""

    name: str = Field(..., description="Evaluation name")
    dataset: str = Field(..., description="Dataset name")
    models: List[Dict[str, str]] = Field(..., description="List of model configs")
    metrics: List[str] = Field(..., description="Metrics to evaluate")
    dataset_version: str = Field(default="latest", description="Dataset version")
    categories: Optional[List[str]] = Field(None, description="Filter by categories")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    parallel: int = Field(default=5, ge=1, le=50, description="Parallel evaluations")
    timeout_seconds: int = Field(default=60, ge=10, description="Timeout per test")
    sample_size: Optional[int] = Field(None, ge=1, description="Sample size")
    save_results: bool = Field(default=True, description="Save results")


class EvaluateResponse(BaseModel):
    """Response from starting an evaluation."""

    evaluation_id: str
    status: str
    message: str


class EvaluationStatusResponse(BaseModel):
    """Response for evaluation status."""

    evaluation_id: str
    status: str  # pending, running, completed, failed
    progress: float  # 0-100
    current_test: Optional[str] = None
    total_tests: int
    completed_tests: int
    failed_tests: int
    estimated_time_remaining: Optional[float] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    error: Optional[str] = None


class ComparisonRequest(BaseModel):
    """Request to compare evaluations."""

    evaluation_ids: List[str] = Field(..., description="Evaluation IDs to compare")
    metrics: Optional[List[str]] = Field(None, description="Metrics to compare")


class DatasetUploadRequest(BaseModel):
    """Request to upload a dataset."""

    name: str = Field(..., description="Dataset name")
    version: str = Field(default="1.0", description="Dataset version")
    description: str = Field(default="", description="Dataset description")
    format: str = Field(default="yaml", description="Format (yaml or json)")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    components: Dict[str, str]
    uptime_seconds: float


# ============================================================================
# In-Memory Storage
# ============================================================================

class EvaluationStore:
    """Store for evaluation state and results."""

    def __init__(self):
        self.evaluations: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, EvaluationResult] = {}

    def create(self, config: EvaluationConfig) -> str:
        """Create a new evaluation entry."""
        evaluation_id = str(uuid.uuid4())
        self.evaluations[evaluation_id] = {
            "id": evaluation_id,
            "config": config,
            "status": "pending",
            "progress": 0.0,
            "current_test": None,
            "total_tests": 0,
            "completed_tests": 0,
            "failed_tests": 0,
            "start_time": None,
            "end_time": None,
            "error": None,
            "created_at": datetime.utcnow().isoformat(),
        }
        return evaluation_id

    def update(self, evaluation_id: str, **kwargs) -> None:
        """Update evaluation state."""
        if evaluation_id in self.evaluations:
            self.evaluations[evaluation_id].update(kwargs)

    def get(self, evaluation_id: str) -> Optional[Dict[str, Any]]:
        """Get evaluation state."""
        return self.evaluations.get(evaluation_id)

    def list_all(self) -> List[Dict[str, Any]]:
        """List all evaluations."""
        return list(self.evaluations.values())

    def set_result(self, evaluation_id: str, result: EvaluationResult) -> None:
        """Store evaluation result."""
        self.results[evaluation_id] = result

    def get_result(self, evaluation_id: str) -> Optional[EvaluationResult]:
        """Get evaluation result."""
        return self.results.get(evaluation_id)


# ============================================================================
# Global State
# ============================================================================

store = EvaluationStore()
runner = EvaluationRunner()
report_generator = ReportGenerator()
dataset_manager = DatasetManager()

# Track app start time
app_start_time = datetime.utcnow()


# ============================================================================
# Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Install security filter to prevent API key exposure in logs
    try:
        from shared.security import SensitiveDataFilter
        root_logger = logging.getLogger()
        root_logger.addFilter(SensitiveDataFilter())
        logger.info("Security filter installed - API keys will be redacted from logs")
    except ImportError:
        logger.warning("Shared security module not available - API keys may be exposed in logs")

    logger.info("Starting LLMOps-Eval API")
    yield
    logger.info("Shutting down LLMOps-Eval API")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="LLMOps-Eval API",
    description="LLM evaluation and deployment pipeline",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Rate Limiting
# ============================================================================

try:
    from shared.rate_limit import limiter, rate_limit_exception_handler, RateLimitExceeded
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exception_handler)
    logger.info("Rate limiting enabled")
except ImportError:
    logger.warning("Shared rate limiting module not available - rate limiting disabled")


# ============================================================================
# Middleware
# ============================================================================

@app.middleware("http")
async def track_requests(request, call_next):
    """Track request metrics."""
    import time

    start_time = time.time()

    # Process request
    response = await call_next(request)

    # Track metrics
    duration = time.time() - start_time
    http_requests_total.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()
    http_request_duration.labels(
        method=request.method,
        endpoint=request.url.path,
    ).observe(duration)

    return response


# ============================================================================
# Health Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns system health status and component availability.
    """
    uptime = (datetime.utcnow() - app_start_time).total_seconds()

    # Check components
    components = {
        "api": "healthy",
        "datasets": "healthy" if settings.datasets_dir.exists() else "unavailable",
        "results": "healthy" if settings.results_dir.exists() else "unavailable",
    }

    # Check if API keys are configured
    if settings.openai_api_key:
        components["openai"] = "configured"
    else:
        components["openai"] = "not_configured"

    if settings.anthropic_api_key:
        components["anthropic"] = "configured"
    else:
        components["anthropic"] = "not_configured"

    return HealthResponse(
        status="healthy",
        version="0.1.0",
        components=components,
        uptime_seconds=uptime,
    )


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "LLMOps-Eval API",
        "version": "0.1.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "evaluate": "/evaluate",
            "datasets": "/datasets",
            "reports": "/reports/{evaluation_id}",
            "metrics": "/metrics",
            "docs": "/docs",
        },
    }


# ============================================================================
# Evaluation Endpoints
# ============================================================================

@app.post(
    "/evaluate",
    response_model=EvaluateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Evaluations"],
)
async def create_evaluation(
    request: EvaluateRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start a new evaluation.

    Runs the evaluation in the background and returns immediately
    with an evaluation ID.
    """
    try:
        # Convert models to ModelConfig
        model_configs = [ModelConfig(**m) for m in request.models]

        # Create config
        config = EvaluationConfig(
            name=request.name,
            dataset=request.dataset,
            models=model_configs,
            metrics=request.metrics,
            dataset_version=request.dataset_version,
            categories=request.categories,
            tags=request.tags,
            parallel=request.parallel,
            timeout_seconds=request.timeout_seconds,
            sample_size=request.sample_size,
            save_results=request.save_results,
        )

        # Create evaluation entry
        evaluation_id = store.create(config)

        # Add background task
        background_tasks.add_task(run_evaluation_task, evaluation_id, config)

        active_evaluations.inc()

        return EvaluateResponse(
            evaluation_id=evaluation_id,
            status="pending",
            message=f"Evaluation '{request.name}' queued successfully",
        )

    except Exception as e:
        logger.error(f"Error creating evaluation: {e}")
        raise HTTPException(status_code=400, detail=str(e))


async def run_evaluation_task(evaluation_id: str, config: EvaluationConfig):
    """Background task to run evaluation."""
    try:
        # Update status
        store.update(evaluation_id, status="running", start_time=datetime.utcnow().isoformat())
        logger.info(f"Starting evaluation {evaluation_id}")

        # Run evaluation
        result = await runner.run(config)

        # Store result
        store.set_result(evaluation_id, result)

        # Update metrics
        evaluation_total.labels(
            dataset=config.dataset,
            status="completed",
        ).inc()
        evaluation_duration.labels(dataset=config.dataset).observe(result.duration_seconds)
        evaluation_cost_usd.inc(result.total_cost_usd)

        # Update status
        store.update(
            evaluation_id,
            status="completed",
            progress=100.0,
            end_time=datetime.utcnow().isoformat(),
            total_tests=result.summary.get("total_tests", 0),
            completed_tests=result.summary.get("successful_tests", 0),
            failed_tests=result.summary.get("failed_tests", 0),
        )

        logger.info(f"Completed evaluation {evaluation_id}")

    except Exception as e:
        logger.error(f"Evaluation {evaluation_id} failed: {e}")

        # Update metrics
        evaluation_total.labels(
            dataset=config.dataset,
            status="failed",
        ).inc()

        # Update status
        store.update(
            evaluation_id,
            status="failed",
            error=str(e),
            end_time=datetime.utcnow().isoformat(),
        )

    finally:
        active_evaluations.dec()


@app.get(
    "/evaluate/{evaluation_id}",
    tags=["Evaluations"],
)
async def get_evaluation(evaluation_id: str):
    """
    Get evaluation result.

    Returns the full evaluation result if complete, or status if still running.
    """
    evaluation = store.get(evaluation_id)

    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    if evaluation["status"] == "completed":
        result = store.get_result(evaluation_id)
        if result:
            return result.to_dict()

    # Return status if not complete
    return EvaluationStatusResponse(**evaluation)


@app.get(
    "/evaluate/{evaluation_id}/status",
    response_model=EvaluationStatusResponse,
    tags=["Evaluations"],
)
async def get_evaluation_status(evaluation_id: str):
    """
    Get evaluation status and progress.

    Returns current progress without waiting for completion.
    """
    evaluation = store.get(evaluation_id)

    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    # Calculate estimated time remaining
    estimated_time = None
    if evaluation["status"] == "running" and evaluation["completed_tests"] > 0:
        elapsed = datetime.utcnow() - datetime.fromisoformat(evaluation["start_time"])
        rate = evaluation["completed_tests"] / elapsed.total_seconds()
        remaining = evaluation["total_tests"] - evaluation["completed_tests"]
        if rate > 0:
            estimated_time = remaining / rate

    return EvaluationStatusResponse(
        **evaluation,
        estimated_time_remaining=estimated_time,
    )


@app.get(
    "/evaluations",
    tags=["Evaluations"],
)
async def list_evaluations(
    limit: int = Query(50, ge=1, le=100),
    status: Optional[str] = Query(None, description="Filter by status"),
):
    """
    List all evaluations.

    Returns a list of all evaluations with optional status filter.
    """
    evaluations = store.list_all()

    # Filter by status
    if status:
        evaluations = [e for e in evaluations if e["status"] == status]

    # Sort by created_at (newest first)
    evaluations.sort(key=lambda x: x["created_at"], reverse=True)

    # Limit
    evaluations = evaluations[:limit]

    return {
        "count": len(evaluations),
        "evaluations": evaluations,
    }


@app.post(
    "/compare",
    tags=["Evaluations"],
)
async def compare_evaluations(request: ComparisonRequest):
    """
    Compare multiple evaluations.

    Generates statistical comparison across multiple evaluation runs.
    """
    # Get results
    results = []
    for eval_id in request.evaluation_ids:
        result = store.get_result(eval_id)
        if result:
            results.append(result)
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Evaluation {eval_id} not found or not completed",
            )

    if not results:
        raise HTTPException(status_code=400, detail="No valid evaluations to compare")

    # Generate comparison
    comparison = report_generator.compare_models(results)

    # Generate rankings
    if len(results) == 1:
        rankings = report_generator.calculate_rankings(results[0])
        comparison["rankings"] = rankings

    return {
        "comparison": comparison,
        "evaluation_count": len(results),
        "datasets": list(set(r.config.dataset for r in results)),
    }


# ============================================================================
# Dataset Endpoints
# ============================================================================

@app.get("/datasets", tags=["Datasets"])
async def list_datasets():
    """
    List all available datasets.

    Returns dataset names, versions, and metadata.
    """
    try:
        datasets = dataset_manager.list_datasets(include_details=True)
        return {
            "count": len(datasets),
            "datasets": datasets,
        }
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/datasets",
    status_code=status.HTTP_201_CREATED,
    tags=["Datasets"],
)
async def upload_dataset(
    request: DatasetUploadRequest,
    file: UploadFile = File(...),
):
    """
    Upload a new dataset.

    Accepts YAML or JSON dataset files.
    """
    try:
        # Read file content
        content = await file.read()

        # Parse based on format
        if request.format.lower() in ["yaml", "yml"]:
            dataset = dataset_manager.create_from_string(
                content.decode("utf-8"),
                format="yaml",
            )
        elif request.format.lower() == "json":
            dataset = dataset_manager.create_from_string(
                content.decode("utf-8"),
                format="json",
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {request.format}",
            )

        # Update metadata
        dataset.name = request.name
        dataset.version = request.version
        dataset.description = request.description

        # Save dataset
        path = dataset_manager.save_dataset(dataset)

        return {
            "message": "Dataset uploaded successfully",
            "name": dataset.name,
            "version": dataset.version,
            "test_cases": dataset.test_case_count,
            "path": str(path),
        }

    except Exception as e:
        logger.error(f"Error uploading dataset: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/datasets/{dataset_name}", tags=["Datasets"])
async def get_dataset(
    dataset_name: str,
    version: str = Query("latest", description="Dataset version"),
):
    """
    Get dataset details.

    Returns dataset information and test cases.
    """
    try:
        dataset = dataset_manager.load_dataset(dataset_name, version=version)

        return {
            "name": dataset.name,
            "version": dataset.version,
            "description": dataset.description,
            "test_cases": [tc.to_dict() for tc in dataset.test_cases],
            "default_metrics": dataset.default_metrics,
            "metadata": dataset.metadata,
            "categories": list(dataset.categories),
            "tags": list(dataset.tags),
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Report Endpoints
# ============================================================================

@app.get(
    "/reports/{evaluation_id}",
    tags=["Reports"],
)
async def get_report(
    evaluation_id: str,
    format: str = Query("html", description="Report format (html or markdown)"),
):
    """
    Generate evaluation report.

    Returns HTML or markdown report for the evaluation.
    """
    result = store.get_result(evaluation_id)

    if not result:
        raise HTTPException(
            status_code=404,
            detail="Evaluation not found or not completed",
        )

    try:
        if format.lower() in ["html", "htm"]:
            content = report_generator.generate_html_report(result)
            return HTMLResponse(content=content)
        elif format.lower() in ["markdown", "md"]:
            content = report_generator.generate_markdown_report(result)
            return Response(content=content, media_type="text/markdown")
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {format}",
            )

    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Metrics Endpoints
# ============================================================================

@app.get("/metrics", tags=["Metrics"])
async def get_metrics():
    """
    List available evaluation metrics.

    Returns information about available metrics.
    """
    from src.evaluation.metrics import METRICS

    metrics_info = {}
    for name, metric_class in METRICS.items():
        # Get docstring
        doc = metric_class.__doc__ or "No description"
        metrics_info[name] = {
            "name": name,
            "description": doc.strip().split("\n")[0] if doc else "",
        }

    return {
        "count": len(metrics_info),
        "metrics": metrics_info,
    }


@app.get("/prometheus", tags=["Monitoring"])
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Alternative metrics path
@app.get("/metrics", include_in_schema=False)
async def metrics_alternate():
    """Alternative metrics endpoint."""
    return await prometheus_metrics()


# ============================================================================
# Utility Endpoints
# ============================================================================

@app.delete("/evaluations/{evaluation_id}", tags=["Evaluations"])
async def delete_evaluation(evaluation_id: str):
    """
    Delete an evaluation and its results.

    Removes evaluation from storage.
    """
    evaluation = store.get(evaluation_id)

    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    # Remove from store
    if evaluation_id in store.evaluations:
        del store.evaluations[evaluation_id]

    if evaluation_id in store.results:
        del store.results[evaluation_id]

    return {"message": "Evaluation deleted successfully"}


@app.get("/info", tags=["Info"])
async def get_info():
    """
    Get system information.

    Returns configuration and system details.
    """
    return {
        "name": "LLMOps-Eval",
        "version": "0.1.0",
        "config": {
            "datasets_dir": str(settings.datasets_dir),
            "results_dir": str(settings.results_dir),
            "api_host": settings.api_host,
            "api_port": settings.api_port,
            "dashboard_port": settings.dashboard_port,
            "enable_metrics": settings.enable_metrics,
            "max_concurrent_evaluations": settings.max_concurrent_evaluations,
        },
        "providers": {
            "openai_configured": bool(settings.openai_api_key),
            "anthropic_configured": bool(settings.anthropic_api_key),
            "cohere_configured": bool(settings.cohere_api_key),
        },
    }


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
