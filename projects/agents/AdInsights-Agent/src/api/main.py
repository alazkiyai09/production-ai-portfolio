"""
FastAPI Application for AdInsights-Agent

REST API for autonomous AdTech campaign analysis with support for
long-running background tasks and streaming responses.
"""

import asyncio
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator
from pathlib import Path

from fastapi import (
    FastAPI,
    HTTPException,
    BackgroundTasks,
    Depends,
    status,
    Query,
    Request,
    Form,
    Body,
)
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import uvicorn

# Shared modules
from shared.security import SensitiveDataFilter, install_security_filter
from shared.rate_limit import limiter, rate_limit_exception_handler, RateLimitExceeded
from shared.auth import (
    get_current_user,
    require_role,
    require_admin,
    create_access_token,
    create_refresh_token,
    verify_token,
    User,
    UserCreate,
    Token,
    TokenData,
    Role,
    InMemoryUserStore,
)
from shared.errors import (
    register_error_handlers,
    AuthenticationError,
    AuthorizationError,
    ValidationError,
    DatabaseError,
    ExternalAPIError,
)
from shared.secrets import get_settings

# Import agent and tools
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.insights_agent import AdInsightsAgent, AdInsightsState
from tools.analysis_tools import (
    fetch_campaign_metrics,
    HEALTHCARE_BENCHMARKS,
)


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class AnalysisRequest(BaseModel):
    """Request model for campaign analysis."""
    campaign_id: str = Field(
        ...,
        description="Campaign ID to analyze",
        examples=["CAMP-001"]
    )
    start_date: str = Field(
        ...,
        description="Start date (YYYY-MM-DD)",
        examples=["2024-01-01"]
    )
    end_date: str = Field(
        ...,
        description="End date (YYYY-MM-DD)",
        examples=["2024-01-31"]
    )
    campaign_type: Optional[str] = Field(
        "healthcare_pharma",
        description="Type of healthcare campaign",
        examples=["healthcare_pharma", "healthcare_hospitals", "healthcare_telehealth", "healthcare_insurance"]
    )
    analysis_types: Optional[List[str]] = Field(
        None,
        description="Specific analyses to run (if None, runs all)",
        examples=[["detect_anomalies", "analyze_trends", "compare_benchmarks"]]
    )
    include_charts: bool = Field(
        True,
        description="Whether to generate charts"
    )
    forecast_periods: int = Field(
        7,
        description="Number of periods to forecast",
        ge=1,
        le=30
    )

    @field_validator('campaign_id')
    @classmethod
    def validate_campaign_id(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError("campaign_id cannot be empty")
        return v.strip().upper()

    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_dates(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")
        return v


class AnalysisResponse(BaseModel):
    """Response model for analysis results."""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status: pending, running, completed, failed")
    campaign_id: str = Field(..., description="Campaign analyzed")
    created_at: str = Field(..., description="Job creation timestamp")
    completed_at: Optional[str] = Field(None, description="Job completion timestamp")
    request: AnalysisRequest = Field(..., description="Original request")

    # Analysis results (populated when complete)
    report: Optional[str] = Field(None, description="Full markdown report")
    insights: List[str] = Field(default_factory=list, description="Generated insights")
    recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations")
    charts: List[str] = Field(default_factory=list, description="Paths to generated charts")

    # Detailed metrics
    metrics_summary: Optional[Dict[str, Any]] = Field(None, description="Summary statistics")
    anomalies: List[Dict[str, Any]] = Field(default_factory=list, description="Detected anomalies")
    trends: Dict[str, Any] = Field(default_factory=dict, description="Trend analysis")
    benchmark_comparison: Optional[Dict[str, Any]] = Field(None, description="Benchmark comparison")
    correlations: List[Dict[str, Any]] = Field(default_factory=list, description="Correlations found")

    # Error handling
    error: Optional[str] = Field(None, description="Error message if failed")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")


class QuickInsightsRequest(BaseModel):
    """Request model for quick insights (simplified analysis)."""
    campaign_id: str = Field(..., description="Campaign ID")
    campaign_type: str = Field("healthcare_pharma", description="Campaign type")
    days: int = Field(7, description="Number of days to analyze", ge=1, le=90)


class QuickInsightsResponse(BaseModel):
    """Response model for quick insights."""
    campaign_id: str = Field(..., description="Campaign analyzed")
    date_range: Dict[str, str] = Field(..., description="Date range analyzed")
    generated_at: str = Field(..., description="Timestamp")

    # Key metrics
    avg_ctr: float = Field(..., description="Average click-through rate")
    avg_cvr: float = Field(..., description="Average conversion rate")
    avg_cpa: float = Field(..., description="Average cost per acquisition")
    avg_roi: float = Field(..., description="Average return on investment")

    # Quick assessment
    performance_rating: str = Field(..., description="Overall performance: excellent, good, average, poor")
    key_insight: str = Field(..., description="Single most important insight")
    top_recommendation: str = Field(..., description="Primary recommendation")


class ComparisonRequest(BaseModel):
    """Request model for comparing multiple campaigns."""
    campaign_ids: List[str] = Field(
        ...,
        min_length=2,
        max_length=10,
        description="Campaign IDs to compare"
    )
    metric: str = Field(
        ...,
        description="Metric to compare",
        examples=["ctr", "cvr", "cpa", "roi"]
    )
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    campaign_type: str = Field("healthcare_pharma", description="Campaign type")


class ComparisonResponse(BaseModel):
    """Response model for campaign comparison."""
    metric: str = Field(..., description="Metric compared")
    campaigns: List[Dict[str, Any]] = Field(..., description="Campaign data")
    winner: str = Field(..., description="Best performing campaign ID")
    insights: List[str] = Field(default_factory=list, description="Comparison insights")


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Job status")
    created_at: str = Field(..., description="Creation timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")
    progress: Optional[float] = Field(None, description="Progress 0-1")
    current_step: Optional[str] = Field(None, description="Current analysis step")
    error: Optional[str] = Field(None, description="Error if failed")


class BenchmarkResponse(BaseModel):
    """Response model for industry benchmarks."""
    industry: str = Field(..., description="Industry segment")
    benchmarks: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Benchmark metrics with median, p25, p75"
    )
    description: str = Field(..., description="Industry description")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Health status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")
    components: Dict[str, str] = Field(..., description="Component status")
    uptime_seconds: float = Field(..., description="Server uptime")


# =============================================================================
# JOB STORE
# =============================================================================

class JobStore:
    """In-memory store for background analysis jobs."""

    def __init__(self):
        self.jobs: Dict[str, AnalysisResponse] = {}
        self.job_status: Dict[str, str] = {}

    def create_job(self, request: AnalysisRequest) -> str:
        """Create a new analysis job."""
        job_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        job = AnalysisResponse(
            job_id=job_id,
            status="pending",
            campaign_id=request.campaign_id,
            created_at=now,
            request=request,
        )

        self.jobs[job_id] = job
        self.job_status[job_id] = "pending"

        return job_id

    def get_job(self, job_id: str) -> Optional[AnalysisResponse]:
        """Get job by ID."""
        return self.jobs.get(job_id)

    def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        current_step: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        """Update job status and results."""
        job = self.jobs.get(job_id)
        if not job:
            return

        if status:
            job.status = status
            self.job_status[job_id] = status

        if status == "completed" and not job.completed_at:
            job.completed_at = datetime.now().isoformat()

        if result:
            job.report = result.get("report")
            job.insights = result.get("insights", [])
            job.recommendations = result.get("recommendations", [])
            job.charts = result.get("charts", [])
            job.metrics_summary = result.get("metrics_summary")
            job.anomalies = result.get("anomalies", [])
            job.trends = result.get("trends", {})
            job.benchmark_comparison = result.get("benchmark_comparison")
            job.correlations = result.get("correlations", [])

        if error:
            job.error = error

    def list_jobs(self) -> List[Dict[str, str]]:
        """List all jobs with basic info."""
        return [
            {
                "job_id": job_id,
                "status": job.status,
                "campaign_id": job.campaign_id,
                "created_at": job.created_at,
            }
            for job_id, job in self.jobs.items()
        ]


# Global job store
job_store = JobStore()


# =============================================================================
# API APPLICATION
# =============================================================================

# Create FastAPI app
app = FastAPI(
    title="AdInsights-Agent API",
    description="Autonomous analytics agent for healthcare AdTech campaign analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Install security filter for logs
install_security_filter()

# Register error handlers
register_error_handlers(app)

# Register rate limit exception handler
app.add_exception_handler(RateLimitExceeded, rate_limit_exception_handler)

# Get settings
settings = get_settings()

# Track start time for uptime
start_time = datetime.now()

# Agent instance (lazy loaded)
_agent_instance = None


def get_agent() -> AdInsightsAgent:
    """Get or create agent instance."""
    global _agent_instance

    if _agent_instance is None:
        _agent_instance = AdInsightsAgent(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=0.0,
        )

    return _agent_instance


# =============================================================================
# BACKGROUND TASKS
# =============================================================================

async def run_analysis_job(
    job_id: str,
    request: AnalysisRequest,
    agent: AdInsightsAgent,
):
    """
    Run analysis as a background task.

    Updates job store with progress and results.
    """
    try:
        # Update status to running
        job_store.update_job(job_id, status="running", current_step="Initializing")

        # Build natural language request from parameters
        analysis_desc = " and ".join(request.analysis_types or ["comprehensive analysis"])
        nl_request = f"Analyze campaign {request.campaign_id} for {analysis_desc}"

        # Run analysis
        result = agent.analyze(
            request=nl_request,
            campaign_id=request.campaign_id,
            date_range=(request.start_date, request.end_date),
        )

        if result.get("success"):
            # Update with results
            job_store.update_job(job_id, status="completed", result=result)
        else:
            # Update with error
            job_store.update_job(
                job_id,
                status="failed",
                error=result.get("error", "Unknown error")
            )

    except Exception as e:
        # Update with error
        job_store.update_job(
            job_id,
            status="failed",
            error=f"Analysis failed: {str(e)}"
        )


async def run_analysis_stream(
    request: AnalysisRequest,
    agent: AdInsightsAgent,
) -> AsyncGenerator[str, None]:
    """
    Run analysis with streaming progress updates.

    Yields SSE-formatted progress messages.
    """
    job_id = str(uuid.uuid4())

    try:
        # Send start event
        yield _sse_event("start", {"job_id": job_id, "status": "initializing"})

        # Build request
        nl_request = f"Analyze campaign {request.campaign_id}"

        # Send step updates
        steps = [
            "Parsing request",
            "Planning analysis",
            "Gathering data",
            "Analyzing metrics",
            "Detecting anomalies",
            "Analyzing trends",
            "Generating insights",
            "Creating report",
        ]

        for i, step in enumerate(steps):
            progress = (i + 1) / len(steps)
            yield _sse_event("progress", {
                "step": step,
                "progress": round(progress, 2),
                "current": i + 1,
                "total": len(steps)
            })
            await asyncio.sleep(0.5)  # Simulate processing

        # Run actual analysis
        result = agent.analyze(
            request=nl_request,
            campaign_id=request.campaign_id,
            date_range=(request.start_date, request.end_date),
        )

        if result.get("success"):
            # Send complete event with results
            yield _sse_event("complete", {
                "job_id": job_id,
                "report": result.get("report", ""),
                "insights": result.get("insights", []),
                "recommendations": result.get("recommendations", []),
                "charts": result.get("charts", []),
            })
        else:
            yield _sse_event("error", {
                "job_id": job_id,
                "error": result.get("error", "Unknown error")
            })

        # Send done event
        yield _sse_event("done", {"job_id": job_id})

    except Exception as e:
        yield _sse_event("error", {
            "job_id": job_id,
            "error": f"Stream failed: {str(e)}"
        })


def _sse_event(event: str, data: Dict[str, Any]) -> str:
    """Format data as Server-Sent Event."""
    import json
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# =============================================================================
# ENDPOINTS
# =============================================================================

# In-memory user store for demo (replace with database in production)
user_store = InMemoryUserStore()

# ============================================================================
# Authentication Endpoints
# ============================================================================

@app.post("/auth/register", tags=["Authentication"])
@limiter.limit("10/hour")
async def register(
    user_data: UserCreate,
    request: Request,
):
    """Register a new user."""
    try:
        user = await user_store.create_user(user_data)
        return {
            "message": "User registered successfully",
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@app.post("/auth/login", tags=["Authentication"])
@limiter.limit("20/minute")
async def login(
    username: str = Form(...),
    password: str = Form(...),
    request: Request = None,
):
    """Login and receive access token."""
    from shared.auth import authenticate_user, login_user

    user = await authenticate_user(username, password, user_store)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_data = await login_user(username, password, user_store)
    return token_data


@app.post("/auth/refresh", tags=["Authentication"])
@limiter.limit("30/minute")
async def refresh(
    refresh_token: str = Body(..., embed=True),
    request: Request = None,
):
    """Refresh access token."""
    from shared.auth import refresh_user_token

    token_data = await refresh_user_token(refresh_token, user_store)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token_data


@app.get("/auth/me", tags=["Authentication"])
@limiter.limit("60/minute")
async def get_current_user_info(current_user: TokenData = Depends(get_current_user)):
    """Get current user information."""
    user = await user_store.get_user_by_id(current_user.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role": user.role.value,
        "is_active": user.is_active,
    }


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AdInsights-Agent API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns API status, component health, and uptime.
    """
    uptime = (datetime.now() - start_time).total_seconds()

    # Check component health
    components = {
        "api": "healthy",
        "agent": "ready",
        "job_store": "healthy" if job_store else "unhealthy",
    }

    # Check if agent can be initialized
    try:
        get_agent()
        components["agent"] = "healthy"
    except Exception as e:
        components["agent"] = f"unhealthy: {str(e)}"

    overall_status = "healthy" if all("healthy" in v for v in components.values()) else "degraded"

    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        components=components,
        uptime_seconds=round(uptime, 2),
    )


@app.post("/analyze", response_model=AnalysisResponse, status_code=status.HTTP_202_ACCEPTED)
@limiter.limit("10/minute")
async def analyze_campaign(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    http_request: Request = None,
):
    """
    Submit a campaign analysis job.

    Runs comprehensive analysis in the background. Use GET /analyze/{job_id}
    to retrieve results when complete.

    Analysis includes:
    - Metrics summary (CTR, CVR, CPA, ROI)
    - Anomaly detection
    - Trend analysis
    - Benchmark comparison
    - Correlation analysis
    - Insight generation
    - Chart creation

    Returns immediately with job_id for tracking.
    """
    # Create job
    job_id = job_store.create_job(request)

    # Get agent
    agent = get_agent()

    # Add background task
    background_tasks.add_task(
        run_analysis_job,
        job_id,
        request,
        agent,
    )

    # Return job info
    job = job_store.get_job(job_id)
    return job


@app.post("/analyze/stream")
async def analyze_campaign_stream(request: AnalysisRequest):
    """
    Stream analysis results in real-time using Server-Sent Events.

    Returns a stream of progress updates and final results.
    Client should handle SSE events: start, progress, complete/error, done.

    Event types:
    - start: Analysis started
    - progress: Progress update with step and percentage
    - complete: Analysis complete with results
    - error: Analysis failed
    - done: Stream finished
    """
    agent = get_agent()

    return StreamingResponse(
        run_analysis_stream(request, agent),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/analyze/{job_id}", response_model=AnalysisResponse)
async def get_analysis_result(job_id: str):
    """
    Get the result of an analysis job.

    Returns the current state of the job including results if completed.
    """
    job = job_store.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    return job


@app.get("/analyze/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the status of an analysis job.

    Returns current status and progress without full results.
    """
    job = job_store.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    # Calculate progress based on status
    progress_map = {
        "pending": 0.0,
        "running": 0.5,
        "completed": 1.0,
        "failed": 0.0,
    }

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        completed_at=job.completed_at,
        progress=progress_map.get(job.status, 0.0),
        current_step=job.request.campaign_id,
        error=job.error,
    )


@app.post("/quick-insights", response_model=QuickInsightsResponse)
async def get_quick_insights(request: QuickInsightsRequest):
    """
    Get quick insights for a campaign (simplified analysis).

    Returns key metrics and a quick assessment for the last N days.
    Faster than full analysis but less detailed.
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.days)

        # Fetch data
        data = fetch_campaign_metrics.invoke({
            "campaign_id": request.campaign_id,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "campaign_type": request.campaign_type,
        })

        if "error" in data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=data["error"]
            )

        summary = data["summary"]

        # Calculate performance rating
        ctr = summary.get("avg_ctr", 0)
        cvr = summary.get("avg_cvr", 0)
        roi = summary.get("avg_roi", 0)

        if ctr > 1.5 and cvr > 3.0 and roi > 3.0:
            rating = "excellent"
        elif ctr > 1.0 and cvr > 2.0 and roi > 2.0:
            rating = "good"
        elif ctr > 0.5 and cvr > 1.0:
            rating = "average"
        else:
            rating = "poor"

        # Generate key insight
        if roi > 3.0:
            key_insight = f"Campaign is performing well with strong ROI of {roi:.2f}x"
        elif ctr < 1.0:
            key_insight = f"Low CTR ({ctr:.2f}%) indicates ad creative or targeting issues"
        elif cvr < 2.0:
            key_insight = f"Low CVR ({cvr:.2f}%) suggests landing page or offer optimization needed"
        else:
            key_insight = "Campaign performance is within normal ranges"

        # Generate recommendation
        if rating == "poor":
            top_recommendation = "Review and optimize ad creative, targeting, and landing pages"
        elif rating == "average":
            top_recommendation = "Consider A/B testing creatives to improve performance"
        else:
            top_recommendation = "Scale budget gradually while monitoring performance"

        return QuickInsightsResponse(
            campaign_id=request.campaign_id,
            date_range={
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d")
            },
            generated_at=datetime.now().isoformat(),
            avg_ctr=round(summary.get("avg_ctr", 0), 2),
            avg_cvr=round(summary.get("avg_cvr", 0), 2),
            avg_cpa=round(summary.get("avg_cpa", 0), 2),
            avg_roi=round(summary.get("avg_roi", 0), 2),
            performance_rating=rating,
            key_insight=key_insight,
            top_recommendation=top_recommendation,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quick insights failed: {str(e)}"
        )


@app.get("/benchmarks/{industry}", response_model=BenchmarkResponse)
async def get_benchmarks(industry: str):
    """
    Get industry benchmark metrics.

    Returns median, 25th percentile, and 75th percentile values
    for key metrics in the specified industry segment.
    """
    # Validate industry
    available_industries = list(HEALTHCARE_BENCHMARKS.keys())

    if industry not in available_industries:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Industry '{industry}' not found. Available: {', '.join(available_industries)}"
        )

    benchmarks = HEALTHCARE_BENCHMARKS[industry]

    # Generate description
    descriptions = {
        "healthcare_pharma": "Pharmaceutical advertising campaigns",
        "healthcare_hospitals": "Hospital and healthcare system advertising",
        "healthcare_telehealth": "Telehealth and virtual care campaigns",
        "healthcare_insurance": "Health insurance and managed care campaigns",
    }

    return BenchmarkResponse(
        industry=industry,
        benchmarks=benchmarks,
        description=descriptions.get(industry, "Healthcare advertising benchmarks")
    )


@app.get("/benchmarks", response_model=List[str])
async def list_benchmark_industries():
    """
    List available benchmark industries.

    Returns all available industry segments for benchmark comparison.
    """
    return list(HEALTHCARE_BENCHMARKS.keys())


@app.post("/compare", response_model=ComparisonResponse)
async def compare_campaigns(request: ComparisonRequest):
    """
    Compare multiple campaigns on a specific metric.

    Returns ranking, comparison data, and insights.
    """
    try:
        # Fetch data for all campaigns
        campaigns_data = []

        for campaign_id in request.campaign_ids:
            data = fetch_campaign_metrics.invoke({
                "campaign_id": campaign_id,
                "start_date": request.start_date,
                "end_date": request.end_date,
                "campaign_type": request.campaign_type,
            })

            if "error" not in data:
                campaigns_data.append({
                    "campaign_id": campaign_id,
                    "metric_value": data["summary"].get(f"avg_{request.metric}", 0),
                    **data["summary"]
                })

        if not campaigns_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No valid campaigns found"
            )

        # Sort by metric (handle CPA specially - lower is better)
        reverse_order = request.metric != "cpa"
        campaigns_data.sort(key=lambda x: x["metric_value"], reverse=reverse_order)

        winner = campaigns_data[0]["campaign_id"]

        # Generate insights
        insights = []

        if len(campaigns_data) >= 2:
            # Calculate spread
            values = [c["metric_value"] for c in campaigns_data]
            spread = max(values) - min(values)
            avg = sum(values) / len(values)

            insights.append(f"Average {request.metric.upper()} across campaigns: {avg:.2f}")
            insights.append(f"Spread between best and worst: {spread:.2f}")

            # Highlight winner
            winner_value = campaigns_data[0]["metric_value"]
            insights.append(
                f"Winner: {winner} with {request.metric.upper()} of {winner_value:.2f}"
            )

            # Check for significant differences
            if spread > avg * 0.5:  # More than 50% variation
                insights.append(
                    f"⚠️ High variation detected. Investigate why performance differs significantly."
                )
            else:
                insights.append(
                    "✓ Campaigns show consistent performance across the group."
                )

        return ComparisonResponse(
            metric=request.metric,
            campaigns=campaigns_data,
            winner=winner,
            insights=insights,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison failed: {str(e)}"
        )


@app.get("/jobs", response_model=List[Dict[str, str]])
async def list_jobs(
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None, description="Filter by status"),
):
    """
    List all analysis jobs.

    Returns paginated list of jobs with optional status filter.
    """
    jobs = job_store.list_jobs()

    # Filter by status if specified
    if status:
        jobs = [j for j in jobs if j["status"] == status]

    # Sort by created_at descending
    jobs.sort(key=lambda x: x["created_at"], reverse=True)

    # Limit results
    jobs = jobs[:limit]

    return jobs


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a completed job and free resources.

    Only allows deletion of completed or failed jobs.
    """
    job = job_store.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    if job.status == "running":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete running job. Wait for completion."
        )

    # Delete from store
    del job_store.jobs[job_id]
    del job_store.job_status[job_id]

    return {"message": f"Job {job_id} deleted"}


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": str(exc)},
    )


@app.exception_handler(KeyError)
async def key_error_handler(request, exc):
    """Handle KeyError exceptions."""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": f"Resource not found: {str(exc)}"},
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the API server."""
    import argparse

    parser = argparse.ArgumentParser(description="AdInsights-Agent API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
