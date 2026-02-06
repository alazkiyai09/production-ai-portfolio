"""
FastAPI application for FraudTriage-Agent.

This module provides REST API endpoints for submitting fraud alerts,
checking status, and providing human review decisions.

Endpoints:
    - POST /triage              - Submit a fraud alert for analysis
    - GET  /triage/{alert_id}   - Get status and results of an alert
    - POST /triage/{alert_id}/approve - Human approval for escalated alerts
    - GET  /health              - Health check

The API integrates with the FraudTriageAgent for async processing
and stores results in-memory for demo purposes.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, status, BackgroundTasks, WebSocket, WebSocketDisconnect, Request, Depends, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

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

from src.agents.fraud_triage_agent import FraudTriageAgent, create_llm
from src.models.state import AlertDecision, AlertType, RiskLevel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class APIConfig:
    """API configuration settings."""

    # API settings
    API_TITLE = "FraudTriage-Agent API"
    API_VERSION = "0.1.0"
    API_DESCRIPTION = "LangGraph-based fraud alert triage system"

    # CORS settings
    ALLOW_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8080",
    ]
    ALLOW_CREDENTIALS = True
    ALLOW_METHODS = ["*"]
    ALLOW_HEADERS = ["*"]

    # Agent settings
    AGENT_ENVIRONMENT = "development"  # Can be overridden via ENVIRONMENT env var

    # Storage
    ALERT_STORE_MAX_SIZE = 1000  # Max alerts to keep in memory


# =============================================================================
# In-Memory Storage (Demo Only)
# =============================================================================

class AlertStore:
    """
    In-memory storage for alert states and results.

    Note: This is for demo purposes. In production, use a proper database
    like PostgreSQL, MongoDB, or Redis.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize the alert store.

        Args:
            max_size: Maximum number of alerts to store
        """
        self._alerts: dict[str, dict[str, Any]] = {}
        self._max_size = max_size
        self._lock = asyncio.Lock()

    async def get(self, alert_id: str) -> dict[str, Any] | None:
        """Get alert by ID."""
        async with self._lock:
            return self._alerts.get(alert_id)

    async def set(self, alert_id: str, data: dict[str, Any]) -> None:
        """Store alert data."""
        async with self._lock:
            # Evict oldest if at capacity
            if len(self._alerts) >= self._max_size and alert_id not in self._alerts:
                oldest = next(iter(self._alerts))
                del self._alerts[oldest]
                logger.warning(f"Alert store full, evicted oldest: {oldest}")

            self._alerts[alert_id] = data

    async def update(self, alert_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Update existing alert."""
        async with self._lock:
            if alert_id not in self._alerts:
                return None

            self._alerts[alert_id].update(updates)
            return self._alerts[alert_id]

    async def list_all(self) -> list[dict[str, Any]]:
        """List all alerts."""
        async with self._lock:
            return list(self._alerts.values())

    async def delete(self, alert_id: str) -> bool:
        """Delete alert by ID."""
        async with self._lock:
            if alert_id in self._alerts:
                del self._alerts[alert_id]
                return True
            return False


# Global alert store instance
alert_store = AlertStore(max_size=APIConfig.ALERT_STORE_MAX_SIZE)

# Global agent instance
fraud_agent = FraudTriageAgent(environment=APIConfig.AGENT_ENVIRONMENT)


# =============================================================================
# Request/Response Models
# =============================================================================

class TriageRequest(BaseModel):
    """
    Request model for submitting a fraud alert for triage.

    Attributes:
        alert_id: Unique alert identifier
        alert_type: Type of fraud alert
        transaction_amount: Amount of the flagged transaction
        customer_id: Customer identifier
        transaction_country: Transaction country code (optional)
        transaction_device_id: Device identifier (optional)
        merchant_name: Merchant name (optional)
        alert_reason: Reason for the alert (optional)
    """

    alert_id: str = Field(..., description="Unique alert identifier", min_length=1)
    alert_type: str = Field(..., description="Type of fraud alert (e.g., account_takeover, unusual_amount, location_mismatch)")
    transaction_amount: float = Field(..., description="Amount of the flagged transaction", gt=0)
    customer_id: str = Field(..., description="Customer identifier", min_length=1)
    transaction_country: str | None = Field(None, description="Transaction country code (ISO 3166-1 alpha-2)")
    transaction_device_id: str | None = Field(None, description="Device identifier")
    merchant_name: str | None = Field(None, description="Merchant name")
    alert_reason: str | None = Field(None, description="Reason for the alert")

    @field_validator("alert_type")
    @classmethod
    def validate_alert_type(cls, v: str) -> str:
        """Validate alert type is a known value."""
        valid_types = [
            "unusual_amount",
            "velocity",
            "location_mismatch",
            "device_change",
            "account_takeover",
        ]
        v_lower = v.lower().replace("-", "_")
        if v_lower not in valid_types:
            raise ValueError(
                f"Invalid alert_type: {v}. Must be one of: {', '.join(valid_types)}"
            )
        return v_lower

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "alert_id": "ALERT-2025-001234",
                "alert_type": "account_takeover",
                "transaction_amount": 7500.00,
                "customer_id": "CUST-004",
                "transaction_country": "NG",
                "transaction_device_id": "DEVICE-NEW-999",
                "merchant_name": "Luxury Electronics Store",
                "alert_reason": "Transaction from high-risk country with new device",
            }
        }


class TriageResponse(BaseModel):
    """
    Response model for fraud alert triage results.

    Attributes:
        alert_id: Alert identifier
        status: Processing status (pending, processing, completed, error)
        decision: Final decision on the alert
        risk_score: Calculated risk score (0.0-1.0)
        risk_level: Risk level category
        risk_factors: List of identified risk factors
        recommendation: Action recommendation
        requires_human_review: Whether human review is needed
        processing_time_ms: Processing duration in milliseconds
        created_at: Alert creation timestamp
        completed_at: Alert completion timestamp (if complete)
        error: Error message (if error occurred)
    """

    alert_id: str
    status: str = Field(..., description="Processing status: pending, processing, completed, error")
    decision: str | None = Field(None, description="Final decision: auto_close, review_required, escalate")
    risk_score: float | None = Field(None, ge=0, le=1, description="Risk score from 0.0 to 1.0")
    risk_level: str | None = Field(None, description="Risk level: low, medium, high, critical")
    risk_factors: list[str] = Field(default_factory=list, description="Identified risk factors")
    recommendation: str | None = Field(None, description="Action recommendation")
    requires_human_review: bool = Field(default=False, description="Whether human review is needed")
    processing_time_ms: int | None = Field(None, description="Processing duration in milliseconds")
    created_at: datetime
    completed_at: datetime | None = None
    error: str | None = Field(None, description="Error message if status is error")

    class Config:
        """Pydantic configuration."""
        json_encoders = {datetime: lambda v: v.isoformat()}
        json_schema_extra = {
            "example": {
                "alert_id": "ALERT-2025-001234",
                "status": "completed",
                "decision": "escalate",
                "risk_score": 0.875,
                "risk_level": "critical",
                "risk_factors": [
                    "Very new account (2 months old)",
                    "Transaction amount 100x higher than average",
                    "Customer KYC not verified",
                ],
                "recommendation": "Alert escalated due to high risk score. Immediate review required.",
                "requires_human_review": True,
                "processing_time_ms": 2340,
                "created_at": "2025-01-30T14:25:00Z",
                "completed_at": "2025-01-30T14:25:02Z",
            }
        }


class ApprovalRequest(BaseModel):
    """
    Request model for human approval/review of escalated alerts.

    Attributes:
        reviewer_id: Reviewer identifier
        reviewer_name: Reviewer name
        decision: Review decision (approve, reject, escalate)
        reasoning: Reasoning for the decision
        tags: Optional tags for categorization
    """

    reviewer_id: str = Field(..., description="Reviewer identifier", min_length=1)
    reviewer_name: str = Field(..., description="Reviewer name", min_length=1)
    decision: str = Field(..., description="Review decision: approve (legitimate), reject (fraud), escalate")
    reasoning: str = Field(..., description="Reasoning for the decision", min_length=10)
    tags: list[str] = Field(default_factory=list, description="Optional tags for categorization")

    @field_validator("decision")
    @classmethod
    def validate_decision(cls, v: str) -> str:
        """Validate decision is a known value."""
        valid_decisions = ["approve", "reject", "escalate"]
        v_lower = v.lower()
        if v_lower not in valid_decisions:
            raise ValueError(
                f"Invalid decision: {v}. Must be one of: {', '.join(valid_decisions)}"
            )
        return v_lower

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "reviewer_id": "ANALYST-001",
                "reviewer_name": "Jane Smith",
                "decision": "reject",
                "reasoning": "Customer confirmed they did not authorize this transaction. Nigeria location and new device are confirmed fraud indicators.",
                "tags": ["confirmed_fraud", "account_takeover", "international"],
            }
        }


class ApprovalResponse(BaseModel):
    """
    Response model for human approval/review.

    Attributes:
        alert_id: Alert identifier
        reviewer_id: Reviewer identifier
        reviewer_name: Reviewer name
        decision: Review decision
        reasoning: Reviewer's reasoning
        reviewed_at: Review timestamp
        updated_decision: Updated alert decision after review
    """

    alert_id: str
    reviewer_id: str
    reviewer_name: str
    decision: str
    reasoning: str
    reviewed_at: datetime = Field(default_factory=datetime.utcnow)
    updated_decision: str | None = Field(None, description="Updated alert decision after review")
    tags: list[str] = Field(default_factory=list)

    class Config:
        """Pydantic configuration."""
        json_encoders = {datetime: lambda v: v.isoformat()}


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    agent_environment: str = Field(..., description="Agent environment (development/demo/production)")
    alerts_processed: int = Field(..., description="Number of alerts processed")
    uptime_seconds: float | None = Field(None, description="Service uptime in seconds")


# =============================================================================
# FastAPI Application
# =============================================================================

# Create FastAPI app
app = FastAPI(
    title=APIConfig.API_TITLE,
    description=APIConfig.API_DESCRIPTION,
    version=APIConfig.API_VERSION,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=APIConfig.ALLOW_ORIGINS,
    allow_credentials=APIConfig.ALLOW_CREDENTIALS,
    allow_methods=APIConfig.ALLOW_METHODS,
    allow_headers=APIConfig.ALLOW_HEADERS,
)

# Install security filter for logs
install_security_filter()

# Register error handlers
register_error_handlers(app)

# Register rate limit exception handler
app.add_exception_handler(RateLimitExceeded, rate_limit_exception_handler)

# Get settings
settings = get_settings()

# Track startup time
_app_start_time = datetime.utcnow()


# =============================================================================
# Background Task
# =============================================================================

async def process_alert_async(
    alert_id: str,
    alert_type: str,
    transaction_amount: float,
    customer_id: str,
    **kwargs: Any,
) -> None:
    """
    Background task to process alert asynchronously.

    Args:
        alert_id: Alert identifier
        alert_type: Type of fraud alert
        transaction_amount: Transaction amount
        customer_id: Customer identifier
        **kwargs: Additional alert data
    """
    logger.info(f"[{alert_id}] 🚀 Background processing started")

    try:
        # Update status to processing
        await alert_store.update(alert_id, {"status": "processing"})

        # Run the fraud triage agent
        result = await fraud_agent.arun(
            alert_id=alert_id,
            alert_type=alert_type,
            transaction_amount=transaction_amount,
            customer_id=customer_id,
            **kwargs,
        )

        # Extract results
        decision = result.get("decision")
        risk_score = result.get("risk_score")
        risk_level = result.get("risk_level")
        risk_factors = result.get("risk_factors", [])
        recommendation = result.get("recommendation")
        requires_human_review = result.get("requires_human_review", False)
        processing_time_ms = result.get("processing_duration_ms")
        error_message = result.get("error_message")

        # Update alert store with results
        await alert_store.update(
            alert_id,
            {
                "status": "completed" if not error_message else "error",
                "decision": decision.value if decision else None,
                "risk_score": risk_score,
                "risk_level": risk_level.value if risk_level else None,
                "risk_factors": risk_factors,
                "recommendation": recommendation,
                "requires_human_review": requires_human_review,
                "processing_time_ms": processing_time_ms,
                "completed_at": datetime.utcnow(),
                "error": error_message,
                "full_result": result,  # Store full result for debugging
            },
        )

        logger.info(
            f"[{alert_id}] ✅ Background processing complete - "
            f"Decision: {decision}, Risk: {risk_score:.3f}"
        )

    except Exception as e:
        logger.error(f"[{alert_id}] ❌ Background processing failed: {e}")
        await alert_store.update(
            alert_id,
            {
                "status": "error",
                "error": str(e),
                "completed_at": datetime.utcnow(),
            },
        )


# =============================================================================
# WebSocket Connection Manager (Optional)
# =============================================================================

class ConnectionManager:
    """
    WebSocket connection manager for real-time updates.

    Note: This is optional and can be enabled for real-time alert updates.
    """

    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, alert_id: str) -> None:
        """Connect a WebSocket client to an alert."""
        await websocket.accept()
        if alert_id not in self.active_connections:
            self.active_connections[alert_id] = []
        self.active_connections[alert_id].append(websocket)
        logger.info(f"[{alert_id}] WebSocket client connected")

    def disconnect(self, websocket: WebSocket, alert_id: str) -> None:
        """Disconnect a WebSocket client."""
        if alert_id in self.active_connections:
            self.active_connections[alert_id].remove(websocket)
            if not self.active_connections[alert_id]:
                del self.active_connections[alert_id]
            logger.info(f"[{alert_id}] WebSocket client disconnected")

    async def send_update(self, alert_id: str, message: dict[str, Any]) -> None:
        """Send update to all connected clients for an alert."""
        if alert_id in self.active_connections:
            for connection in self.active_connections[alert_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"[{alert_id}] Error sending WebSocket update: {e}")


# Global connection manager
# manager = ConnectionManager()  # Uncomment to enable WebSocket support


# =============================================================================
# Endpoints
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
    refresh_token: str = Form(...),
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


@app.get("/", response_model=dict)
async def root() -> dict:
    """
    Root endpoint with API information.

    Returns:
        API metadata and available endpoints
    """
    return {
        "name": APIConfig.API_TITLE,
        "version": APIConfig.API_VERSION,
        "description": APIConfig.API_DESCRIPTION,
        "endpoints": {
            "triage": "POST /triage - Submit a fraud alert for analysis",
            "get_status": "GET /triage/{alert_id} - Get alert status and results",
            "approve": "POST /triage/{alert_id}/approve - Submit human review",
            "health": "GET /health - Health check",
            "docs": "/docs - Interactive API documentation (Swagger UI)",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        Service health status and metrics
    """
    # Calculate uptime
    uptime = (datetime.utcnow() - _app_start_time).total_seconds()

    # Count processed alerts
    all_alerts = await alert_store.list_all()
    completed = [a for a in all_alerts if a.get("status") == "completed"]

    return HealthResponse(
        status="healthy",
        version=APIConfig.API_VERSION,
        agent_environment=APIConfig.AGENT_ENVIRONMENT,
        alerts_processed=len(completed),
        uptime_seconds=uptime,
    )


@app.post("/triage", response_model=TriageResponse, status_code=status.HTTP_202_ACCEPTED)
@limiter.limit("30/minute")
async def submit_alert(
    request: TriageRequest,
    background_tasks: BackgroundTasks,
    http_request: Request = None,
) -> TriageResponse:
    """
    Submit a fraud alert for triage analysis.

    This endpoint accepts fraud alerts and processes them asynchronously.
    Returns immediately with a 202 Accepted response.

    Args:
        request: Fraud alert submission request
        background_tasks: FastAPI background tasks

    Returns:
        TriageResponse with initial status (pending/processing)

    Raises:
        HTTPException: If alert_id already exists
    """
    alert_id = request.alert_id

    logger.info(
        f"Received alert submission: {alert_id} - "
        f"Type: {request.alert_type}, Amount: ${request.transaction_amount:.2f}"
    )

    # Check if alert already exists
    existing = await alert_store.get(alert_id)
    if existing:
        logger.warning(f"Alert {alert_id} already exists")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Alert {alert_id} already exists. Use GET /triage/{alert_id} to check status."
        )

    # Create initial alert record
    created_at = datetime.utcnow()
    alert_data = {
        "alert_id": alert_id,
        "status": "pending",
        "created_at": created_at,
        "decision": None,
        "risk_score": None,
        "risk_level": None,
        "risk_factors": [],
        "recommendation": None,
        "requires_human_review": False,
        "processing_time_ms": None,
        "completed_at": None,
        "error": None,
    }

    # Store alert
    await alert_store.set(alert_id, alert_data)

    # Start background processing
    background_tasks.add_task(
        process_alert_async,
        alert_id=alert_id,
        alert_type=request.alert_type,
        transaction_amount=request.transaction_amount,
        customer_id=request.customer_id,
        transaction_country=request.transaction_country,
        transaction_device_id=request.transaction_device_id,
        merchant_name=request.merchant_name,
        alert_reason=request.alert_reason,
    )

    logger.info(f"Alert {alert_id} accepted for processing")

    # Return initial response
    return TriageResponse(**alert_data)


@app.get("/triage/{alert_id}", response_model=TriageResponse)
async def get_alert_status(alert_id: str) -> TriageResponse:
    """
    Get the status and results of a fraud alert.

    Args:
        alert_id: Alert identifier

    Returns:
        TriageResponse with current status and results (if complete)

    Raises:
        HTTPException: If alert not found
    """
    logger.debug(f"Status check for alert: {alert_id}")

    alert_data = await alert_store.get(alert_id)

    if not alert_data:
        logger.warning(f"Alert not found: {alert_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert {alert_id} not found"
        )

    return TriageResponse(**alert_data)


@app.post("/triage/{alert_id}/approve", response_model=ApprovalResponse)
async def approve_alert(
    alert_id: str,
    request: ApprovalRequest,
) -> ApprovalResponse:
    """
    Submit human approval/review for an escalated alert.

    This endpoint allows fraud analysts to provide their decision
    on alerts that require human review.

    Args:
        alert_id: Alert identifier
        request: Human review request

    Returns:
        ApprovalResponse with review details

    Raises:
        HTTPException: If alert not found or doesn't require review
    """
    logger.info(
        f"Human review submitted for alert {alert_id}: "
        f"{request.reviewer_name} - {request.decision}"
    )

    # Get alert
    alert_data = await alert_store.get(alert_id)

    if not alert_data:
        logger.warning(f"Alert not found: {alert_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert {alert_id} not found"
        )

    # Check if alert requires review
    if not alert_data.get("requires_human_review", False):
        logger.warning(f"Alert {alert_id} does not require human review")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Alert {alert_id} does not require human review. "
                   f"Current status: {alert_data.get('status')}"
        )

    # Map human decision to alert decision
    decision_mapping = {
        "approve": AlertDecision.AUTO_CLOSE,  # Legitimate
        "reject": AlertDecision.ESCALATE,     # Confirmed fraud
        "escalate": AlertDecision.REVIEW_REQUIRED,  # Need more review
    }

    updated_decision = decision_mapping[request.decision].value

    # Update alert with human review
    updated_alert = await alert_store.update(
        alert_id,
        {
            "human_review": {
                "reviewer_id": request.reviewer_id,
                "reviewer_name": request.reviewer_name,
                "decision": request.decision,
                "reasoning": request.reasoning,
                "tags": request.tags,
                "reviewed_at": datetime.utcnow().isoformat(),
            },
            "decision": updated_decision,
            "recommendation": f"Human reviewer ({request.reviewer_name}) {request.decision}ed alert. "
                             f"Reasoning: {request.reasoning}",
        },
    )

    logger.info(f"Alert {alert_id} updated with human review: {request.decision}")

    # Send WebSocket update if enabled
    # await manager.send_update(alert_id, {
    #     "type": "human_review",
    #     "alert_id": alert_id,
    #     "decision": request.decision,
    #     "reviewer": request.reviewer_name,
    # })

    return ApprovalResponse(
        alert_id=alert_id,
        reviewer_id=request.reviewer_id,
        reviewer_name=request.reviewer_name,
        decision=request.decision,
        reasoning=request.reasoning,
        updated_decision=updated_decision,
        tags=request.tags,
    )


# =============================================================================
# Optional WebSocket Endpoint
# =============================================================================

# Uncomment to enable WebSocket support

# @app.websocket("/ws/triage/{alert_id}")
# async def websocket_endpoint(websocket: WebSocket, alert_id: str):
#     """
#     WebSocket endpoint for real-time alert updates.
#
#     Clients can connect to receive real-time updates as the alert
#     is processed through the triage workflow.
#
#     Args:
#         websocket: WebSocket connection
#         alert_id: Alert identifier to subscribe to
#     """
#     await manager.connect(websocket, alert_id)
#     try:
#         # Send initial status
#         alert_data = await alert_store.get(alert_id)
#         if alert_data:
#             await websocket.send_json({
#                 "type": "status",
#                 "alert_id": alert_id,
#                 "status": alert_data.get("status"),
#             })
#
#         # Keep connection alive and listen for messages
#         while True:
#             data = await websocket.receive_text()
#             # Echo back or handle client messages
#             await websocket.send_json({
#                 "type": "echo",
#                 "message": f"Received: {data}",
#             })
#     except WebSocketDisconnect:
#         manager.disconnect(websocket, alert_id)


# =============================================================================
# Startup/Shutdown Events
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Handle application startup."""
    logger.info("=" * 60)
    logger.info(f"🚀 {APIConfig.API_TITLE} v{APIConfig.API_VERSION}")
    logger.info(f"Environment: {APIConfig.AGENT_ENVIRONMENT}")
    logger.info(f"Agent initialized: {fraud_agent.environment}")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Handle application shutdown."""
    logger.info("Shutting down API...")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """
    Main entry point for running the API server.

    Usage:
        python -m src.api.main
    """
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
