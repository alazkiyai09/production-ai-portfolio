"""
State definition and data models for FraudTriage-Agent.

This module defines the TypedDict state used in the LangGraph workflow,
along with enums and Pydantic models for type safety and validation.
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import (
    Annotated,
    Any,
    Required,
    TypeAlias,
    Literal,
)

from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict

# LangGraph annotation for message reduction
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class RiskLevel(str, Enum):
    """
    Risk level classification.

    Attributes:
        LOW: Low risk, typically auto-closed
        MEDIUM: Medium risk, may require monitoring
        HIGH: High risk, requires human review
        CRITICAL: Critical risk, immediate action required
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertDecision(str, Enum):
    """
    Decision on how to handle the fraud alert.

    Attributes:
        AUTO_CLOSE: Alert is false positive, close automatically
        REVIEW_REQUIRED: Alert needs human analyst review
        ESCALATE: Alert should be escalated to fraud team
        BLOCK_TRANSACTION: Transaction should be blocked
    """
    AUTO_CLOSE = "auto_close"
    REVIEW_REQUIRED = "review_required"
    ESCALATE = "escalate"
    BLOCK_TRANSACTION = "block_transaction"


class AlertType(str, Enum):
    """
    Type of fraud alert.

    Attributes:
        UNUSUAL_AMOUNT: Transaction amount differs from customer's typical pattern
        VELOCITY: Too many transactions in a short time period
        LOCATION_MISMATCH: Transaction from unusual geographic location
        DEVICE_CHANGE: Transaction from new or unrecognized device
        ACCOUNT_TAKEOVER: Signs of account compromise
    """
    UNUSUAL_AMOUNT = "unusual_amount"
    VELOCITY = "velocity"
    LOCATION_MISMATCH = "location_mismatch"
    DEVICE_CHANGE = "device_change"
    ACCOUNT_TAKEOVER = "account_takeover"
    OTHER = "other"  # Fallback for unknown alert types


class WorkflowStage(str, Enum):
    """
    Workflow stage for state transition validation.

    Tracks the current stage of the fraud triage workflow to enable
    proper state transition validation and prevent invalid state updates.

    Attributes:
        INITIALIZED: Alert created, workflow not started
        PARSING: Alert data being parsed and validated
        GATHERING_CONTEXT: Collecting customer, transaction, and device data
        ASSESSING_RISK: Computing risk score and analyzing factors
        HUMAN_REVIEW: Waiting for human analyst input
        FINALIZING: Computing final metrics and completing workflow
        COMPLETED: Workflow finished successfully
        ERROR: Workflow encountered an error
    """
    INITIALIZED = "initialized"
    PARSING = "parsing"
    GATHERING_CONTEXT = "gathering_context"
    ASSESSING_RISK = "assessing_risk"
    HUMAN_REVIEW = "human_review"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    ERROR = "error"


# Valid state transitions for workflow stages
# Format: {current_stage: set(valid_next_stages)}
VALID_TRANSITIONS: dict[WorkflowStage, set[WorkflowStage]] = {
    WorkflowStage.INITIALIZED: {WorkflowStage.PARSING, WorkflowStage.ERROR},
    WorkflowStage.PARSING: {WorkflowStage.GATHERING_CONTEXT, WorkflowStage.ERROR},
    WorkflowStage.GATHERING_CONTEXT: {WorkflowStage.ASSESSING_RISK, WorkflowStage.ERROR},
    WorkflowStage.ASSESSING_RISK: {WorkflowStage.HUMAN_REVIEW, WorkflowStage.FINALIZING, WorkflowStage.ERROR},
    WorkflowStage.HUMAN_REVIEW: {WorkflowStage.HUMAN_REVIEW, WorkflowStage.FINALIZING, WorkflowStage.ERROR},
    WorkflowStage.FINALIZING: {WorkflowStage.COMPLETED, WorkflowStage.ERROR},
    WorkflowStage.COMPLETED: set(),  # Terminal state
    WorkflowStage.ERROR: {WorkflowStage.COMPLETED},  # Can only complete after error
}


class StateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    def __init__(self, current_stage: WorkflowStage, target_stage: WorkflowStage):
        self.current_stage = current_stage
        self.target_stage = target_stage
        valid_targets = VALID_TRANSITIONS.get(current_stage, set())
        super().__init__(
            f"Invalid state transition: {current_stage.value} -> {target_stage.value}. "
            f"Valid transitions from {current_stage.value}: "
            f"{[s.value for s in valid_targets]}"
        )


# =============================================================================
# Component Models (used within state)
# =============================================================================

class CustomerProfile(BaseModel):
    """
    Customer profile information.

    Contains customer details, account history, and risk indicators.
    """
    customer_id: str
    account_id: str
    name: str
    email: str | None = None
    phone: str | None = None
    account_age_days: int = Field(description="Age of account in days")
    kyc_verified: bool = Field(default=False, description="KYC verification status")
    customer_segment: str | None = Field(None, description="Customer segment (retail, premium, etc.)")
    risk_level: RiskLevel = Field(default=RiskLevel.LOW, description="Customer risk level")
    previous_fraud_cases: int = Field(default=0, description="Number of confirmed fraud cases")
    false_positive_count: int = Field(default=0, description="Number of false positives")
    average_transaction_amount: float = Field(default=0.0, description="Average transaction amount")
    highest_transaction_amount: float = Field(default=0.0, description="Highest historical transaction")
    typical_countries: list[str] = Field(default_factory=list, description="Typical transaction countries")
    registered_devices: list[str] = Field(default_factory=list, description="Known device IDs")

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class TransactionInfo(BaseModel):
    """
    Individual transaction information.
    """
    transaction_id: str
    account_id: str
    amount: float
    currency: str = Field(default="USD")
    merchant_name: str | None = None
    merchant_category: str | None = None
    location_city: str | None = None
    location_country: str | None = None
    timestamp: datetime
    device_id: str | None = None
    ip_address: str | None = None
    status: str = Field(default="completed")

    @field_validator("amount")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        """Validate amount is positive."""
        if v < 0:
            raise ValueError("Transaction amount must be positive")
        return v


class DeviceInfo(BaseModel):
    """
    Device fingerprint and security information.
    """
    device_id: str | None = None
    device_type: str | None = Field(None, description="mobile, desktop, tablet, etc.")
    operating_system: str | None = None
    browser: str | None = None
    is_new_device: bool = Field(default=False, description="Whether device is previously unseen")
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    transaction_count: int = Field(default=0, description="Number of transactions from this device")
    risk_score: int = Field(default=0, ge=0, le=100, description="Device risk score")
    is_emulator: bool = Field(default=False, description="Whether device appears to be an emulator")
    is_rooted: bool = Field(default=False, description="Whether device is rooted/jailbroken")

    class Config:
        """Pydantic configuration."""
        json_encoders = {datetime: lambda v: v.isoformat()}


class WatchlistHit(BaseModel):
    """
    Watchlist or sanctions list match information.
    """
    list_name: str = Field(..., description="Name of the watchlist")
    match_type: str = Field(..., description="Type of match (exact, partial, fuzzy)")
    match_confidence: float = Field(..., ge=0, le=1, description="Confidence of the match")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional match details")


class SimilarAlert(BaseModel):
    """
    Historical similar alert for comparison.
    """
    alert_id: str
    alert_type: AlertType
    similarity_score: float = Field(..., ge=0, le=1, description="Similarity to current alert")
    outcome: str = Field(..., description="Outcome of the alert (fraud, legitimate, pending)")
    risk_score: int = Field(..., ge=0, le=100)
    created_at: datetime


# =============================================================================
# TypedDict State for LangGraph
# =============================================================================

class FraudTriageState(TypedDict, total=False):
    """
    LangGraph state for fraud alert triage workflow.

    This TypedDict defines the state structure that flows through
    the LangGraph nodes and edges. Fields use Required[] to indicate
    mandatory inputs and total=False makes all fields optional.

    State Flow:
        1. Input: alert_id, alert_type, transaction data
        2. Context Gathering: customer_profile, transaction_history, etc.
        3. Analysis: risk_score, risk_level, risk_factors
        4. Decision: decision, recommendation, requires_human_review

    Attributes:
        # Input Fields (Required)
        ------------------------
        alert_id: Unique identifier for the alert
        alert_type: Type of fraud alert (enum)
        transaction_amount: Amount of the flagged transaction
        customer_id: Customer identifier

        # Message History (LangGraph managed)
        -------------------------------------
        messages: Conversation history with automatic reduction

        # Context Fields (populated during workflow)
        --------------------------------------------
        customer_profile: Customer profile and account information
        transaction_history: Historical transactions for pattern analysis
        watchlist_hits: Matches against sanctions/watchlists
        similar_alerts: Historical similar alerts for comparison
        device_info: Device fingerprint and security information

        # Analysis Fields (output from risk assessment)
        -----------------------------------------------
        risk_score: Numerical risk score (0-100)
        risk_level: Categorical risk level (enum)
        risk_factors: List of identified risk factors
        confidence: Confidence in the assessment (0-1)

        # Decision Fields (final routing decision)
        ------------------------------------------
        decision: How to handle the alert (enum)
        recommendation: Text explanation of recommendation
        requires_human_review: Whether human review is needed

        # Human Review Fields (for human-in-the-loop)
        ---------------------------------------------
        human_review_input: Human analyst's review input
        human_review_decision: Final decision after human review
        human_review_reasoning: Explanation from human reviewer
        human_reviewer_id: ID of the human reviewer

        # Metadata Fields
        -----------------
        processing_started: When processing started
        processing_completed: When processing completed
        processing_duration_ms: Total processing time in milliseconds
        iteration_count: Number of workflow iterations
        error_message: Any error that occurred during processing
    """

    # -------------------------------------------------------------------------
    # Input Fields (Required when creating state)
    # -------------------------------------------------------------------------
    alert_id: Required[str]
    alert_type: Required[AlertType]
    transaction_amount: Required[float]
    customer_id: Required[str]

    # Additional input fields (optional but commonly provided)
    account_id: str | None
    transaction_id: str | None
    transaction_country: str | None
    transaction_device_id: str | None
    transaction_ip: str | None
    rule_id: str | None
    alert_reason: str | None

    # -------------------------------------------------------------------------
    # Message History (LangGraph automatic reduction)
    # -------------------------------------------------------------------------
    messages: Annotated[list[BaseMessage], add_messages]

    # -------------------------------------------------------------------------
    # Context Fields (populated by context gathering tools)
    # -------------------------------------------------------------------------
    customer_profile: CustomerProfile | dict[str, Any] | None
    transaction_history: list[TransactionInfo] | list[dict[str, Any]] | None
    watchlist_hits: list[WatchlistHit] | list[dict[str, Any]] | None
    similar_alerts: list[SimilarAlert] | list[dict[str, Any]] | None
    device_info: DeviceInfo | dict[str, Any] | None

    # Additional context
    transaction_patterns: dict[str, Any] | None  # Pattern analysis results
    customer_risk_history: dict[str, Any] | None  # Historical risk information
    ip_reputation: dict[str, Any] | None  # IP reputation check results

    # -------------------------------------------------------------------------
    # Analysis Fields (output from risk assessment node)
    # -------------------------------------------------------------------------
    risk_score: float  # 0-100 numerical score
    risk_level: RiskLevel  # Categorical level
    risk_factors: list[str]  # List of specific risk indicators
    confidence: float  # 0-1 confidence in assessment

    # Analysis details
    risk_reasoning: str | None  # Explanation of risk assessment
    anomaly_flags: list[str] | None  # Specific anomalies detected
    behavioral_flags: list[str] | None  # Behavioral risk indicators

    # -------------------------------------------------------------------------
    # Decision Fields (final routing decision)
    # -------------------------------------------------------------------------
    decision: AlertDecision  # How to handle the alert
    recommendation: str  # Human-readable recommendation
    requires_human_review: bool  # Whether human review is needed

    # Decision rationale
    decision_rationale: str | None  # Explanation for the decision
    suggested_actions: list[str] | None  # Recommended follow-up actions

    # -------------------------------------------------------------------------
    # Human Review Fields (for human-in-the-loop workflow)
    # -------------------------------------------------------------------------
    human_review_input: str | None  # Input from human reviewer
    human_review_decision: AlertDecision | None  # Decision after human review
    human_review_reasoning: str | None  # Human reviewer's explanation
    human_reviewer_id: str | None  # ID of the human reviewer
    human_reviewer_name: str | None  # Name of the human reviewer
    human_review_timestamp: datetime | None  # When review was submitted

    # -------------------------------------------------------------------------
    # Metadata Fields
    # -------------------------------------------------------------------------
    processing_started: datetime | None  # Workflow start time
    processing_completed: datetime | None  # Workflow completion time
    processing_duration_ms: int | None  # Total processing duration
    iteration_count: int  # Number of workflow iterations
    error_message: str | None  # Any error that occurred
    model_used: str | None  # LLM model used for analysis

    # Workflow stage tracking for state transition validation
    workflow_stage: WorkflowStage  # Current workflow stage

    # Additional metadata
    workflow_version: str | None  # Version of the workflow
    tools_used: list[str] | None  # List of tools invoked
    checkpoint_data: dict[str, Any] | None  # Checkpoint for persistence


# =============================================================================
# API Request/Response Models
# =============================================================================

class FraudAlertRequest(BaseModel):
    """
    Request model for submitting a fraud alert.

    This is the input schema for the API endpoint that accepts
    new fraud alerts for triage.
    """
    alert_id: str | None = Field(None, description="Unique alert identifier (auto-generated if not provided)")
    alert_type: AlertType = Field(..., description="Type of fraud alert")
    customer_id: str = Field(..., description="Customer identifier", min_length=1)
    account_id: str = Field(..., description="Account identifier", min_length=1)

    # Transaction details
    transaction_id: str = Field(..., description="Transaction identifier", min_length=1)
    transaction_amount: float = Field(..., description="Transaction amount", gt=0)
    transaction_currency: str = Field(default="USD", description="Transaction currency code")
    transaction_country: str = Field(..., description="Transaction country code (ISO 3166-1 alpha-2)", min_length=2, max_length=2)
    transaction_city: str | None = Field(None, description="Transaction city")
    merchant_name: str | None = Field(None, description="Merchant name")
    merchant_category: str | None = Field(None, description="Merchant category code")

    # Device and IP
    device_id: str | None = Field(None, description="Device identifier")
    ip_address: str | None = Field(None, description="IP address")

    # Alert metadata
    rule_id: str | None = Field(None, description="Rule that triggered the alert")
    alert_reason: str = Field(..., description="Reason for the alert", min_length=10)
    severity: str = Field(default="medium", description="Initial severity assessment")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "alert_type": "location_mismatch",
                "customer_id": "CUST-41256",
                "account_id": "ACC-78542",
                "transaction_id": "TXN-2025-014785",
                "transaction_amount": 4250.00,
                "transaction_currency": "USD",
                "transaction_country": "NG",
                "transaction_city": "Lagos",
                "merchant_name": "Luxury Electronics Store",
                "merchant_category": "electronics",
                "device_id": "DEVICE-NEW-999888",
                "ip_address": "197.210.77.52",
                "rule_id": "RULE-UNUSUAL-LOC-001",
                "alert_reason": "Customer with no prior Nigeria transactions attempts $4,250 purchase from Lagos",
                "severity": "high",
            }
        }


class FraudAlertResponse(BaseModel):
    """
    Response model for fraud alert submission.
    """
    alert_id: str = Field(..., description="Alert identifier")
    status: str = Field(..., description="Alert status (pending, investigating, reviewed, etc.)")
    message: str = Field(..., description="Response message")
    submitted_at: datetime = Field(default_factory=datetime.utcnow, description="Submission timestamp")

    class Config:
        """Pydantic configuration."""
        json_encoders = {datetime: lambda v: v.isoformat()}


class TriageResultResponse(BaseModel):
    """
    Response model for triage analysis results.

    Contains the complete analysis including risk assessment,
    decision, and recommendations.
    """
    alert_id: str = Field(..., description="Alert identifier")
    status: str = Field(..., description="Final alert status")

    # Risk assessment
    risk_score: float = Field(..., description="Risk score (0-100)", ge=0, le=100)
    risk_level: RiskLevel = Field(..., description="Risk level classification")
    risk_factors: list[str] = Field(default_factory=list, description="Identified risk factors")
    confidence: float = Field(..., description="Confidence in assessment (0-1)", ge=0, le=1)

    # Decision
    decision: AlertDecision = Field(..., description="Final decision on alert handling")
    recommendation: str = Field(..., description="Detailed recommendation")
    requires_human_review: bool = Field(..., description="Whether human review is required")

    # Human review (if applicable)
    human_review_decision: AlertDecision | None = Field(None, description="Human reviewer's decision")
    human_review_reasoning: str | None = Field(None, description="Human reviewer's explanation")
    human_reviewer_name: str | None = Field(None, description="Name of human reviewer")

    # Context summary
    customer_summary: str | None = Field(None, description="Customer profile summary")
    transaction_summary: str | None = Field(None, description="Transaction pattern summary")
    device_summary: str | None = Field(None, description="Device information summary")

    # Metadata
    processing_started: datetime = Field(..., description="When processing started")
    processing_completed: datetime = Field(..., description="When processing completed")
    processing_duration_ms: int = Field(..., description="Processing duration in milliseconds")
    model_used: str | None = Field(None, description="LLM model used")

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}
        json_schema_extra = {
            "example": {
                "alert_id": "SAMPLE-001",
                "status": "reviewed",
                "risk_score": 78.5,
                "risk_level": "high",
                "risk_factors": [
                    "First transaction from Nigeria",
                    "Amount 14x higher than customer average",
                    "New device never seen before",
                    "High-risk IP address (proxy/VPN detected)"
                ],
                "confidence": 0.85,
                "decision": "review_required",
                "recommendation": "Escalate to fraud analyst for manual review due to multiple high-risk indicators including unusual location, new device, and significant amount deviation from customer's typical pattern.",
                "requires_human_review": True,
                "human_review_decision": None,
                "human_review_reasoning": None,
                "human_reviewer_name": None,
                "customer_summary": "Customer: John Smith, 3-year account, retail segment, low risk history",
                "transaction_summary": "47 transactions in 90 days, avg $285, typical countries: US, CA, UK",
                "device_summary": "New device (mobile, iOS), first transaction, high risk score 70",
                "processing_started": "2025-01-30T14:25:00Z",
                "processing_completed": "2025-01-30T14:25:03Z",
                "processing_duration_ms": 3240,
                "model_used": "glm-4-plus"
            }
        }


class HumanReviewRequest(BaseModel):
    """
    Request model for human review submission.

    Allows fraud analysts to provide their decision and reasoning.
    """
    reviewer_id: str = Field(..., description="Analyst ID", min_length=1)
    reviewer_name: str = Field(..., description="Analyst name", min_length=1)
    decision: AlertDecision = Field(..., description="Review decision")
    reasoning: str = Field(..., description="Reasoning for the decision", min_length=20)
    agreed_with_agent: bool | None = Field(None, description="Whether analyst agreed with agent recommendation")
    suggested_risk_score: int | None = Field(None, description="Analyst's opinion on correct risk score", ge=0, le=100)
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    additional_notes: str | None = Field(None, description="Additional notes or comments")

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "reviewer_id": "ANALYST-001",
                "reviewer_name": "Jane Smith",
                "decision": "escalate",
                "reasoning": "Contacted customer who confirmed they did not authorize this transaction. Nigeria location and new device are confirmed fraud indicators.",
                "agreed_with_agent": True,
                "suggested_risk_score": 85,
                "tags": ["confirmed_fraud", "account_takeover", "international"],
                "additional_notes": "Case created for follow-up investigation. Block card."
            }
        }


class HumanReviewResponse(BaseModel):
    """
    Response model for human review submission.
    """
    review_id: str = Field(..., description="Review identifier")
    alert_id: str = Field(..., description="Alert identifier")
    decision: AlertDecision = Field(..., description="Review decision")
    message: str = Field(..., description="Response message")
    reviewed_at: datetime = Field(default_factory=datetime.utcnow, description="Review timestamp")

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class AlertStatusResponse(BaseModel):
    """
    Response model for alert status check.

    Provides current status without full details.
    """
    alert_id: str = Field(..., description="Alert identifier")
    status: str = Field(..., description="Current status")
    created_at: datetime = Field(..., description="Alert creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    requires_human_review: bool = Field(..., description="Whether human review is required")
    risk_score: float | None = Field(None, description="Risk score if analysis complete")

    class Config:
        """Pydantic configuration."""
        json_encoders = {datetime: lambda v: v.isoformat()}


# =============================================================================
# Helper Functions
# =============================================================================

def create_initial_state(
    alert_id: str,
    alert_type: AlertType,
    transaction_amount: float,
    customer_id: str,
    **kwargs: Any,
) -> FraudTriageState:
    """
    Create initial FraudTriageState from alert data.

    Args:
        alert_id: Unique alert identifier
        alert_type: Type of fraud alert
        transaction_amount: Amount of flagged transaction
        customer_id: Customer identifier
        **kwargs: Additional optional fields

    Returns:
        Initial state dictionary with required fields populated
    """
    from datetime import datetime

    state: FraudTriageState = {
        # Required input fields
        "alert_id": alert_id,
        "alert_type": alert_type,
        "transaction_amount": transaction_amount,
        "customer_id": customer_id,

        # Messages (empty initially)
        "messages": [],

        # Context fields (will be populated by tools)
        "customer_profile": None,
        "transaction_history": None,
        "watchlist_hits": None,
        "similar_alerts": None,
        "device_info": None,
        "transaction_patterns": None,
        "customer_risk_history": None,
        "ip_reputation": None,

        # Analysis fields (will be computed)
        "risk_score": 0.0,
        "risk_level": RiskLevel.LOW,
        "risk_factors": [],
        "confidence": 0.0,

        # Decision fields (will be determined)
        "decision": AlertDecision.REVIEW_REQUIRED,
        "recommendation": "",
        "requires_human_review": False,

        # Human review fields (empty initially)
        "human_review_input": None,
        "human_review_decision": None,
        "human_review_reasoning": None,
        "human_reviewer_id": None,
        "human_reviewer_name": None,
        "human_review_timestamp": None,

        # Metadata
        "processing_started": datetime.utcnow(),
        "processing_completed": None,
        "processing_duration_ms": None,
        "iteration_count": 0,
        "error_message": None,
        "model_used": None,
        "workflow_stage": WorkflowStage.INITIALIZED,  # Initialize stage
        "workflow_version": "0.1.0",
        "tools_used": None,
        "checkpoint_data": None,
    }

    # Add optional fields from kwargs
    for key, value in kwargs.items():
        if key in state:
            state[key] = value  # type: ignore[typeddict-item]

    return state


def validate_state_transition(
    state: FraudTriageState,
    target_stage: WorkflowStage,
) -> None:
    """
    Validate that a state transition is allowed.

    Args:
        state: Current workflow state
        target_stage: Desired workflow stage

    Raises:
        StateTransitionError: If transition is not valid

    Example:
        >>> current_state = create_initial_state(...)
        >>> validate_state_transition(current_state, WorkflowStage.PARSING)
        >>> # Proceed with parsing
        >>> validate_state_transition(current_state, WorkflowStage.COMPLETED)
        >>> StateTransitionError: Cannot jump from initialized to completed
    """
    current_stage = state.get("workflow_stage", WorkflowStage.INITIALIZED)

    if current_stage == target_stage:
        # No transition needed
        return

    valid_transitions = VALID_TRANSITIONS.get(current_stage, set())

    if target_stage not in valid_transitions:
        raise StateTransitionError(current_stage, target_stage)

    logger.debug(
        f"State transition validated: {current_stage.value} -> {target_stage.value}"
    )


def transition_to_stage(
    state: FraudTriageState,
    target_stage: WorkflowStage,
) -> FraudTriageState:
    """
    Transition state to a new workflow stage with validation.

    Args:
        state: Current workflow state
        target_stage: Desired workflow stage

    Returns:
        Updated state with new workflow_stage

    Raises:
        StateTransitionError: If transition is not valid

    Example:
        >>> state = create_initial_state(...)
        >>> state = transition_to_stage(state, WorkflowStage.PARSING)
        >>> assert state["workflow_stage"] == WorkflowStage.PARSING
    """
    validate_state_transition(state, target_stage)

    # Create updated state with new stage
    state["workflow_stage"] = target_stage  # type: ignore[typeddict-item]

    return state


def state_to_response(state: FraudTriageState) -> TriageResultResponse:
    """
    Convert FraudTriageState to TriageResultResponse for API response.

    Args:
        state: Complete or partial fraud triage state

    Returns:
        API response model with relevant state information
    """
    return TriageResultResponse(
        alert_id=state["alert_id"],
        status="completed" if state.get("processing_completed") else "in_progress",
        risk_score=state.get("risk_score", 0.0),
        risk_level=state.get("risk_level", RiskLevel.LOW),
        risk_factors=state.get("risk_factors", []),
        confidence=state.get("confidence", 0.0),
        decision=state.get("decision", AlertDecision.REVIEW_REQUIRED),
        recommendation=state.get("recommendation", "Analysis in progress"),
        requires_human_review=state.get("requires_human_review", False),
        human_review_decision=state.get("human_review_decision"),
        human_review_reasoning=state.get("human_review_reasoning"),
        human_reviewer_name=state.get("human_reviewer_name"),
        customer_summary=_build_customer_summary(state.get("customer_profile")),
        transaction_summary=_build_transaction_summary(state.get("transaction_history")),
        device_summary=_build_device_summary(state.get("device_info")),
        processing_started=state.get("processing_started") or datetime.utcnow(),
        processing_completed=state.get("processing_completed") or datetime.utcnow(),
        processing_duration_ms=state.get("processing_duration_ms", 0),
        model_used=state.get("model_used"),
    )


def _build_customer_summary(profile: Any) -> str | None:
    """Build customer summary from profile data."""
    if not profile or isinstance(profile, dict) and not profile:
        return None

    if isinstance(profile, CustomerProfile):
        return f"Customer: {profile.name}, {profile.account_age_days // 365}-year account, {profile.customer_segment} segment, {profile.risk_level.value} risk"
    elif isinstance(profile, dict):
        name = profile.get("name", "Unknown")
        age_years = profile.get("account_age_years", profile.get("account_age_days", 0) // 365)
        segment = profile.get("customer_segment", "unknown")
        risk = profile.get("risk_level", "unknown")
        return f"Customer: {name}, {age_years}-year account, {segment} segment, {risk} risk"

    return None


def _build_transaction_summary(history: Any) -> str | None:
    """Build transaction summary from history data."""
    if not history or isinstance(history, list) and not history:
        return None

    count = len(history) if isinstance(history, list) else 0
    if count > 0:
        return f"{count} historical transactions available for analysis"
    return None


def _build_device_summary(device: Any) -> str | None:
    """Build device summary from device info."""
    if not device or isinstance(device, dict) and not device:
        return None

    if isinstance(device, DeviceInfo):
        status = "NEW" if device.is_new_device else "KNOWN"
        return f"{status} device ({device.device_type}), risk score {device.risk_score}/100"
    elif isinstance(device, dict):
        is_new = device.get("is_new_device", False)
        status = "NEW" if is_new else "KNOWN"
        dev_type = device.get("device_type", "Unknown")
        risk = device.get("risk_score", 0)
        return f"{status} device ({dev_type}), risk score {risk}/100"

    return None
