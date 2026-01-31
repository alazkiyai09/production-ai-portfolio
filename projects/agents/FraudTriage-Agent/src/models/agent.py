"""
LangGraph agent state and related models.

Defines the TypedDict state used throughout the LangGraph workflow
and result models for agent outputs.
"""

from typing import Any, Required, TypedDict, override

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class AgentState(TypedDict, total=False):
    """
    LangGraph Agent State for fraud triage workflow.

    This TypedDict defines the state structure that flows through
    the LangGraph nodes and edges.

    Note: Required keys are marked with Required[] from typing_extensions
    to support Python 3.11 compatibility.
    """

    # Input: Alert data
    alert_id: Required[str]
    alert_data: Required[dict[str, Any]]

    # Context: Gathered during investigation
    transaction_history: list[dict[str, Any]]
    customer_profile: dict[str, Any]
    device_fingerprint: dict[str, Any]
    similar_alerts: list[dict[str, Any]]

    # Analysis: Risk assessment results
    risk_score: int
    risk_factors: list[str]
    confidence: float

    # Output: Recommendations and routing
    recommendation: str
    next_action: str
    requires_human_review: bool

    # Human-in-the-loop
    human_review_required: bool
    human_decision: str | None
    human_reasoning: str | None

    # Internal: Message history for LLM
    messages: list[BaseMessage]

    # Metadata
    iteration_count: int
    error_message: str | None


class TriageResult(BaseModel):
    """
    Result of the fraud triage analysis.

    This model represents the final output of the agent workflow
    after all nodes have executed.
    """

    alert_id: str = Field(..., description="Alert identifier")
    risk_score: int = Field(..., description="Final risk score (0-100)", ge=0, le=100)
    risk_factors: list[str] = Field(
        default_factory=list,
        description="List of identified risk factors"
    )
    confidence: float = Field(
        ...,
        description="Confidence in the assessment (0-1)",
        ge=0,
        le=1
    )
    recommendation: str = Field(..., description="Action recommendation")
    next_action: str = Field(..., description="Next action to take")
    requires_human_review: bool = Field(
        default=False,
        description="Whether human review is required"
    )

    # Context gathered
    transaction_summary: str | None = Field(None, description="Transaction history summary")
    customer_summary: str | None = Field(None, description="Customer profile summary")
    device_summary: str | None = Field(None, description="Device fingerprint summary")

    # Human review (if applicable)
    human_decision: str | None = Field(None, description="Human reviewer decision")
    human_reasoning: str | None = Field(None, description="Human reviewer reasoning")

    # Metadata
    analysis_duration_ms: int | None = Field(None, description="Analysis duration in milliseconds")
    model_used: str | None = Field(None, description="LLM model used")
    timestamp: str = Field(..., description="Analysis timestamp")

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "alert_id": "alert-12345",
                "risk_score": 75,
                "risk_factors": [
                    "Unusual location transaction",
                    "High value compared to historical average",
                    "New device fingerprint"
                ],
                "confidence": 0.85,
                "recommendation": "Escalate to fraud analyst for manual review",
                "next_action": "create_case",
                "requires_human_review": True,
                "transaction_summary": "Customer has 47 historical transactions...",
                "customer_summary": "Customer: John Doe, account age 3 years...",
                "device_summary": "New device detected, first transaction from this device...",
                "human_decision": None,
                "human_reasoning": None,
                "analysis_duration_ms": 2340,
                "model_used": "glm-4-plus",
                "timestamp": "2025-01-30T12:34:56Z"
            }
        }


class RiskAssessment(BaseModel):
    """Risk assessment result from the LLM."""

    risk_score: int = Field(..., description="Risk score from 0-100", ge=0, le=100)
    risk_factors: list[str] = Field(
        default_factory=list,
        description="Identified risk factors"
    )
    reasoning: str = Field(..., description="Reasoning behind the assessment")
    confidence: float = Field(..., description="Confidence level (0-1)", ge=0, le=1)
    suggested_action: str = Field(..., description="Suggested action to take")

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "risk_score": 72,
                "risk_factors": [
                    "Transaction from unusual location (Nigeria)",
                    "Amount 5x higher than customer average",
                    "First transaction from this IP address"
                ],
                "reasoning": "Multiple high-risk indicators present. Location is high-risk country...",
                "confidence": 0.82,
                "suggested_action": "escalate_for_review"
            }
        }
