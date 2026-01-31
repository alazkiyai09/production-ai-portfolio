"""
Human review models for the human-in-the-loop workflow.

Defines models for human review decisions and feedback.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class HumanReviewDecision(str, Enum):
    """Human review decision enum."""

    CONFIRM_FRAUD = "confirm_fraud"
    CONFIRM_LEGITIMATE = "confirm_legitimate"
    NEED_MORE_INFO = "need_more_info"
    ESCALATE = "escalate"


class HumanReview(BaseModel):
    """Human review model for analyst feedback."""

    review_id: UUID = Field(default_factory=uuid4, description="Unique review identifier")
    alert_id: str = Field(..., description="Associated alert ID")
    reviewer_id: str = Field(..., description="Reviewer identifier")
    reviewer_name: str = Field(..., description="Reviewer name")

    # Decision
    decision: HumanReviewDecision = Field(..., description="Review decision")
    reasoning: str = Field(..., description="Reasoning for the decision")

    # Additional feedback
    additional_notes: str | None = Field(None, description="Additional notes")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")

    # Case management
    case_created: bool = Field(default=False, description="Whether a case was created")
    case_id: str | None = Field(None, description="Created case ID")

    # Timestamps
    reviewed_at: datetime = Field(default_factory=datetime.utcnow, description="Review timestamp")
    time_to_review_minutes: int | None = Field(
        None,
        description="Time from alert creation to review (minutes)"
    )

    # Feedback for model improvement
    agreed_with_agent: bool | None = Field(
        None,
        description="Whether reviewer agreed with agent recommendation"
    )
    correct_risk_score: int | None = Field(
        None,
        description="Reviewer's opinion on correct risk score",
        ge=0,
        le=100
    )

    model_config = {
        "json_encoders": {datetime: lambda v: v.isoformat()},
        "json_schema_extra": {
            "examples": [{
                "review_id": "550e8400-e29b-41d4-a716-446655440000",
                "alert_id": "alert-12345",
                "reviewer_id": "analyst-001",
                "reviewer_name": "Jane Smith",
                "decision": "confirm_fraud",
                "reasoning": "Transaction location is confirmed fraudulent. Customer confirmed they did not authorize.",
                "additional_notes": "Contacted customer via phone for verification.",
                "tags": ["confirmed_fraud", "account_takeover"],
                "case_created": True,
                "case_id": "CASE-2025-001234",
                "reviewed_at": "2025-01-30T14:30:00Z",
                "time_to_review_minutes": 45,
                "agreed_with_agent": True,
                "correct_risk_score": 80
            }]
        }
    }


class ReviewRequest(BaseModel):
    """Request model for submitting a human review."""

    reviewer_id: str = Field(..., description="Reviewer identifier")
    reviewer_name: str = Field(..., description="Reviewer name")
    decision: HumanReviewDecision = Field(..., description="Review decision")
    reasoning: str = Field(..., description="Reasoning for the decision", min_length=10)
    additional_notes: str | None = Field(None, description="Additional notes")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    agreed_with_agent: bool | None = Field(None, description="Agreement with agent recommendation")
    correct_risk_score: int | None = Field(None, description="Correct risk score opinion", ge=0, le=100)

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "reviewer_id": "analyst-001",
                "reviewer_name": "Jane Smith",
                "decision": "confirm_fraud",
                "reasoning": "Transaction location is confirmed fraudulent based on customer verification.",
                "additional_notes": "Case created for follow-up investigation.",
                "tags": ["confirmed_fraud"],
                "agreed_with_agent": True,
                "correct_risk_score": 85
            }]
        }
    }
