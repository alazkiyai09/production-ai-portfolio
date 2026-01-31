"""
Pydantic models for fraud alerts and related data.

These models represent the structure of fraud alerts, transactions,
and related data throughout the triage workflow.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TransactionType(str, Enum):
    """Transaction type enum."""

    PURCHASE = "purchase"
    WITHDRAWAL = "withdrawal"
    TRANSFER = "transfer"
    DEPOSIT = "deposit"
    PAYMENT = "payment"
    REFUND = "refund"


class AlertType(str, Enum):
    """Fraud alert type enum."""

    UNUSUAL_LOCATION = "unusual_location"
    HIGH_VALUE_TRANSACTION = "high_value_transaction"
    VELOCITY_CHECK = "velocity_check"
    ACCOUNT_TAKEOVER = "account_takeover"
    NEW_DEVICE = "new_device"
    SUSPICIOUS_MERCHANT = "suspicious_merchant"
    PHISHING_INDICATOR = "phishing_indicator"
    OTHER = "other"


class AlertStatus(str, Enum):
    """Alert status enum."""

    PENDING = "pending"
    INVESTIGATING = "investigating"
    AWAITING_REVIEW = "awaiting_review"
    REVIEWED = "reviewed"
    AUTO_CLOSED = "auto_closed"
    ESCALATED = "escalated"


class Transaction(BaseModel):
    """Transaction model."""

    transaction_id: str = Field(..., description="Unique transaction identifier")
    account_id: str = Field(..., description="Account identifier")
    amount: float = Field(..., description="Transaction amount", gt=0)
    currency: str = Field(default="USD", description="Currency code")
    transaction_type: TransactionType = Field(..., description="Type of transaction")
    merchant_name: str | None = Field(None, description="Merchant name")
    merchant_category: str | None = Field(None, description="Merchant category code")
    location_city: str | None = Field(None, description="Transaction city")
    location_country: str = Field(..., description="Transaction country code")
    timestamp: datetime = Field(..., description="Transaction timestamp")
    device_id: str | None = Field(None, description="Device identifier")
    ip_address: str | None = Field(None, description="IP address")
    status: str = Field(default="completed", description="Transaction status")

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class Alert(BaseModel):
    """Base alert model."""

    alert_id: UUID = Field(default_factory=uuid4, description="Unique alert identifier")
    alert_type: AlertType = Field(..., description="Type of fraud alert")
    status: AlertStatus = Field(default=AlertStatus.PENDING, description="Alert status")
    severity: str = Field(default="medium", description="Alert severity (low/medium/high)")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class FraudAlert(Alert):
    """
    Fraud alert model with full context.

    This is the main model used throughout the triage workflow.
    """

    # Core alert data
    account_id: str = Field(..., description="Customer account identifier")
    customer_id: str = Field(..., description="Customer identifier")
    transaction: Transaction = Field(..., description="Flagged transaction")

    # Alert-specific data
    rule_id: str | None = Field(None, description="Rule that triggered the alert")
    rule_description: str | None = Field(None, description="Rule description")
    risk_score: int | None = Field(None, description="Initial risk score (0-100)", ge=0, le=100)
    alert_reason: str = Field(..., description="Reason for alert")

    # Triage results (populated during workflow)
    triage_result: dict[str, Any] | None = Field(None, description="Triage analysis results")
    risk_factors: list[str] = Field(default_factory=list, description="Identified risk factors")
    recommendation: str | None = Field(None, description="Action recommendation")
    requires_human_review: bool = Field(default=False, description="Whether human review is needed")
    human_review: dict[str, Any] | None = Field(None, description="Human review details")

    # Case management
    case_id: str | None = Field(None, description="Case ID if escalated")
    resolved: bool = Field(default=False, description="Whether alert is resolved")
    resolution_note: str | None = Field(None, description="Resolution notes")

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}
