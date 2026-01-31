"""Data models for FraudTriage-Agent."""

# State definition (primary state model for LangGraph)
from .state import (
    # State
    FraudTriageState,
    create_initial_state,
    state_to_response,
    # Enums
    RiskLevel,
    AlertDecision,
    AlertType as StateAlertType,
    # Component models
    CustomerProfile,
    TransactionInfo,
    DeviceInfo,
    WatchlistHit,
    SimilarAlert,
    # API models
    FraudAlertRequest,
    FraudAlertResponse,
    TriageResultResponse,
    HumanReviewRequest,
    HumanReviewResponse,
    AlertStatusResponse,
)

__all__ = [
    # State (NEW - primary state definition)
    "FraudTriageState",
    "create_initial_state",
    "state_to_response",
    # State enums (NEW)
    "RiskLevel",
    "AlertDecision",
    "StateAlertType",
    # Component models (NEW)
    "CustomerProfile",
    "TransactionInfo",
    "DeviceInfo",
    "WatchlistHit",
    "SimilarAlert",
    # API models (NEW)
    "FraudAlertRequest",
    "FraudAlertResponse",
    "TriageResultResponse",
    "HumanReviewRequest",
    "HumanReviewResponse",
    "AlertStatusResponse",
]
