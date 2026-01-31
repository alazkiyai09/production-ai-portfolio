"""
Agent state definition for LangGraph workflow.

This module defines the TypedDict state structure and helper functions
for state management throughout the fraud triage workflow.
"""

from typing import Any, Required, TypedDict, override

from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    """
    LangGraph Agent State for fraud triage workflow.

    This state flows through all nodes in the LangGraph workflow.
    Each node can read and update specific fields in the state.

    Fields:
        alert_id: Unique identifier for the alert
        alert_data: Raw alert data from the input
        transaction_history: Historical transactions for context
        customer_profile: Customer profile information
        device_fingerprint: Device and IP information
        similar_alerts: Historical similar alerts for comparison
        risk_score: Calculated risk score (0-100)
        risk_factors: List of identified risk factors
        confidence: Confidence in the assessment (0-1)
        recommendation: Action recommendation
        next_action: Next action to take (routes to different nodes)
        requires_human_review: Whether human review is needed
        human_review_required: Flag for human-in-the-loop
        human_decision: Human reviewer's decision
        human_reasoning: Human reviewer's reasoning
        messages: Message history for LLM context
        iteration_count: Number of workflow iterations
        error_message: Any error that occurred during processing
    """

    # Input data
    alert_id: Required[str]
    alert_data: Required[dict[str, Any]]

    # Gathered context
    transaction_history: list[dict[str, Any]]
    customer_profile: dict[str, Any]
    device_fingerprint: dict[str, Any]
    similar_alerts: list[dict[str, Any]]

    # Risk assessment
    risk_score: int
    risk_factors: list[str]
    confidence: float

    # Decision making
    recommendation: str
    next_action: str
    requires_human_review: bool

    # Human-in-the-loop
    human_review_required: bool
    human_decision: str | None
    human_reasoning: str | None

    # Internal state
    messages: list[BaseMessage]
    iteration_count: int
    error_message: str | None


def create_initial_state(alert_data: dict[str, Any]) -> AgentState:
    """
    Create initial agent state from alert data.

    Args:
        alert_data: Raw alert data

    Returns:
        Initial agent state with required fields populated
    """
    alert_id = alert_data.get("alert_id") or alert_data.get("id") or "unknown"

    return {
        "alert_id": alert_id,
        "alert_data": alert_data,
        "transaction_history": [],
        "customer_profile": {},
        "device_fingerprint": {},
        "similar_alerts": [],
        "risk_score": 0,
        "risk_factors": [],
        "confidence": 0.0,
        "recommendation": "",
        "next_action": "gather_context",
        "requires_human_review": False,
        "human_review_required": False,
        "human_decision": None,
        "human_reasoning": None,
        "messages": [],
        "iteration_count": 0,
        "error_message": None,
    }


def state_to_dict(state: AgentState) -> dict[str, Any]:
    """
    Convert agent state to a regular dictionary for serialization.

    Args:
        state: Agent state

    Returns:
        Dictionary representation of state
    """
    result = {}

    for key, value in state.items():
        if isinstance(value, list) and value and isinstance(value[0], BaseMessage):
            # Convert BaseMessage list to dict
            result[key] = [msg.dict() for msg in value]
        else:
            result[key] = value

    return result
