"""
Unit tests for agent nodes and workflow.

Tests individual nodes and the complete LangGraph workflow.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.nodes import (
    parse_alert_node,
    gather_context_node,
    assess_risk_node,
    route_alert,
)
from src.agents.state import create_initial_state, AgentState


@pytest.mark.asyncio
class TestParseAlertNode:
    """Test parse_alert node."""

    async def test_parse_alert_basic(self, sample_alert):
        """Test basic alert parsing."""
        state = create_initial_state(sample_alert)

        result = await parse_alert_node(state)

        assert result["alert_id"] == "test-alert-001"
        assert "parsed_alert" in result["alert_data"]
        assert result["alert_data"]["parsed_alert"]["alert_type"] == "unusual_location"
        assert result["iteration_count"] == 1
        assert len(result["messages"]) > 0

    async def test_parse_alert_extracts_transaction_info(self, sample_alert):
        """Test that transaction info is extracted."""
        state = create_initial_state(sample_alert)

        result = await parse_alert_node(state)
        parsed = result["alert_data"]["parsed_alert"]

        assert parsed["transaction_amount"] == 2500.00
        assert parsed["transaction_location"] == "NG"
        assert parsed["transaction_id"] == "txn-001"


@pytest.mark.asyncio
class TestGatherContextNode:
    """Test gather_context node."""

    async def test_gather_context_with_mock_data(self, sample_alert, monkeypatch):
        """Test context gathering with mock APIs."""
        from src.config.settings import settings
        monkeypatch.setattr(settings, "mock_external_apis", True)

        # Add parsed alert data
        sample_alert["parsed_alert"] = {
            "account_id": "acc-12345",
            "customer_id": "cust-67890",
            "transaction_id": "txn-001",
        }

        state = create_initial_state(sample_alert)
        state = await parse_alert_node(state)  # Parse first

        result = await gather_context_node(state)

        assert len(result["transaction_history"]) > 0
        assert result["customer_profile"]["customer_id"] == "cust-67890"
        assert result["device_fingerprint"]["device_id"] == "device-new-123"
        assert result["iteration_count"] >= 2


@pytest.mark.asyncio
class TestAssessRiskNode:
    """Test assess_risk node."""

    async def test_assess_risk_with_mock_llm(self, sample_alert, mock_llm, monkeypatch):
        """Test risk assessment with mocked LLM."""
        from src.config.settings import settings
        monkeypatch.setattr(settings, "mock_external_apis", True)

        # Prepare state with context
        sample_alert["parsed_alert"] = {
            "account_id": "acc-12345",
            "customer_id": "cust-67890",
            "transaction_id": "txn-001",
        }

        state = create_initial_state(sample_alert)
        state = await parse_alert_node(state)
        state = await gather_context_node(state)

        result = await assess_risk_node(state)

        assert "risk_score" in result
        assert result["risk_score"] >= 0
        assert result["risk_score"] <= 100
        assert "risk_factors" in result
        assert "confidence" in result
        assert "recommendation" in result
        assert "next_action" in result


@pytest.mark.asyncio
class TestRouteAlert:
    """Test route_alert conditional edge."""

    def test_route_high_risk(self):
        """Test routing for high-risk alerts."""
        state: AgentState = {
            "alert_id": "test-001",
            "alert_data": {},
            "risk_score": 85,
            "next_action": "create_case",
            "requires_human_review": True,
            "human_review_required": True,
            "human_decision": None,
            "messages": [],
            "transaction_history": [],
            "customer_profile": {},
            "device_fingerprint": {},
            "similar_alerts": [],
            "risk_factors": [],
            "confidence": 0.0,
            "recommendation": "",
            "human_reasoning": None,
            "iteration_count": 0,
            "error_message": None,
        }

        result = route_alert(state)
        assert result == "escalate"

    def test_route_low_risk(self):
        """Test routing for low-risk alerts."""
        state: AgentState = {
            "alert_id": "test-002",
            "alert_data": {},
            "risk_score": 25,
            "next_action": "auto_close",
            "requires_human_review": False,
            "human_review_required": False,
            "human_decision": None,
            "messages": [],
            "transaction_history": [],
            "customer_profile": {},
            "device_fingerprint": {},
            "similar_alerts": [],
            "risk_factors": [],
            "confidence": 0.0,
            "recommendation": "",
            "human_reasoning": None,
            "iteration_count": 0,
            "error_message": None,
        }

        result = route_alert(state)
        assert result == "auto_close"

    def test_route_human_review_confirmed_fraud(self):
        """Test routing after human review confirms fraud."""
        state: AgentState = {
            "alert_id": "test-003",
            "alert_data": {},
            "risk_score": 70,
            "next_action": "awaiting_human_review",
            "requires_human_review": True,
            "human_review_required": True,
            "human_decision": "confirm_fraud",
            "messages": [],
            "transaction_history": [],
            "customer_profile": {},
            "device_fingerprint": {},
            "similar_alerts": [],
            "risk_factors": [],
            "confidence": 0.0,
            "recommendation": "",
            "human_reasoning": None,
            "iteration_count": 0,
            "error_message": None,
        }

        result = route_alert(state)
        assert result == "escalate"

    def test_route_human_review_legitimate(self):
        """Test routing after human review confirms legitimate."""
        state: AgentState = {
            "alert_id": "test-004",
            "alert_data": {},
            "risk_score": 70,
            "next_action": "awaiting_human_review",
            "requires_human_review": True,
            "human_review_required": True,
            "human_decision": "confirm_legitimate",
            "messages": [],
            "transaction_history": [],
            "customer_profile": {},
            "device_fingerprint": {},
            "similar_alerts": [],
            "risk_factors": [],
            "confidence": 0.0,
            "recommendation": "",
            "human_reasoning": None,
            "iteration_count": 0,
            "error_message": None,
        }

        result = route_alert(state)
        assert result == "auto_close"


@pytest.mark.asyncio
class TestCompleteWorkflow:
    """Test the complete workflow integration."""

    async def test_workflow_low_risk_alert(self, sample_alert_low_risk, mock_llm, monkeypatch):
        """Test complete workflow for low-risk alert."""
        from src.config.settings import settings
        monkeypatch.setattr(settings, "mock_external_apis", True)
        monkeypatch.setattr(settings, "high_risk_threshold", 70)
        monkeypatch.setattr(settings, "medium_risk_threshold", 40)

        # Mock LLM to return low risk
        mock_response = MagicMock()
        mock_response.content = """
        Risk Score: 25
        Risk Factors:
        - Known customer
        - Regular transaction pattern
        - Established device
        Confidence: 0.9
        Suggested Action: auto_close
        """
        mock_llm.ainvoke.return_value = mock_response

        sample_alert_low_risk["parsed_alert"] = {
            "account_id": "acc-12345",
            "customer_id": "cust-67890",
            "transaction_id": "txn-002",
        }

        from src.agents.graph import get_fraud_triage_graph
        graph = get_fraud_triage_graph()

        result = await graph.arun(sample_alert_low_risk)

        assert result["risk_score"] <= 40
        assert result["next_action"] in ["auto_close", "monitor"]

    async def test_workflow_high_risk_alert(self, sample_alert, mock_llm, monkeypatch):
        """Test complete workflow for high-risk alert."""
        from src.config.settings import settings
        monkeypatch.setattr(settings, "mock_external_apis", True)

        sample_alert["parsed_alert"] = {
            "account_id": "acc-12345",
            "customer_id": "cust-67890",
            "transaction_id": "txn-001",
        }

        from src.agents.graph import get_fraud_triage_graph
        graph = get_fraud_triage_graph()

        result = await graph.arun(sample_alert)

        assert result["risk_score"] > 0
        assert "risk_factors" in result
        assert result["iteration_count"] > 0
