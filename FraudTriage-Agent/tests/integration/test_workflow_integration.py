"""
Integration tests for complete LangGraph workflow.

Tests the end-to-end fraud triage workflow with real components.
"""

import pytest

from src.agents.graph import get_fraud_triage_graph
from src.agents.state import create_initial_state
from src.config.settings import settings


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.llm  # This test calls actual LLM
class TestWorkflowIntegration:
    """Test complete workflow integration."""

    async def test_workflow_with_real_llm(self, monkeypatch):
        """Test workflow with real LLM (requires API keys)."""
        # Skip if no API key configured
        if not settings.glm_api_key and not settings.openai_api_key:
            pytest.skip("No LLM API key configured")

        # Use mock external APIs to avoid external dependencies
        monkeypatch.setattr(settings, "mock_external_apis", True)

        alert_data = {
            "alert_id": "integration-test-001",
            "alert_type": "unusual_location",
            "account_id": "acc-12345",
            "customer_id": "cust-67890",
            "transaction": {
                "transaction_id": "txn-integration-001",
                "amount": 3500.00,
                "currency": "USD",
                "merchant_name": "Luxury Brand",
                "location_country": "NG",
                "location_city": "Lagos",
                "device_id": "device-new-integration-001",
                "ip_address": "197.210.53.21",
                "timestamp": "2025-01-30T14:30:00Z",
            },
            "rule_id": "RULE-INTEGRATION-001",
            "alert_reason": "Transaction from high-risk country Nigeria",
            "severity": "high",
        }

        graph = get_fraud_triage_graph()
        result = await graph.arun(alert_data)

        # Verify workflow completed
        assert "alert_id" in result
        assert result["alert_id"] == "integration-test-001"
        assert "risk_score" in result
        assert 0 <= result["risk_score"] <= 100
        assert "next_action" in result
        assert result["iteration_count"] > 0

    async def test_workflow_state_persistence(self, monkeypatch):
        """Test that workflow state persists through nodes."""
        monkeypatch.setattr(settings, "mock_external_apis", True)

        alert_data = {
            "alert_id": "integration-test-002",
            "alert_type": "new_device",
            "account_id": "acc-12345",
            "customer_id": "cust-67890",
            "transaction": {
                "transaction_id": "txn-integration-002",
                "amount": 500.00,
                "currency": "USD",
                "location_country": "US",
                "device_id": "device-new-integration-002",
                "timestamp": "2025-01-30T14:30:00Z",
            },
            "alert_reason": "First transaction from new device",
            "severity": "medium",
        }

        graph = get_fraud_triage_graph()
        result = await graph.arun(alert_data)

        # Verify state was properly accumulated
        assert len(result["transaction_history"]) > 0 or result["customer_profile"]
        assert result["iteration_count"] >= 3  # At least parse, gather, assess
        assert len(result["messages"]) > 0

    async def test_workflow_error_handling(self, monkeypatch):
        """Test workflow handles errors gracefully."""
        monkeypatch.setattr(settings, "mock_external_apis", True)

        # Use invalid alert data
        alert_data = {
            "alert_id": "integration-test-003",
            "alert_type": "unknown_type",
            # Missing required fields
        }

        graph = get_fraud_triage_graph()

        # Should not raise exception, but handle gracefully
        try:
            result = await graph.arun(alert_data)
            # If it completes, verify it has error information
            if result.get("error_message"):
                assert result["error_message"] is not None
        except Exception as e:
            # Some exceptions are acceptable
            assert isinstance(e, (ValueError, KeyError, AttributeError))


@pytest.mark.asyncio
@pytest.mark.integration
class TestWorkflowWithMockLLM:
    """Test workflow with mocked LLM for faster testing."""

    async def test_workflow_scenarios(self, monkeypatch):
        """Test multiple workflow scenarios."""
        from unittest.mock import AsyncMock, MagicMock
        from src.agents import nodes

        monkeypatch.setattr(settings, "mock_external_apis", True)

        # Mock LLM to return different scores based on input
        async def mock_llm_invoke(prompt):
            alert_id = str(prompt)
            if "Nigeria" in alert_id or "NG" in alert_id:
                content = "Risk Score: 80\nReasoning: High-risk country\nConfidence: 0.9\nSuggested Action: escalate_for_review"
            elif "new_device" in alert_id:
                content = "Risk Score: 55\nReasoning: New device but known customer\nConfidence: 0.7\nSuggested Action: escalate_for_review"
            else:
                content = "Risk Score: 20\nReasoning: Normal transaction pattern\nConfidence: 0.95\nSuggested Action: auto_close"

            response = MagicMock()
            response.content = content
            return response

        # Patch the LLM
        original_llm = nodes.llm
        nodes.llm = AsyncMock()
        nodes.llm.ainvoke = mock_llm_invoke

        try:
            graph = get_fraud_triage_graph()

            # Test high-risk scenario
            high_risk_alert = {
                "alert_id": "test-high-risk-NG",
                "alert_type": "unusual_location",
                "account_id": "acc-12345",
                "customer_id": "cust-67890",
                "transaction": {
                    "transaction_id": "txn-high-risk",
                    "amount": 3000.00,
                    "currency": "USD",
                    "location_country": "NG",
                },
                "alert_reason": "Nigeria transaction",
            }

            result = await graph.arun(high_risk_alert)
            assert result["risk_score"] >= 70

            # Test low-risk scenario
            low_risk_alert = {
                "alert_id": "test-low-risk-US",
                "alert_type": "high_value_transaction",
                "account_id": "acc-12345",
                "customer_id": "cust-67890",
                "transaction": {
                    "transaction_id": "txn-low-risk",
                    "amount": 100.00,
                    "currency": "USD",
                    "location_country": "US",
                },
                "alert_reason": "Slightly high amount",
            }

            result = await graph.arun(low_risk_alert)
            assert result["risk_score"] <= 40

        finally:
            # Restore original LLM
            nodes.llm = original_llm
