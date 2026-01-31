"""
Integration tests for FastAPI endpoints.

Tests the complete API workflow including alert submission,
status checking, and review submission.
"""

import pytest
from httpx import AsyncClient, ASGITransport

from src.api.main import app
from src.config.settings import settings


@pytest.mark.asyncio
@pytest.mark.integration
class TestAlertSubmissionAPI:
    """Test alert submission API endpoints."""

    async def test_root_endpoint(self):
        """Test root endpoint returns API info."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "FraudTriage-Agent API"
        assert "endpoints" in data

    async def test_health_check(self):
        """Test health check endpoint."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    async def test_submit_alert(self, monkeypatch):
        """Test alert submission endpoint."""
        # Enable mock mode
        monkeypatch.setattr(settings, "mock_external_apis", True)

        alert_data = {
            "alert_type": "unusual_location",
            "account_id": "acc-12345",
            "customer_id": "cust-67890",
            "transaction": {
                "transaction_id": "txn-001",
                "amount": 2500.00,
                "currency": "USD",
                "location_country": "NG",
                "device_id": "device-new-123",
                "ip_address": "197.210.53.21",
            },
            "alert_reason": "Transaction from high-risk country",
            "severity": "high",
        }

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/alerts",
                json=alert_data,
            )

        assert response.status_code == 202
        data = response.json()
        assert "alert_id" in data
        assert data["status"] == "pending"

    async def test_get_alert_status(self, monkeypatch):
        """Test getting alert status."""
        monkeypatch.setattr(settings, "mock_external_apis", True)

        # First submit an alert
        alert_data = {
            "alert_type": "high_value_transaction",
            "account_id": "acc-12345",
            "customer_id": "cust-67890",
            "transaction": {
                "transaction_id": "txn-002",
                "amount": 5000.00,
                "currency": "USD",
                "location_country": "US",
            },
            "alert_reason": "High value transaction",
            "severity": "medium",
        }

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            submit_response = await client.post("/api/v1/alerts", json=alert_data)
            alert_id = submit_response.json()["alert_id"]

            # Get status
            status_response = await client.get(f"/api/v1/alerts/{alert_id}/status")

        assert status_response.status_code == 200
        data = status_response.json()
        assert data["alert_id"] == alert_id
        assert "status" in data

    async def test_submit_human_review(self, monkeypatch):
        """Test submitting human review for an alert."""
        monkeypatch.setattr(settings, "mock_external_apis", True)

        # Submit an alert that will require review
        alert_data = {
            "alert_type": "account_takeover",
            "account_id": "acc-12345",
            "customer_id": "cust-67890",
            "transaction": {
                "transaction_id": "txn-003",
                "amount": 10000.00,
                "currency": "USD",
                "location_country": "RU",
                "device_id": "device-unknown-999",
                "ip_address": "185.100.100.100",
            },
            "alert_reason": "Suspicious account takeover attempt",
            "severity": "high",
        }

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Submit alert
            submit_response = await client.post("/api/v1/alerts", json=alert_data)
            alert_id = submit_response.json()["alert_id"]

            # Wait a bit for processing (in real scenario, would poll)
            import asyncio
            await asyncio.sleep(0.5)

            # Simulate that alert requires review
            review_data = {
                "reviewer_id": "analyst-001",
                "reviewer_name": "Jane Smith",
                "decision": "confirm_fraud",
                "reasoning": "Customer confirmed unauthorized transaction attempt",
                "agreed_with_agent": True,
                "correct_risk_score": 90,
            }

            review_response = await client.post(
                f"/api/v1/alerts/{alert_id}/review",
                json=review_data,
            )

        assert review_response.status_code == 200
        data = review_response.json()
        assert data["alert_id"] == alert_id
        assert data["decision"] == "confirm_fraud"

    async def test_list_alerts(self, monkeypatch):
        """Test listing alerts."""
        monkeypatch.setattr(settings, "mock_external_apis", True)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/alerts")

        assert response.status_code == 200
        data = response.json()
        assert "alerts" in data
        assert "total" in data
        assert isinstance(data["alerts"], list)

    async def test_list_alerts_with_filter(self, monkeypatch):
        """Test listing alerts with status filter."""
        monkeypatch.setattr(settings, "mock_external_apis", True)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/alerts?status=pending")

        assert response.status_code == 200
        data = response.json()
        assert "alerts" in data

    async def test_get_nonexistent_alert(self):
        """Test getting a non-existent alert returns 404."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/alerts/nonexistent-alert")

        assert response.status_code == 404
