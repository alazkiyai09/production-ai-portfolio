"""
Unit tests for context gathering tools.

Tests tool functions for transaction, customer, and device data.
"""

import pytest
from unittest.mock import AsyncMock, patch

from src.tools.transaction_tools import (
    get_transaction_history,
    get_transaction_by_id,
    analyze_transaction_patterns,
)
from src.tools.customer_tools import (
    get_customer_profile,
    get_customer_risk_history,
)
from src.tools.device_tools import (
    get_device_fingerprint,
    check_ip_reputation,
)


@pytest.mark.asyncio
class TestTransactionTools:
    """Test transaction-related tools."""

    async def test_get_transaction_history_mock(self, monkeypatch):
        """Test transaction history with mock data."""
        # Enable mock mode
        from src.config.settings import settings
        monkeypatch.setattr(settings, "mock_external_apis", True)

        result = await get_transaction_history(account_id="acc-123", days=30, limit=10)

        assert "transactions" in result
        assert isinstance(result["transactions"], list)
        assert result["account_id"] == "acc-123"

    async def test_get_transaction_by_id_mock(self, monkeypatch):
        """Test get transaction by ID with mock data."""
        from src.config.settings import settings
        monkeypatch.setattr(settings, "mock_external_apis", True)

        result = await get_transaction_by_id(transaction_id="txn-001")

        assert result["transaction_id"] == "txn-001"
        assert result["amount"] == 1250.00
        assert result["location_country"] == "NG"

    async def test_analyze_transaction_patterns(self, monkeypatch):
        """Test transaction pattern analysis."""
        from src.config.settings import settings
        monkeypatch.setattr(settings, "mock_external_apis", True)

        result = await analyze_transaction_patterns(
            account_id="acc-123",
            transaction_id="txn-001",
        )

        assert "transaction_count" in result
        assert "average_amount" in result
        assert "anomalies" in result
        assert isinstance(result["anomalies"], list)


@pytest.mark.asyncio
class TestCustomerTools:
    """Test customer-related tools."""

    async def test_get_customer_profile_mock(self, monkeypatch):
        """Test customer profile with mock data."""
        from src.config.settings import settings
        monkeypatch.setattr(settings, "mock_external_apis", True)

        result = await get_customer_profile(customer_id="cust-001")

        assert result["customer_id"] == "cust-001"
        assert result["name"] == "John Smith"
        assert result["account_age_years"] == 3
        assert result["kyc_verified"] is True

    async def test_get_customer_risk_history_mock(self, monkeypatch):
        """Test customer risk history with mock data."""
        from src.config.settings import settings
        monkeypatch.setattr(settings, "mock_external_apis", True)

        result = await get_customer_risk_history(customer_id="cust-001", months=12)

        assert "total_alerts" in result
        assert "confirmed_fraud_cases" in result
        assert "average_risk_score" in result
        assert isinstance(result["alerts"], list)


@pytest.mark.asyncio
class TestDeviceTools:
    """Test device and IP tools."""

    async def test_get_device_fingerprint_mock(self, monkeypatch):
        """Test device fingerprint with mock data."""
        from src.config.settings import settings
        monkeypatch.setattr(settings, "mock_external_apis", True)

        result = await get_device_fingerprint(device_id="device-new-123")

        assert result["device_id"] == "device-new-123"
        assert "device_type" in result
        assert "is_new_device" in result
        assert "risk_score" in result

    async def test_check_ip_reputation_mock(self, monkeypatch):
        """Test IP reputation check with mock data."""
        from src.config.settings import settings
        monkeypatch.setattr(settings, "mock_external_apis", True)

        # Test high-risk IP
        result = await check_ip_reputation(ip_address="197.210.53.21")

        assert result["ip_address"] == "197.210.53.21"
        assert result["country"] == "Nigeria"
        assert result["risk_score"] > 50

    async def test_ip_reputation_low_risk(self, monkeypatch):
        """Test IP reputation for low-risk IP."""
        from src.config.settings import settings
        monkeypatch.setattr(settings, "mock_external_apis", True)

        result = await check_ip_reputation(ip_address="8.8.8.8")

        assert result["country"] == "United States"
        assert result["risk_score"] < 50


@pytest.mark.asyncio
class TestToolRegistry:
    """Test tool registry utilities."""

    def test_create_tool_registry(self):
        """Test tool registry creation."""
        from src.tools.utils import create_tool_registry

        registry = create_tool_registry()

        assert isinstance(registry, dict)
        assert "get_transaction_history" in registry
        assert "get_customer_profile" in registry
        assert "get_device_fingerprint" in registry
        assert "check_ip_reputation" in registry

    def test_get_tool_descriptions(self):
        """Test getting tool descriptions."""
        from src.tools.utils import get_tool_descriptions

        descriptions = get_tool_descriptions()

        assert isinstance(descriptions, dict)
        assert all(isinstance(desc, str) for desc in descriptions.values())

    def test_get_tools_by_category(self):
        """Test grouping tools by category."""
        from src.tools.utils import get_tools_by_category

        categories = get_tools_by_category()

        assert "transaction" in categories
        assert "customer" in categories
        assert "device" in categories
        assert len(categories["transaction"]) == 3
        assert len(categories["customer"]) == 2
        assert len(categories["device"]) == 2
