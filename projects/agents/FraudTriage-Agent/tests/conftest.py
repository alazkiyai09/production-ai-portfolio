"""
Pytest configuration and shared fixtures.

This module provides common fixtures used across tests.
"""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import settings


@pytest.fixture(autouse=True)
def reset_settings():
    """Reset settings before each test."""
    # Clear settings cache
    from src.config.settings import get_settings
    get_settings.cache_clear()


@pytest.fixture
def mock_llm(monkeypatch):
    """Mock LLM for testing."""
    mock_response = MagicMock()
    mock_response.content = """
    Risk Score: 65
    Risk Factors:
    - Unusual location (Nigeria)
    - High transaction amount
    - New device detected
    Confidence: 0.8
    Suggested Action: escalate_for_review
    """

    mock_llm_instance = AsyncMock()
    mock_llm_instance.ainvoke.return_value = mock_response

    # Patch the LLM creation
    monkeypatch.setattr("src.agents.nodes.create_llm", lambda: mock_llm_instance)

    return mock_llm_instance


@pytest.fixture
def sample_alert():
    """Sample fraud alert for testing."""
    return {
        "alert_id": "test-alert-001",
        "alert_type": "unusual_location",
        "account_id": "acc-12345",
        "customer_id": "cust-67890",
        "transaction": {
            "transaction_id": "txn-001",
            "amount": 2500.00,
            "currency": "USD",
            "merchant_name": "Luxury Goods Store",
            "location_country": "NG",
            "location_city": "Lagos",
            "device_id": "device-new-123",
            "ip_address": "197.210.53.21",
            "timestamp": "2025-01-30T10:30:00Z",
        },
        "rule_id": "RULE-001",
        "alert_reason": "Transaction from high-risk country",
        "severity": "high",
    }


@pytest.fixture
def sample_alert_low_risk():
    """Sample low-risk alert for testing."""
    return {
        "alert_id": "test-alert-002",
        "alert_type": "high_value_transaction",
        "account_id": "acc-12345",
        "customer_id": "cust-67890",
        "transaction": {
            "transaction_id": "txn-002",
            "amount": 150.00,
            "currency": "USD",
            "merchant_name": "Amazon",
            "location_country": "US",
            "location_city": "New York",
            "device_id": "device-known-456",
            "ip_address": "8.8.8.8",
            "timestamp": "2025-01-30T10:30:00Z",
        },
        "rule_id": "RULE-002",
        "alert_reason": "Transaction above threshold",
        "severity": "low",
    }
