"""
Customer-related tools for gathering context.

These tools fetch customer profiles, risk history,
and account information for fraud triage.
"""

import logging
from typing import Any

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.config.settings import settings

logger = logging.getLogger(__name__)


class CustomerProfileInput(BaseModel):
    """Input schema for customer profile tool."""

    customer_id: str = Field(..., description="Customer ID")


class CustomerRiskHistoryInput(BaseModel):
    """Input schema for customer risk history tool."""

    customer_id: str = Field(..., description="Customer ID")
    months: int = Field(default=12, description="Number of months of history", ge=1, le=60)


async def get_customer_profile(customer_id: str) -> dict[str, Any]:
    """
    Get customer profile information.

    Args:
        customer_id: Customer ID

    Returns:
        Customer profile data
    """
    logger.info(f"Fetching customer profile for {customer_id}")

    if settings.mock_external_apis:
        return _mock_customer_profile(customer_id)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{settings.customer_service_url}/customers/{customer_id}",
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Error fetching customer profile: {e}")
        return {"error": str(e)}


async def get_customer_risk_history(
    customer_id: str,
    months: int = 12,
) -> dict[str, Any]:
    """
    Get customer risk history including previous fraud alerts.

    Args:
        customer_id: Customer ID
        months: Number of months of history

    Returns:
        Risk history data
    """
    logger.info(f"Fetching risk history for customer {customer_id}")

    if settings.mock_external_apis:
        return _mock_risk_history(customer_id, months)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{settings.customer_service_url}/customers/{customer_id}/risk-history",
                params={"months": months},
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Error fetching risk history: {e}")
        return {"error": str(e), "alerts": [], "total_risk_score": 0}


# LangChain tools
get_customer_profile_tool = StructuredTool.from_function(
    coroutine=get_customer_profile,
    name="get_customer_profile",
    description="Get customer profile information including account details, demographics, and account age.",
    args_schema=CustomerProfileInput,
)

get_customer_risk_history_tool = StructuredTool.from_function(
    coroutine=get_customer_risk_history,
    name="get_customer_risk_history",
    description="Get customer risk history including previous fraud alerts and risk scores. Use this to check if the customer has a history of fraud.",
    args_schema=CustomerRiskHistoryInput,
)


# Mock data functions
def _mock_customer_profile(customer_id: str) -> dict[str, Any]:
    """Generate mock customer profile."""
    return {
        "customer_id": customer_id,
        "account_id": "acc-12345",
        "name": "John Smith",
        "email": "john.smith@example.com",
        "phone": "+1-555-0123",
        "date_of_birth": "1985-06-15",
        "account_created_date": "2021-03-10",
        "account_age_years": 3,
        "account_status": "active",
        "customer_segment": "retail",
        "kyc_verified": True,
        "risk_level": "low",
        "addresses": [
            {
                "type": "primary",
                "street": "123 Main St",
                "city": "New York",
                "state": "NY",
                "country": "US",
                "postal_code": "10001",
            }
        ],
        "registered_devices": [
            {"device_id": "device-abc-123", "device_type": "mobile", "last_seen": "2025-01-28"},
            {"device_id": "device-def-456", "device_type": "desktop", "last_seen": "2025-01-25"},
        ],
    }


def _mock_risk_history(customer_id: str, months: int) -> dict[str, Any]:
    """Generate mock risk history."""
    # Generate varying history based on customer_id hash
    num_alerts = hash(customer_id) % 5

    return {
        "customer_id": customer_id,
        "period_months": months,
        "total_alerts": num_alerts,
        "confirmed_fraud_cases": num_alerts // 3,
        "false_positives": num_alerts - (num_alerts // 3),
        "average_risk_score": 25 + (hash(customer_id) % 30),
        "alerts": [
            {
                "alert_id": f"alert-{i}",
                "date": "2024-12-01",
                "risk_score": 40 + (i * 15),
                "outcome": "false_positive" if i % 2 == 0 else "confirmed_fraud",
            }
            for i in range(num_alerts)
        ],
    }
