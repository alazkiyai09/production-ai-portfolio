"""
Transaction-related tools for gathering context.

These tools fetch transaction history, analyze patterns,
and provide context for fraud triage.
"""

import functools
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, TypeVar

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.config.settings import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Error Handling Decorator
# =============================================================================

T = TypeVar('T')


def handle_tool_errors(
    default_return: Any = None,
    log_level: str = "error"
) -> Callable[[Callable[..., T]], Callable[..., Any]]:
    """
    Decorator for consistent tool error handling.

    Args:
        default_return: Value to return on error (if None, returns error dict)
        log_level: Logging level for errors (debug, info, warning, error)

    Returns:
        Decorated function with standardized error handling

    Example:
        @handle_tool_errors(default_return={"transactions": []})
        async def get_transaction_history(account_id: str) -> dict:
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except ValueError as e:
                log_msg = f"Validation error in {func.__name__}: {e}"
                getattr(logger, log_level)(log_msg)
                return {
                    "success": False,
                    "error": str(e),
                    "error_type": "validation_error"
                } if default_return is None else default_return
            except ConnectionError as e:
                log_msg = f"Connection error in {func.__name__}: {e}"
                getattr(logger, log_level)(log_msg)
                return {
                    "success": False,
                    "error": "Service temporarily unavailable",
                    "error_type": "connection_error"
                } if default_return is None else default_return
            except httpx.TimeoutException:
                log_msg = f"Timeout error in {func.__name__}"
                getattr(logger, log_level)(log_msg)
                return {
                    "success": False,
                    "error": "Request timed out",
                    "error_type": "timeout_error"
                } if default_return is None else default_return
            except httpx.HTTPStatusError as e:
                log_msg = f"HTTP error in {func.__name__}: {e.response.status_code}"
                getattr(logger, log_level)(log_msg)
                return {
                    "success": False,
                    "error": f"Service error: {e.response.status_code}",
                    "error_type": "http_error",
                    "status_code": e.response.status_code
                } if default_return is None else default_return
            except KeyError as e:
                log_msg = f"Missing key error in {func.__name__}: {e}"
                getattr(logger, log_level)(log_msg)
                return {
                    "success": False,
                    "error": f"Missing required data: {e}",
                    "error_type": "data_error"
                } if default_return is None else default_return
            except Exception as e:
                log_msg = f"Unexpected error in {func.__name__}: {e}"
                getattr(logger, log_level)(log_msg, exc_info=True)
                return {
                    "success": False,
                    "error": "An unexpected error occurred",
                    "error_type": "unknown_error"
                } if default_return is None else default_return

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except ValueError as e:
                log_msg = f"Validation error in {func.__name__}: {e}"
                getattr(logger, log_level)(log_msg)
                return {
                    "success": False,
                    "error": str(e),
                    "error_type": "validation_error"
                } if default_return is None else default_return
            except ConnectionError as e:
                log_msg = f"Connection error in {func.__name__}: {e}"
                getattr(logger, log_level)(log_msg)
                return {
                    "success": False,
                    "error": "Service temporarily unavailable",
                    "error_type": "connection_error"
                } if default_return is None else default_return
            except KeyError as e:
                log_msg = f"Missing key error in {func.__name__}: {e}"
                getattr(logger, log_level)(log_msg)
                return {
                    "success": False,
                    "error": f"Missing required data: {e}",
                    "error_type": "data_error"
                } if default_return is None else default_return
            except Exception as e:
                log_msg = f"Unexpected error in {func.__name__}: {e}"
                getattr(logger, log_level)(log_msg, exc_info=True)
                return {
                    "success": False,
                    "error": "An unexpected error occurred",
                    "error_type": "unknown_error"
                } if default_return is None else default_return

        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


import asyncio


class TransactionHistoryInput(BaseModel):
    """Input schema for transaction history tool."""

    account_id: str = Field(..., description="Customer account ID")
    days: int = Field(default=30, description="Number of days of history to retrieve", ge=1, le=365)
    limit: int = Field(default=50, description="Maximum number of transactions", ge=1, le=500)


class TransactionByIdInput(BaseModel):
    """Input schema for get transaction by ID tool."""

    transaction_id: str = Field(..., description="Transaction ID")


class TransactionPatternInput(BaseModel):
    """Input schema for transaction pattern analysis."""

    account_id: str = Field(..., description="Customer account ID")
    transaction_id: str = Field(..., description="Transaction ID to compare against")


@handle_tool_errors(default_return={"transactions": [], "total_count": 0})
async def get_transaction_history(
    account_id: str,
    days: int = 30,
    limit: int = 50,
) -> dict[str, Any]:
    """
    Get transaction history for an account.

    Args:
        account_id: Customer account ID
        days: Number of days of history to retrieve
        limit: Maximum number of transactions

    Returns:
        Dictionary with transaction history

    Raises:
        ValueError: If account_id is empty or days/limit are invalid
    """
    # Input validation
    if not account_id or not account_id.strip():
        raise ValueError("account_id cannot be empty")

    if not (1 <= days <= 365):
        raise ValueError("days must be between 1 and 365")

    if not (1 <= limit <= 500):
        raise ValueError("limit must be between 1 and 500")

    logger.info(f"Fetching transaction history for account {account_id}")

    if settings.mock_external_apis:
        # Return mock data
        return _mock_transaction_history(account_id, days, limit)

    # Call real API
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{settings.transaction_service_url}/accounts/{account_id}/transactions",
            params={"days": days, "limit": limit},
        )
        response.raise_for_status()
        return response.json()


@handle_tool_errors()
async def get_transaction_by_id(transaction_id: str) -> dict[str, Any]:
    """
    Get a specific transaction by ID.

    Args:
        transaction_id: Transaction ID

    Returns:
        Transaction details

    Raises:
        ValueError: If transaction_id is empty
    """
    if not transaction_id or not transaction_id.strip():
        raise ValueError("transaction_id cannot be empty")

    logger.info(f"Fetching transaction {transaction_id}")

    if settings.mock_external_apis:
        return _mock_transaction(transaction_id)

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{settings.transaction_service_url}/transactions/{transaction_id}",
        )
        response.raise_for_status()
        return response.json()


@handle_tool_errors(default_return={
    "transaction_count": 0,
    "average_amount": 0,
    "max_amount": 0,
    "min_amount": 0,
    "common_locations": [],
    "common_merchants": [],
    "velocity_analysis": {},
    "anomalies": []
})
async def analyze_transaction_patterns(
    account_id: str,
    transaction_id: str,
) -> dict[str, Any]:
    """
    Analyze transaction patterns for an account.

    Compares the specified transaction against historical patterns
    to identify anomalies.

    Args:
        account_id: Customer account ID
        transaction_id: Transaction ID to analyze

    Returns:
        Pattern analysis results

    Raises:
        ValueError: If account_id or transaction_id is empty
    """
    if not account_id or not account_id.strip():
        raise ValueError("account_id cannot be empty")

    if not transaction_id or not transaction_id.strip():
        raise ValueError("transaction_id cannot be empty")

    logger.info(f"Analyzing transaction patterns for account {account_id}, transaction {transaction_id}")

    # Get transaction history
    history_result = await get_transaction_history(account_id, days=90, limit=200)
    transactions = history_result.get("transactions", [])

    # Get current transaction
    current_txn = await get_transaction_by_id(transaction_id)

    if "error" in current_txn or current_txn.get("error_type"):
        raise ValueError(f"Could not fetch current transaction: {current_txn.get('error', 'Unknown error')}")

    # Analyze patterns
    analysis = {
        "transaction_count": len(transactions),
        "average_amount": 0,
        "max_amount": 0,
        "min_amount": 0,
        "common_locations": [],
        "common_merchants": [],
        "velocity_analysis": {},
        "anomalies": [],
    }

    if transactions:
        amounts = [t.get("amount", 0) for t in transactions]
        analysis["average_amount"] = sum(amounts) / len(amounts)
        analysis["max_amount"] = max(amounts)
        analysis["min_amount"] = min(amounts)

        # Check if current transaction is anomalous
        current_amount = current_txn.get("amount", 0)
        if current_amount > analysis["average_amount"] * 3:
            analysis["anomalies"].append({
                "type": "high_amount",
                "description": f"Transaction amount ${current_amount:.2f} is 3x higher than average ${analysis['average_amount']:.2f}"
            })

        # Check velocity (transaction frequency)
        recent_txns = [
            t for t in transactions
            if datetime.fromisoformat(t["timestamp"]) > datetime.now() - timedelta(days=7)
        ]
        analysis["velocity_analysis"] = {
            "last_7_days": len(recent_txns),
            "last_24_hours": len([
                t for t in recent_txns
                if datetime.fromisoformat(t["timestamp"]) > datetime.now() - timedelta(days=1)
            ]),
        }

    return analysis


# Create LangChain tools
get_transaction_history_tool = StructuredTool.from_function(
    coroutine=get_transaction_history,
    name="get_transaction_history",
    description="Get transaction history for a customer account. Use this to gather context about typical transaction patterns.",
    args_schema=TransactionHistoryInput,
)

get_transaction_by_id_tool = StructuredTool.from_function(
    coroutine=get_transaction_by_id,
    name="get_transaction_by_id",
    description="Get details of a specific transaction by ID. Use this to get full details of the flagged transaction.",
    args_schema=TransactionByIdInput,
)

analyze_transaction_patterns_tool = StructuredTool.from_function(
    coroutine=analyze_transaction_patterns,
    name="analyze_transaction_patterns",
    description="Analyze transaction patterns to identify anomalies. Compares the flagged transaction against historical patterns.",
    args_schema=TransactionPatternInput,
)


# Mock data functions
def _mock_transaction_history(
    account_id: str,
    days: int,
    limit: int,
) -> dict[str, Any]:
    """Generate mock transaction history."""
    transactions = []
    base_date = datetime.now()

    for i in range(min(limit, 20)):
        date = base_date - timedelta(days=i % days, hours=i % 24)
        transactions.append({
            "transaction_id": f"txn-{account_id}-{i}",
            "account_id": account_id,
            "amount": 50.0 + (i * 15) + (hash(account_id) % 100),
            "currency": "USD",
            "transaction_type": "purchase" if i % 3 != 0 else "withdrawal",
            "merchant_name": ["Amazon", "Walmart", "Target", "Starbucks", "Shell"][i % 5],
            "merchant_category": ["retail", "grocery", "dining", "fuel"][i % 4],
            "location_city": ["New York", "Los Angeles", "Chicago", "Houston"][i % 4],
            "location_country": "US",
            "timestamp": date.isoformat(),
            "status": "completed",
        })

    return {
        "account_id": account_id,
        "transactions": transactions[:limit],
        "total_count": len(transactions),
    }


def _mock_transaction(transaction_id: str) -> dict[str, Any]:
    """Generate mock transaction."""
    return {
        "transaction_id": transaction_id,
        "account_id": "acc-12345",
        "amount": 1250.00,
        "currency": "USD",
        "transaction_type": "purchase",
        "merchant_name": "Apple Store",
        "merchant_category": "electronics",
        "location_city": "Lagos",
        "location_country": "NG",
        "timestamp": datetime.now().isoformat(),
        "device_id": "device-xyz-123",
        "ip_address": "197.210.53.21",
        "status": "completed",
    }
