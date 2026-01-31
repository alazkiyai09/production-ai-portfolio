"""
Fraud detection tools for the FraudTriage-Agent.

This module provides LangChain tools for gathering fraud-relevant data
including customer profiles, transaction history, watchlist checks,
risk scoring, and similar alert detection.

All tools use the @tool decorator and include realistic mock data
representing common fraud scenarios in banking.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Mock Data - Customer Profiles
# =============================================================================

_MOCK_CUSTOMERS = {
    "CUST-001": {
        "customer_id": "CUST-001",
        "name": "John Smith",
        "account_id": "ACC-001",
        "account_age_months": 48,  # 4 years
        "verification_status": "verified",
        "risk_rating": "low",
        "total_transactions_ytd": 127,
        "average_transaction": 285.00,
        "country": "US",
        "occupation": "Software Engineer",
        "segment": "retail",
        "kyc_completed": True,
        "previous_fraud_cases": 0,
        "false_positive_count": 2,
        "registered_devices": ["DEVICE-001-APPLE", "DEVICE-001-WINDOWS"],
        "typical_countries": ["US", "CA", "UK"],
    },
    "CUST-002": {
        "customer_id": "CUST-002",
        "name": "Maria Garcia",
        "account_id": "ACC-002",
        "account_age_months": 6,  # New account
        "verification_status": "pending",
        "risk_rating": "medium",
        "total_transactions_ytd": 12,
        "average_transaction": 150.00,
        "country": "ES",
        "occupation": "Teacher",
        "segment": "retail",
        "kyc_completed": False,
        "previous_fraud_cases": 0,
        "false_positive_count": 0,
        "registered_devices": ["DEVICE-002-MOBILE"],
        "typical_countries": ["ES", "PT", "FR"],
    },
    "CUST-003": {
        "customer_id": "CUST-003",
        "name": "Chen Wei",
        "account_id": "ACC-003",
        "account_age_months": 120,  # 10 years
        "verification_status": "verified",
        "risk_rating": "low",
        "total_transactions_ytd": 342,
        "average_transaction": 520.00,
        "country": "SG",
        "occupation": "Business Manager",
        "segment": "premium",
        "kyc_completed": True,
        "previous_fraud_cases": 1,  # Was a victim of fraud
        "false_positive_count": 5,
        "registered_devices": ["DEVICE-003-IOS", "DEVICE-003-ANDROID", "DEVICE-003-LAPTOP"],
        "typical_countries": ["SG", "MY", "TH", "JP", "US"],
    },
    "CUST-004": {
        "customer_id": "CUST-004",
        "name": "Ahmed Hassan",
        "account_id": "ACC-004",
        "account_age_months": 2,  # Very new account
        "verification_status": "not_verified",
        "risk_rating": "high",
        "total_transactions_ytd": 3,
        "average_transaction": 75.00,
        "country": "AE",
        "occupation": "Unknown",
        "segment": "retail",
        "kyc_completed": False,
        "previous_fraud_cases": 0,
        "false_positive_count": 0,
        "registered_devices": [],
        "typical_countries": ["AE"],
    },
    "CUST-005": {
        "customer_id": "CUST-005",
        "name": "Sarah Johnson",
        "account_id": "ACC-005",
        "account_age_months": 36,
        "verification_status": "verified",
        "risk_rating": "low",
        "total_transactions_ytd": 89,
        "average_transaction": 1950.00,  # High-value customer
        "country": "US",
        "occupation": "Doctor",
        "segment": "premium",
        "kyc_completed": True,
        "previous_fraud_cases": 0,
        "false_positive_count": 1,
        "registered_devices": ["DEVICE-005-IPHONE", "DEVICE-005-MACBOOK"],
        "typical_countries": ["US", "CA", "GB", "FR", "IT"],
    },
}


# =============================================================================
# Mock Data - Transaction History
# =============================================================================

def _generate_mock_transactions(customer_id: str, days: int = 30) -> list[dict[str, Any]]:
    """
    Generate mock transaction history for a customer.

    Args:
        customer_id: Customer identifier
        days: Number of days of history to generate

    Returns:
        List of transaction dictionaries
    """
    customer = _MOCK_CUSTOMERS.get(customer_id)
    if not customer:
        return []

    transactions = []
    base_date = datetime.now()
    avg_amount = customer.get("average_transaction", 200)
    typical_countries = customer.get("typical_countries", ["US"])

    # Generate realistic transaction patterns
    merchants = [
        "Amazon", "Walmart", "Target", "Starbucks", "Shell", "Costco",
        "Apple Store", "Netflix", "Uber", "Airbnb", "Delta Airlines",
        "Whole Foods", "Best Buy", "Walgreens", "Home Depot"
    ]

    # Number of transactions based on customer profile
    num_transactions = min(
        int(customer.get("total_transactions_ytd", 50) * (days / 365)),
        100  # Cap at 100 for performance
    )

    for i in range(num_transactions):
        # Vary amount around average (±50%)
        import random
        random.seed(hash(f"{customer_id}-{i}"))  # Consistent seed
        amount = avg_amount * (0.5 + random.random())

        # Random date within range
        days_ago = random.randint(0, days)
        txn_date = base_date - timedelta(days=days_ago, hours=random.randint(0, 23))

        # Random merchant from list
        merchant = random.choice(merchants)

        # Random location (mostly typical countries)
        if random.random() < 0.9:  # 90% from typical countries
            location = random.choice(typical_countries)
        else:  # 10% from other countries (potential anomalies)
            location = random.choice(["NG", "RU", "UA", "BR", "IN"])

        transaction = {
            "transaction_id": f"TXN-{customer_id}-{i:05d}",
            "customer_id": customer_id,
            "account_id": customer.get("account_id"),
            "date": txn_date.strftime("%Y-%m-%d"),
            "timestamp": txn_date.isoformat(),
            "amount": round(amount, 2),
            "currency": "USD",
            "merchant": merchant,
            "category": _get_merchant_category(merchant),
            "status": "completed",
            "location": location,
            "device_id": random.choice(customer.get("registered_devices", ["DEVICE-UNKNOWN"])),
        }

        transactions.append(transaction)

    # Sort by date descending (newest first)
    transactions.sort(key=lambda x: x["timestamp"], reverse=True)

    return transactions


def _get_merchant_category(merchant: str) -> str:
    """Get merchant category code."""
    categories = {
        "Amazon": "retail",
        "Walmart": "retail",
        "Target": "retail",
        "Starbucks": "dining",
        "Shell": "fuel",
        "Costco": "retail",
        "Apple Store": "electronics",
        "Netflix": "entertainment",
        "Uber": "transport",
        "Airbnb": "travel",
        "Delta Airlines": "travel",
        "Whole Foods": "grocery",
        "Best Buy": "electronics",
        "Walgreens": "pharmacy",
        "Home Depot": "home_improvement",
    }
    return categories.get(merchant, "other")


# =============================================================================
# Mock Data - Watchlists
# =============================================================================

_MOCK_WATCHLIST_HITS = {
    "CUST-SUSPICIOUS-001": [
        {
            "list_name": "Internal Fraud Database",
            "match_type": "exact",
            "match_confidence": 0.95,
            "details": {
                "reason": "Previous confirmed fraud case",
                "case_date": "2024-08-15",
                "case_id": "FRAUD-2024-0847",
            }
        }
    ],
    "CUST-SANCTIONS-001": [
        {
            "list_name": "OFAC SDN List",
            "match_type": "partial",
            "match_confidence": 0.75,
            "details": {
                "reason": "Name similarity to sanctioned individual",
                "sanctions_list": "Specially Designated Nationals (SDN)",
                "country": "IR",
            }
        }
    ],
}


# =============================================================================
# Mock Data - Similar Alerts
# =============================================================================

_MOCK_SIMILAR_ALERTS = {
    "location_mismatch": [
        {
            "alert_id": "ALERT-2024-0891",
            "date": "2024-11-15",
            "customer_id": "CUST-001",
            "outcome": "false_positive",
            "reason": "Customer was traveling",
            "risk_score": 65,
            "similarity": 0.85,
        },
        {
            "alert_id": "ALERT-2024-1023",
            "date": "2024-12-03",
            "customer_id": "CUST-007",
            "outcome": "confirmed_fraud",
            "reason": "Account takeover, customer confirmed unauthorized",
            "risk_score": 88,
            "similarity": 0.72,
        },
        {
            "alert_id": "ALERT-2024-1156",
            "date": "2024-12-20",
            "customer_id": "CUST-012",
            "outcome": "false_positive",
            "reason": "Customer moved to new country",
            "risk_score": 55,
            "similarity": 0.68,
        },
    ],
    "unusual_amount": [
        {
            "alert_id": "ALERT-2024-0782",
            "date": "2024-10-28",
            "customer_id": "CUST-003",
            "outcome": "false_positive",
            "reason": "Large purchase customer confirmed",
            "risk_score": 45,
            "similarity": 0.90,
        },
        {
            "alert_id": "ALERT-2024-0945",
            "date": "2024-11-22",
            "customer_id": "CUST-015",
            "outcome": "confirmed_fraud",
            "reason": "Card testing followed by large transaction",
            "risk_score": 92,
            "similarity": 0.78,
        },
    ],
    "velocity": [
        {
            "alert_id": "ALERT-2024-0812",
            "date": "2024-11-02",
            "customer_id": "CUST-004",
            "outcome": "confirmed_fraud",
            "reason": "Bot attack, automated high-velocity transactions",
            "risk_score": 95,
            "similarity": 0.88,
        },
        {
            "alert_id": "ALERT-2024-1088",
            "date": "2024-12-08",
            "customer_id": "CUST-008",
            "outcome": "false_positive",
            "reason": "Customer doing holiday shopping",
            "risk_score": 35,
            "similarity": 0.82,
        },
    ],
    "device_change": [
        {
            "alert_id": "ALERT-2024-0923",
            "date": "2024-11-18",
            "customer_id": "CUST-002",
            "outcome": "confirmed_fraud",
            "reason": "New device from high-risk IP, account takeover",
            "risk_score": 85,
            "similarity": 0.80,
        },
        {
            "alert_id": "ALERT-2024-1099",
            "date": "2024-12-10",
            "customer_id": "CUST-005",
            "outcome": "false_positive",
            "reason": "Customer upgraded phone",
            "risk_score": 25,
            "similarity": 0.75,
        },
    ],
}


# =============================================================================
# Tool 1: Get Customer Profile
# =============================================================================

@tool
def get_customer_profile(customer_id: str) -> dict[str, Any]:
    """
    Retrieve customer profile information including account details and risk indicators.

    This tool fetches comprehensive customer data used for fraud risk assessment,
    including account age, verification status, transaction patterns, and risk rating.

    Args:
        customer_id: Unique customer identifier (e.g., "CUST-001")

    Returns:
        Dictionary containing:
            - customer_id: Customer unique identifier
            - name: Customer's full name
            - account_id: Associated account identifier
            - account_age_months: Age of account in months
            - verification_status: KYC verification status (verified/pending/not_verified)
            - risk_rating: Customer risk level (low/medium/high)
            - total_transactions_ytd: Total transactions year-to-date
            - average_transaction: Average transaction amount
            - country: Customer's home country
            - occupation: Customer's occupation
            - segment: Customer segment (retail/premium)
            - kyc_completed: Whether KYC is complete
            - previous_fraud_cases: Number of confirmed fraud cases
            - false_positive_count: Number of false positive alerts
            - registered_devices: List of known device IDs
            - typical_countries: List of typical transaction countries

    Example:
        >>> profile = get_customer_profile("CUST-001")
        >>> print(profile["name"])
        John Smith
        >>> print(profile["risk_rating"])
        low
    """
    logger.info(f"Fetching customer profile for: {customer_id}")

    # Get customer from mock data
    customer = _MOCK_CUSTOMERS.get(customer_id)

    if not customer:
        logger.warning(f"Customer not found: {customer_id}")
        return {
            "error": f"Customer {customer_id} not found",
            "customer_id": customer_id,
        }

    logger.info(
        f"Retrieved profile for {customer['name']}: "
        f"risk={customer['risk_rating']}, "
        f"account_age={customer['account_age_months']} months"
    )

    return customer


# =============================================================================
# Tool 2: Get Transaction History
# =============================================================================

@tool
def get_transaction_history(customer_id: str, days: int = 30) -> list[dict[str, Any]]:
    """
    Retrieve transaction history for a customer over a specified time period.

    This tool provides recent transaction data used for pattern analysis,
    velocity checks, and deviation detection in fraud assessment.

    Args:
        customer_id: Unique customer identifier (e.g., "CUST-001")
        days: Number of days of history to retrieve (default: 30, max: 90)

    Returns:
        List of transaction dictionaries, each containing:
            - transaction_id: Unique transaction identifier
            - customer_id: Customer identifier
            - account_id: Account identifier
            - date: Transaction date (YYYY-MM-DD)
            - timestamp: Full ISO timestamp
            - amount: Transaction amount in USD
            - currency: Currency code (USD)
            - merchant: Merchant name
            - category: Merchant category code
            - status: Transaction status (completed/pending/failed)
            - location: Transaction country code
            - device_id: Device identifier used for transaction

    Example:
        >>> history = get_transaction_history("CUST-001", days=30)
        >>> print(f"Retrieved {len(history)} transactions")
        >>> for txn in history[:5]:
        ...     print(f"{txn['date']}: {txn['merchant']} - ${txn['amount']}")
    """
    logger.info(f"Fetching transaction history for {customer_id}: last {days} days")

    # Validate days parameter
    if days < 1:
        days = 1
    elif days > 90:
        days = 90
        logger.warning("Days parameter capped at 90")

    # Generate mock transactions
    transactions = _generate_mock_transactions(customer_id, days)

    logger.info(f"Retrieved {len(transactions)} transactions for {customer_id}")

    return transactions


# =============================================================================
# Tool 3: Check Watchlists
# =============================================================================

@tool
def check_watchlists(customer_id: str, transaction_details: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Check customer and transaction against internal and external watchlists.

    This tool screens against OFAC sanctions lists, internal fraud databases,
    and money mule patterns to identify high-risk entities.

    Args:
        customer_id: Unique customer identifier (e.g., "CUST-001")
        transaction_details: Transaction details for screening:
            - amount: Transaction amount
            - country: Transaction country code
            - beneficiary_name: Payee/beneficiary name (optional)
            - beneficiary_account: Beneficiary account number (optional)

    Returns:
        List of watchlist hits. Each hit contains:
            - list_name: Name of the watchlist
            - match_type: Type of match (exact/partial/fuzzy)
            - match_confidence: Confidence score (0.0-1.0)
            - details: Additional details about the match

        Returns empty list if no matches found.

    Example:
        >>> txn = {"amount": 5000, "country": "IR", "beneficiary_name": "Suspicious Entity"}
        >>> hits = check_watchlists("CUST-001", txn)
        >>> if hits:
        ...     for hit in hits:
        ...         print(f"Match on {hit['list_name']}: {hit['match_confidence']}")
    """
    logger.info(f"Checking watchlists for customer: {customer_id}")

    hits = []

    # Check internal fraud database
    if customer_id in _MOCK_WATCHLIST_HITS:
        customer_hits = _MOCK_WATCHLIST_HITS[customer_id]
        for hit in customer_hits:
            if "Internal" in hit["list_name"] or "Fraud" in hit["list_name"]:
                hits.append(hit)
                logger.warning(f"Internal fraud database hit for {customer_id}")

    # Check transaction details for high-risk patterns
    country = transaction_details.get("country", "")
    amount = transaction_details.get("amount", 0)
    beneficiary = transaction_details.get("beneficiary_name", "")

    # High-risk countries check (mock OFAC)
    high_risk_countries = ["IR", "KP", "SY", "CU", "MM", "RU", "BY"]
    if country in high_risk_countries:
        hits.append({
            "list_name": "OFAC Sanctions List (Country-Based)",
            "match_type": "partial",
            "match_confidence": 0.60,
            "details": {
                "reason": f"Transaction to/from high-risk country: {country}",
                "country": country,
                "sanctions_program": "Country-Based Sanctions",
            }
        })
        logger.warning(f"High-risk country transaction: {country}")

    # Large amount check (potential money laundering)
    if amount > 10000:
        hits.append({
            "list_name": "Internal AML Monitoring",
            "match_type": "pattern",
            "match_confidence": 0.50,
            "details": {
                "reason": f"Large transaction amount: ${amount:.2f}",
                "threshold": "$10,000",
                "regulation": "Bank Secrecy Act (BSA) reporting threshold",
            }
        })
        logger.info(f"Large transaction flagged: ${amount:.2f}")

    # Check beneficiary name against suspicious patterns (mock)
    suspicious_keywords = ["unknown", "suspicious", "anonymous", "crypto exchange"]
    if beneficiary and any(keyword in beneficiary.lower() for keyword in suspicious_keywords):
        hits.append({
            "list_name": "Internal Suspicious Beneficiary List",
            "match_type": "fuzzy",
            "match_confidence": 0.45,
            "details": {
                "reason": f"Suspicious beneficiary name pattern: {beneficiary}",
                "pattern_match": next(keyword for keyword in suspicious_keywords if keyword in beneficiary.lower()),
            }
        })
        logger.warning(f"Suspicious beneficiary: {beneficiary}")

    if hits:
        logger.warning(f"Found {len(hits)} watchlist hit(s) for {customer_id}")
    else:
        logger.info(f"No watchlist hits for {customer_id}")

    return hits


# =============================================================================
# Tool 4: Calculate Risk Score
# =============================================================================

@tool
def calculate_risk_score(
    customer_profile: dict[str, Any],
    transaction_amount: float,
    transaction_history: list[dict[str, Any]],
    alert_type: str,
) -> dict[str, Any]:
    """
    Calculate comprehensive risk score based on multiple fraud indicators.

    This tool analyzes customer profile, transaction patterns, and alert type
    to produce a risk score (0.0-1.0) with detailed contributing factors.

    Risk Score Factors:
        - New account (< 3 months): +0.3
        - Transaction amount > 5x average: +0.3
        - Unverified customer: +0.2
        - High-risk alert type: +0.2
        - Watchlist/sanctions hits: +0.4 per hit
        - No transaction history: +0.15
        - High-risk location: +0.1
        - Unusual velocity: +0.15

    Risk Levels:
        - 0.0 - 0.25: LOW
        - 0.26 - 0.50: MEDIUM
        - 0.51 - 0.75: HIGH
        - 0.76 - 1.00: CRITICAL

    Args:
        customer_profile: Customer profile dictionary from get_customer_profile
        transaction_amount: Amount of the current transaction being assessed
        transaction_history: List of historical transactions from get_transaction_history
        alert_type: Type of alert that triggered this assessment

    Returns:
        Dictionary containing:
            - score: Final risk score (0.0-1.0)
            - level: Risk level category (low/medium/high/critical)
            - factors: List of identified risk factors with their contributions
            - confidence: Confidence in the assessment (0.0-1.0)
            - recommendations: List of recommended actions

    Example:
        >>> profile = get_customer_profile("CUST-001")
        >>> history = get_transaction_history("CUST-001", days=30)
        >>> risk = calculate_risk_score(profile, 5000, history, "unusual_amount")
        >>> print(f"Risk Score: {risk['score']:.2f} ({risk['level']})")
        >>> for factor in risk['factors']:
        ...     print(f"  - {factor['reason']}: +{factor['contribution']}")
    """
    logger.info("Calculating risk score")

    risk_score = 0.0
    factors = []
    recommendations = []

    # Factor 1: Account age
    account_age_months = customer_profile.get("account_age_months", 0)
    if account_age_months < 3:
        risk_score += 0.30
        factors.append({
            "reason": f"Very new account ({account_age_months} months old)",
            "contribution": 0.30,
            "data_point": f"account_age_months={account_age_months}"
        })
        recommendations.append("Enhanced monitoring recommended for new account")
    elif account_age_months < 12:
        risk_score += 0.10
        factors.append({
            "reason": f"Relatively new account ({account_age_months} months old)",
            "contribution": 0.10,
            "data_point": f"account_age_months={account_age_months}"
        })

    # Factor 2: Transaction amount deviation
    average_transaction = customer_profile.get("average_transaction", transaction_amount)
    if average_transaction > 0:
        deviation_ratio = transaction_amount / average_transaction
        if deviation_ratio > 10:
            risk_score += 0.40
            factors.append({
                "reason": f"Transaction amount {deviation_ratio:.1f}x higher than average",
                "contribution": 0.40,
                "data_point": f"ratio={deviation_ratio:.1f}"
            })
            recommendations.append("Verify transaction legitimacy with customer")
        elif deviation_ratio > 5:
            risk_score += 0.30
            factors.append({
                "reason": f"Transaction amount {deviation_ratio:.1f}x higher than average",
                "contribution": 0.30,
                "data_point": f"ratio={deviation_ratio:.1f}"
            })
            recommendations.append("Consider customer contact for verification")
        elif deviation_ratio > 3:
            risk_score += 0.15
            factors.append({
                "reason": f"Transaction amount {deviation_ratio:.1f}x higher than average",
                "contribution": 0.15,
                "data_point": f"ratio={deviation_ratio:.1f}"
            })

    # Factor 3: Verification status
    verification_status = customer_profile.get("verification_status", "unknown")
    if verification_status == "not_verified":
        risk_score += 0.20
        factors.append({
            "reason": "Customer KYC not verified",
            "contribution": 0.20,
            "data_point": f"verification_status={verification_status}"
        })
        recommendations.append("Require KYC verification")
    elif verification_status == "pending":
        risk_score += 0.10
        factors.append({
            "reason": "Customer KYC verification pending",
            "contribution": 0.10,
            "data_point": f"verification_status={verification_status}"
        })

    # Factor 4: Alert type risk
    high_risk_alerts = ["account_takeover", "velocity", "device_change"]
    medium_risk_alerts = ["location_mismatch", "unusual_amount"]

    if alert_type in high_risk_alerts:
        risk_score += 0.25
        factors.append({
            "reason": f"High-risk alert type: {alert_type}",
            "contribution": 0.25,
            "data_point": f"alert_type={alert_type}"
        })
    elif alert_type in medium_risk_alerts:
        risk_score += 0.15
        factors.append({
            "reason": f"Medium-risk alert type: {alert_type}",
            "contribution": 0.15,
            "data_point": f"alert_type={alert_type}"
        })

    # Factor 5: Transaction history
    if not transaction_history or len(transaction_history) == 0:
        risk_score += 0.15
        factors.append({
            "reason": "No transaction history available",
            "contribution": 0.15,
            "data_point": "history_length=0"
        })
    else:
        # Check for velocity (many recent transactions)
        recent_txns = [t for t in transaction_history if t.get("date")]
        if len(recent_txns) > 20:  # More than 20 transactions in period
            risk_score += 0.15
            factors.append({
                "reason": f"High transaction velocity: {len(recent_txns)} transactions",
                "contribution": 0.15,
                "data_point": f"transaction_count={len(recent_txns)}"
            })
            recommendations.append("Monitor for possible bot activity or money mule")

    # Factor 6: Previous fraud history
    previous_fraud = customer_profile.get("previous_fraud_cases", 0)
    if previous_fraud > 0:
        risk_score += 0.10
        factors.append({
            "reason": f"Customer has {previous_fraud} previous fraud case(s)",
            "contribution": 0.10,
            "data_point": f"previous_fraud={previous_fraud}"
        })
        recommendations.append("Customer was previously a fraud victim - enhanced caution")

    # Factor 7: Risk rating from profile
    profile_risk = customer_profile.get("risk_rating", "low")
    if profile_risk == "high":
        risk_score += 0.20
        factors.append({
            "reason": "Customer profile marked as high risk",
            "contribution": 0.20,
            "data_point": f"risk_rating={profile_risk}"
        })
    elif profile_risk == "medium":
        risk_score += 0.10
        factors.append({
            "reason": "Customer profile marked as medium risk",
            "contribution": 0.10,
            "data_point": f"risk_rating={profile_risk}"
        })

    # Cap score at 1.0
    risk_score = min(risk_score, 1.0)

    # Determine risk level
    if risk_score <= 0.25:
        level = "low"
    elif risk_score <= 0.50:
        level = "medium"
    elif risk_score <= 0.75:
        level = "high"
    else:
        level = "critical"

    # Calculate confidence based on data completeness
    has_profile = bool(customer_profile and not customer_profile.get("error"))
    has_history = bool(transaction_history and len(transaction_history) > 0)
    confidence = 0.5 + (0.3 if has_profile else 0) + (0.2 if has_history else 0)

    # Add default recommendations if none
    if not recommendations:
        if level == "low":
            recommendations.append("Standard monitoring - no action required")
        elif level == "medium":
            recommendations.append("Continue monitoring - may require investigation")
        else:
            recommendations.append("Immediate review recommended")

    result = {
        "score": round(risk_score, 3),
        "level": level,
        "factors": factors,
        "confidence": round(confidence, 2),
        "recommendations": recommendations,
        "customer_base_risk": profile_risk,
        "data_points": {
            "account_age_months": account_age_months,
            "average_transaction": average_transaction,
            "transaction_count": len(transaction_history) if transaction_history else 0,
            "verification_status": verification_status,
        }
    }

    logger.info(
        f"Risk score calculated: {result['score']:.3f} ({level}), "
        f"confidence: {confidence:.2f}, factors: {len(factors)}"
    )

    return result


# =============================================================================
# Tool 5: Get Similar Alerts
# =============================================================================

@tool
def get_similar_alerts(alert_type: str, customer_id: str) -> list[dict[str, Any]]:
    """
    Find historical alerts with similar patterns for comparison and outcome analysis.

    This tool searches for past alerts of the same type to help assess the current
    alert based on historical outcomes (false positives vs confirmed fraud).

    Args:
        alert_type: Type of alert (e.g., "location_mismatch", "unusual_amount")
        customer_id: Customer identifier (used to find similar customer profiles)

    Returns:
        List of similar alert dictionaries, each containing:
            - alert_id: Unique alert identifier
            - date: Alert creation date
            - customer_id: Customer ID for the historical alert
            - outcome: Alert outcome (false_positive/confirmed_fraud/pending)
            - reason: Explanation of the outcome
            - risk_score: Risk score that was assigned
            - similarity: Similarity score to current alert (0.0-1.0)

    Example:
        >>> similar = get_similar_alerts("location_mismatch", "CUST-001")
        >>> for alert in similar:
        ...     print(f"{alert['alert_id']}: {alert['outcome']} (similarity: {alert['similarity']})")
    """
    logger.info(f"Searching for similar alerts: type={alert_type}, customer={customer_id}")

    # Normalize alert type
    alert_type_normalized = alert_type.lower().replace("-", "_")

    # Get similar alerts from mock data
    similar_alerts = _MOCK_SIMILAR_ALERTS.get(alert_type_normalized, [])

    if not similar_alerts:
        logger.info(f"No similar alerts found for type: {alert_type}")
        return []

    # Filter by similarity threshold and sort
    filtered_alerts = [a for a in similar_alerts if a.get("similarity", 0) >= 0.60]
    filtered_alerts.sort(key=lambda x: x.get("similarity", 0), reverse=True)

    # Limit to top 5 most similar
    result = filtered_alerts[:5]

    logger.info(f"Found {len(result)} similar alerts for {alert_type}")

    # Calculate outcome statistics
    if result:
        fraud_count = sum(1 for a in result if a.get("outcome") == "confirmed_fraud")
        false_positive_count = sum(1 for a in result if a.get("outcome") == "false_positive")
        logger.info(
            f"Historical outcomes: {fraud_count} confirmed fraud, "
            f"{false_positive_count} false positives"
        )

    return result


# =============================================================================
# Tool Registry
# =============================================================================

def get_all_fraud_tools() -> list:
    """
    Get all fraud detection tools as a list.

    Returns:
        List of LangChain tool functions
    """
    return [
        get_customer_profile,
        get_transaction_history,
        check_watchlists,
        calculate_risk_score,
        get_similar_alerts,
    ]


def get_tool_descriptions() -> dict[str, str]:
    """
    Get descriptions of all fraud tools.

    Returns:
        Dictionary mapping tool names to descriptions
    """
    tools = get_all_fraud_tools()
    return {tool.name: tool.description for tool in tools}
