#!/usr/bin/env python3
"""
Test script for fraud detection tools.

This script demonstrates all the fraud tools with realistic scenarios.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.fraud_tools import (
    get_customer_profile,
    get_transaction_history,
    check_watchlists,
    calculate_risk_score,
    get_similar_alerts,
    get_tool_descriptions,
)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def test_customer_profiles():
    """Test customer profile retrieval."""
    print_section("TEST 1: Customer Profile Retrieval")

    # Test different customers
    customers = ["CUST-001", "CUST-002", "CUST-004"]

    for customer_id in customers:
        print(f"Customer: {customer_id}")
        profile = get_customer_profile.invoke({"customer_id": customer_id})

        if "error" in profile:
            print(f"  ❌ Error: {profile['error']}\n")
            continue

        print(f"  Name: {profile['name']}")
        print(f"  Account Age: {profile['account_age_months']} months")
        print(f"  Verification: {profile['verification_status']}")
        print(f"  Risk Rating: {profile['risk_rating']}")
        print(f"  Avg Transaction: ${profile['average_transaction']:.2f}")
        print(f"  Country: {profile['country']}")
        print(f"  KYC Complete: {profile['kyc_completed']}")
        print()


def test_transaction_history():
    """Test transaction history retrieval."""
    print_section("TEST 2: Transaction History")

    customer_id = "CUST-001"
    print(f"Fetching transactions for: {customer_id}\n")

    transactions = get_transaction_history.invoke({
        "customer_id": customer_id,
        "days": 30
    })

    print(f"Found {len(transactions)} transactions in the last 30 days\n")

    # Show first 5 transactions
    for i, txn in enumerate(transactions[:5], 1):
        print(f"{i}. {txn['date']} - {txn['merchant']}")
        print(f"   Amount: ${txn['amount']:.2f} | Location: {txn['location']} | Device: {txn['device_id']}")

    if len(transactions) > 5:
        print(f"\n... and {len(transactions) - 5} more transactions")


def test_watchlist_checks():
    """Test watchlist screening."""
    print_section("TEST 3: Watchlist Screening")

    # Test normal customer
    print("Scenario 1: Normal customer, normal transaction")
    hits = check_watchlists.invoke({
        "customer_id": "CUST-001",
        "transaction_details": {
            "amount": 250.00,
            "country": "US",
            "beneficiary_name": "Amazon.com"
        }
    })

    if hits:
        print(f"  ⚠️  Found {len(hits)} watchlist hit(s):")
        for hit in hits:
            print(f"    - {hit['list_name']}: {hit['match_confidence']:.0%} confidence")
    else:
        print("  ✅ No watchlist hits")

    print()

    # Test high-risk country
    print("Scenario 2: Transaction to high-risk country")
    hits = check_watchlists.invoke({
        "customer_id": "CUST-001",
        "transaction_details": {
            "amount": 5000.00,
            "country": "IR",  # Iran - OFAC sanctioned
            "beneficiary_name": "Unknown Entity"
        }
    })

    if hits:
        print(f"  ⚠️  Found {len(hits)} watchlist hit(s):")
        for hit in hits:
            print(f"    - {hit['list_name']}: {hit['match_confidence']:.0%} confidence")
            print(f"      Reason: {hit['details']['reason']}")
    else:
        print("  ✅ No watchlist hits")

    print()

    # Test customer with fraud history
    print("Scenario 3: Customer with previous fraud case")
    hits = check_watchlists.invoke({
        "customer_id": "CUST-SUSPICIOUS-001",
        "transaction_details": {
            "amount": 1500.00,
            "country": "US",
        }
    })

    if hits:
        print(f"  ⚠️  Found {len(hits)} watchlist hit(s):")
        for hit in hits:
            print(f"    - {hit['list_name']}: {hit['match_confidence']:.0%} confidence")
            print(f"      {hit['details']['reason']}")


def test_risk_scoring():
    """Test risk score calculation."""
    print_section("TEST 4: Risk Score Calculation")

    scenarios = [
        {
            "name": "Low Risk - Established Customer",
            "customer_id": "CUST-001",
            "transaction_amount": 285.00,  # Close to average
            "alert_type": "unusual_amount",
        },
        {
            "name": "High Risk - New Account, Large Amount",
            "customer_id": "CUST-004",  # 2 months old, not verified
            "transaction_amount": 7500.00,  # 100x average
            "alert_type": "account_takeover",
        },
        {
            "name": "Medium Risk - Pending KYC",
            "customer_id": "CUST-002",  # 6 months, pending verification
            "transaction_amount": 450.00,  # 3x average
            "alert_type": "location_mismatch",
        },
    ]

    for scenario in scenarios:
        print(f"Scenario: {scenario['name']}")
        print(f"  Customer: {scenario['customer_id']}")
        print(f"  Amount: ${scenario['transaction_amount']:.2f}")
        print(f"  Alert Type: {scenario['alert_type']}")
        print()

        # Get customer profile
        profile = get_customer_profile.invoke({"customer_id": scenario['customer_id']})
        if "error" in profile:
            print(f"  ❌ Error: {profile['error']}\n")
            continue

        # Get transaction history
        history = get_transaction_history.invoke({
            "customer_id": scenario['customer_id'],
            "days": 30
        })

        # Calculate risk score
        risk_result = calculate_risk_score.invoke({
            "customer_profile": profile,
            "transaction_amount": scenario['transaction_amount'],
            "transaction_history": history,
            "alert_type": scenario['alert_type'],
        })

        # Display results
        score = risk_result['score']
        level = risk_result['level'].upper()
        confidence = risk_result['confidence']

        print(f"  📊 Risk Score: {score:.3f} / 1.0")
        print(f"  🎯 Risk Level: {level}")
        print(f"  📈 Confidence: {confidence:.0%}")
        print()

        if risk_result['factors']:
            print(f"  Risk Factors:")
            for factor in risk_result['factors']:
                print(f"    • {factor['reason']} (+{factor['contribution']})")

        print()
        print(f"  Recommendations:")
        for rec in risk_result['recommendations']:
            print(f"    → {rec}")

        print("\n" + "-" * 70 + "\n")


def test_similar_alerts():
    """Test similar alert retrieval."""
    print_section("TEST 5: Similar Alerts Lookup")

    alert_types = ["location_mismatch", "unusual_amount", "velocity", "device_change"]

    for alert_type in alert_types:
        print(f"Alert Type: {alert_type}")

        similar = get_similar_alerts.invoke({
            "alert_type": alert_type,
            "customer_id": "CUST-001"
        })

        if not similar:
            print("  No similar alerts found\n")
            continue

        print(f"  Found {len(similar)} similar alert(s):\n")

        for alert in similar:
            outcome_icon = "🔴" if alert['outcome'] == "confirmed_fraud" else "✅"
            print(f"    {outcome_icon} {alert['alert_id']}")
            print(f"       Date: {alert['date']} | Outcome: {alert['outcome']}")
            print(f"       Risk Score: {alert['risk_score']} | Similarity: {alert['similarity']:.0%}")
            print(f"       Reason: {alert['reason']}")
            print()


def test_tool_descriptions():
    """Display tool descriptions."""
    print_section("Tool Descriptions")

    descriptions = get_tool_descriptions()

    for name, desc in descriptions.items():
        print(f"📦 {name}")
        print(f"   {desc[:100]}...")
        print()


def main():
    """Run all tests."""
    print("\n" + "🔍" * 35)
    print("  FRAUD DETECTION TOOLS - TEST SUITE")
    print("🔍" * 35)

    # Run tests
    test_tool_descriptions()
    test_customer_profiles()
    test_transaction_history()
    test_watchlist_checks()
    test_risk_scoring()
    test_similar_alerts()

    print("\n" + "=" * 70)
    print("  All tests completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
