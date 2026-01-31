#!/usr/bin/env python3
"""
Test script to submit a sample fraud alert to the triage system.

Usage:
    python scripts/test_sample_alert.py [--alert-id SAMPLE-001]
"""

import asyncio
import json
import sys
from pathlib import Path
from argparse import ArgumentParser

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.graph import get_fraud_triage_graph
from src.config.settings import settings
from src.utils.formatting import format_triage_result


async def main():
    """Main function to test the alert triage."""
    parser = ArgumentParser(description="Test fraud alert triage")
    parser.add_argument(
        "--alert-file",
        default="data/sample_alerts/sample_alert.json",
        help="Path to sample alert JSON file"
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use API instead of direct graph invocation"
    )
    args = parser.parse_args()

    # Load sample alert
    alert_path = Path(args.alert_file)
    if not alert_path.exists():
        print(f"Error: Alert file not found: {alert_path}")
        sys.exit(1)

    with open(alert_path) as f:
        alert_data = json.load(f)

    print("=" * 60)
    print("FRAUD TRIAGE AGENT - TEST ALERT")
    print("=" * 60)
    print(f"\nAlert ID: {alert_data['alert_id']}")
    print(f"Alert Type: {alert_data['alert_type']}")
    print(f"Amount: ${alert_data['transaction']['amount']}")
    print(f"Location: {alert_data['transaction']['location_city']}, {alert_data['transaction']['location_country']}")
    print(f"Reason: {alert_data['alert_reason']}")
    print("\n" + "=" * 60)
    print("RUNNING TRIAGE WORKFLOW...")
    print("=" * 60 + "\n")

    if args.use_api:
        print("Using API endpoint...")
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/api/v1/alerts",
                json=alert_data,
            )
            if response.status_code == 202:
                result = response.json()
                print(f"Alert submitted: {result['alert_id']}")
                print(f"Status: {result['status']}")

                # Poll for completion
                import time
                for _ in range(30):
                    await asyncio.sleep(1)
                    status_response = await client.get(f"http://localhost:8000/api/v1/alerts/{result['alert_id']}")
                    alert_status = status_response.json()
                    if alert_status.get("risk_score") is not None:
                        print(format_triage_result(alert_status))
                        break
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
    else:
        # Use direct graph invocation
        print("Using direct graph invocation...\n")
        graph = get_fraud_triage_graph()
        result = await graph.arun(alert_data)

        # Format and print result
        print(format_triage_result(result))

        # Print state summary
        print("\n" + "=" * 60)
        print("WORKFLOW SUMMARY")
        print("=" * 60)
        print(f"Iterations: {result.get('iteration_count', 0)}")
        print(f"Messages exchanged: {len(result.get('messages', []))}")
        print(f"Transactions gathered: {len(result.get('transaction_history', []))}")
        print(f"Customer profile: {'Loaded' if result.get('customer_profile') else 'Not loaded'}")
        print(f"Device info: {'Loaded' if result.get('device_fingerprint') else 'Not loaded'}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
