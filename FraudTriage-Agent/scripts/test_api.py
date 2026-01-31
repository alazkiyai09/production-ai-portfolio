#!/usr/bin/env python3
"""
Test script for FraudTriage-Agent API.

This script starts the server, tests endpoints, then shuts down.
"""

import asyncio
import json
import os
import sys
import time
from contextlib import asynccontextmanager

import httpx

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment for development
os.environ['ENVIRONMENT'] = 'development'

from src.api.main import app
from uvicorn import Config, Server


BASE_URL = "http://127.0.0.1:8888"

async def test_api():
    """Test the API endpoints."""
    print("\n" + "=" * 70)
    print("  Testing FraudTriage-Agent API")
    print("=" * 70 + "\n")

    async with httpx.AsyncClient() as client:
        # Test 1: Root endpoint
        print("1️⃣  Testing GET /")
        try:
            response = await client.get(f"{BASE_URL}/")
            data = response.json()
            print(f"   Status: {response.status_code}")
            print(f"   Name: {data.get('name', 'API')}")
            print(f"   Endpoints: {list(data.get('endpoints', {}).keys())}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False

        # Test 2: Health check
        print("\n2️⃣  Testing GET /health")
        try:
            response = await client.get(f"{BASE_URL}/health")
            data = response.json()
            print(f"   Status: {response.status_code}")
            print(f"   Health: {data.get('status')}")
            print(f"   Version: {data.get('version')}")
            print(f"   Environment: {data.get('agent_environment')}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False

        # Test 3: Submit alert
        print("\n3️⃣  Testing POST /triage (submit alert)")
        alert_request = {
            "alert_id": "TEST-ALERT-001",
            "alert_type": "account_takeover",
            "transaction_amount": 7500.00,
            "customer_id": "CUST-004",
            "transaction_country": "NG",
            "transaction_device_id": "DEVICE-NEW-999",
            "merchant_name": "Luxury Electronics",
            "alert_reason": "Transaction from high-risk country with new device"
        }

        try:
            response = await client.post(
                f"{BASE_URL}/triage",
                json=alert_request,
                timeout=10.0
            )
            print(f"   Status: {response.status_code}")
            if response.status_code == 202:
                data = response.json()
                print(f"   ✅ Alert accepted: {data['alert_id']}")
                print(f"   Status: {data['status']}")
                alert_id = data['alert_id']
            else:
                print(f"   ❌ Error: {response.text}")
                return False
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False

        # Wait a bit for processing
        print("\n⏳  Waiting for processing...")
        for i in range(10):
            await asyncio.sleep(1)

            # Test 4: Get alert status
            try:
                response = await client.get(f"{BASE_URL}/triage/{alert_id}")
                data = response.json()

                if data['status'] == 'completed':
                    print(f"\n4️⃣  Processing complete!")
                    print(f"   Status: {data['status']}")
                    print(f"   Decision: {data.get('decision', 'N/A')}")
                    print(f"   Risk Score: {data.get('risk_score', 'N/A')}")
                    print(f"   Risk Level: {data.get('risk_level', 'N/A')}")
                    if data.get('risk_factors'):
                        print(f"   Risk Factors:")
                        for factor in data['risk_factors'][:3]:
                            print(f"      - {factor}")
                    print(f"   Recommendation: {data.get('recommendation', 'N/A')[:80]}...")
                    print(f"   Processing Time: {data.get('processing_time_ms', 0)}ms")
                    break
                elif data['status'] == 'error':
                    print(f"\n❌ Processing error: {data.get('error', 'Unknown')}")
                    break
                else:
                    print(f"   Status: {data['status']}... (waiting)")
            except Exception as e:
                print(f"   Error checking status: {e}")
                break
        else:
            print("\n⏱️  Timeout waiting for completion")

        # Test 5: Human review (if needed)
        if data.get('requires_human_review'):
            print(f"\n5️⃣  Testing POST /triage/{alert_id}/approve (human review)")
            review_request = {
                "reviewer_id": "ANALYST-TEST",
                "reviewer_name": "Test Analyst",
                "decision": "reject",
                "reasoning": "Testing human review workflow - confirming fraudulent transaction based on high risk indicators.",
                "tags": ["test", "fraud"]
            }

            try:
                response = await client.post(
                    f"{BASE_URL}/triage/{alert_id}/approve",
                    json=review_request
                )
                print(f"   Status: {response.status_code}")
                if response.status_code == 200:
                    review_data = response.json()
                    print(f"   ✅ Review submitted")
                    print(f"   Decision: {review_data['decision']}")
                    print(f"   Updated Decision: {review_data['updated_decision']}")
                else:
                    print(f"   Response: {response.text}")
            except Exception as e:
                print(f"   ❌ Error: {e}")

    print("\n" + "=" * 70)
    print("  ✅ API Tests Complete!")
    print("=" * 70 + "\n")
    return True


@asynccontextmanager
async def run_server():
    """Context manager to run the server temporarily."""
    config = Config(app, host="127.0.0.1", port=8888, log_level="warning")
    server = Server(config=config)

    # Start server in background
    import threading
    thread = threading.Thread(target=server.run)
    thread.daemon = True
    thread.start()

    # Wait for server to start
    await asyncio.sleep(2)

    yield

    # Shutdown
    server.should_exit = True
    await asyncio.sleep(1)


async def main():
    """Main function to run server and tests."""
    print("\n🚀 Starting FraudTriage-Agent API server...")

    async with run_server():
        print(f"📡 Server running at: {BASE_URL}")
        print("   (Press Ctrl+C to stop)\n")

        # Run tests
        await test_api()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests stopped by user")
