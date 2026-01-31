# FastAPI Application - Complete Guide

## Overview

The FastAPI application provides REST API endpoints for the FraudTriage-Agent, integrating with the LangGraph workflow for async fraud alert processing.

## File Created

**`src/api/main.py`** (~800 lines)

---

## API Endpoints

### 1. `POST /triage` - Submit Fraud Alert

Submit a fraud alert for asynchronous triage analysis.

**Request:**
```json
{
  "alert_id": "ALERT-2025-001234",
  "alert_type": "account_takeover",
  "transaction_amount": 7500.00,
  "customer_id": "CUST-004",
  "transaction_country": "NG",
  "transaction_device_id": "DEVICE-NEW-999",
  "merchant_name": "Luxury Electronics Store",
  "alert_reason": "Transaction from high-risk country with new device"
}
```

**Response (202 Accepted):**
```json
{
  "alert_id": "ALERT-2025-001234",
  "status": "pending",
  "decision": null,
  "risk_score": null,
  "risk_level": null,
  "risk_factors": [],
  "recommendation": null,
  "requires_human_review": false,
  "processing_time_ms": null,
  "created_at": "2025-01-30T14:25:00Z",
  "completed_at": null,
  "error": null
}
```

**Valid Alert Types:**
- `unusual_amount`
- `velocity`
- `location_mismatch`
- `device_change`
- `account_takeover`

---

### 2. `GET /triage/{alert_id}` - Get Alert Status

Retrieve the current status and results of a submitted alert.

**Response (Processing):**
```json
{
  "alert_id": "ALERT-2025-001234",
  "status": "processing",
  "decision": null,
  "risk_score": null,
  ...
}
```

**Response (Completed):**
```json
{
  "alert_id": "ALERT-2025-001234",
  "status": "completed",
  "decision": "escalate",
  "risk_score": 0.875,
  "risk_level": "critical",
  "risk_factors": [
    "Very new account (2 months old)",
    "Transaction amount 100x higher than average",
    "Customer KYC not verified"
  ],
  "recommendation": "Alert escalated due to high risk score. Immediate review required.",
  "requires_human_review": true,
  "processing_time_ms": 2340,
  "created_at": "2025-01-30T14:25:00Z",
  "completed_at": "2025-01-30T14:25:02Z",
  "error": null
}
```

**Status Values:**
- `pending` - Alert queued, not yet processed
- `processing` - Agent is analyzing the alert
- `completed` - Analysis complete
- `error` - Processing failed

---

### 3. `POST /triage/{alert_id}/approve` - Human Review

Submit human analyst review/approval for escalated alerts.

**Request:**
```json
{
  "reviewer_id": "ANALYST-001",
  "reviewer_name": "Jane Smith",
  "decision": "reject",
  "reasoning": "Customer confirmed they did not authorize this transaction. Nigeria location and new device are confirmed fraud indicators.",
  "tags": ["confirmed_fraud", "account_takeover"]
}
```

**Valid Decisions:**
- `approve` - Transaction is legitimate (auto-close)
- `reject` - Transaction is fraudulent (escalate)
- `escalate` - Need additional review

**Response:**
```json
{
  "alert_id": "ALERT-2025-001234",
  "reviewer_id": "ANALYST-001",
  "reviewer_name": "Jane Smith",
  "decision": "reject",
  "reasoning": "Customer confirmed they did not authorize this transaction...",
  "reviewed_at": "2025-01-30T14:26:00Z",
  "updated_decision": "escalate",
  "tags": ["confirmed_fraud", "account_takeover"]
}
```

---

### 4. `GET /health` - Health Check

Check API health status and metrics.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "agent_environment": "development",
  "alerts_processed": 42,
  "uptime_seconds": 3600.5
}
```

---

## Request/Response Models

### TriageRequest

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| alert_id | str | ✅ | Unique alert identifier |
| alert_type | str | ✅ | Type of fraud alert |
| transaction_amount | float | ✅ | Transaction amount (must be > 0) |
| customer_id | str | ✅ | Customer identifier |
| transaction_country | str | ❌ | ISO 3166-1 alpha-2 country code |
| transaction_device_id | str | ❌ | Device identifier |
| merchant_name | str | ❌ | Merchant name |
| alert_reason | str | ❌ | Reason for the alert |

### TriageResponse

| Field | Type | Description |
|-------|------|-------------|
| alert_id | str | Alert identifier |
| status | str | pending/processing/completed/error |
| decision | str \| null | auto_close/review_required/escalate |
| risk_score | float \| null | 0.0 - 1.0 |
| risk_level | str \| null | low/medium/high/critical |
| risk_factors | list[str] | Identified risk factors |
| recommendation | str \| null | Action recommendation |
| requires_human_review | bool | Whether human review needed |
| processing_time_ms | int \| null | Processing duration |
| created_at | datetime | Alert creation time |
| completed_at | datetime \| null | Alert completion time |
| error | str \| null | Error message if status is error |

---

## In-Memory Storage

The API uses an `AlertStore` class for in-memory alert storage:

```python
class AlertStore:
    async def get(alert_id: str) -> dict | None
    async def set(alert_id: str, data: dict) -> None
    async def update(alert_id: str, updates: dict) -> dict | None
    async def list_all() -> list[dict]
    async def delete(alert_id: str) -> bool
```

**Features:**
- Thread-safe with `asyncio.Lock`
- Auto-evicts oldest alert when at capacity (1000 alerts)
- For demo purposes only - use a real database in production

---

## Background Processing

Alerts are processed asynchronously using FastAPI `BackgroundTasks`:

```python
async def process_alert_async(
    alert_id: str,
    alert_type: str,
    transaction_amount: float,
    customer_id: str,
    **kwargs
) -> None:
    # Update status to processing
    await alert_store.update(alert_id, {"status": "processing"})

    # Run fraud triage agent
    result = await fraud_agent.arun(...)

    # Store results
    await alert_store.update(alert_id, {
        "status": "completed",
        "decision": result["decision"].value,
        "risk_score": result["risk_score"],
        ...
    })
```

---

## WebSocket Support (Optional)

WebSocket endpoint for real-time updates is included but commented out:

```python
# Uncomment to enable:
# @app.websocket("/ws/triage/{alert_id}")
# async def websocket_endpoint(websocket: WebSocket, alert_id: str):
#     await manager.connect(websocket, alert_id)
#     # Send real-time updates...
```

**To enable:**
1. Uncomment `manager = ConnectionManager()`
2. Uncomment the WebSocket endpoint
3. Uncomment `await manager.send_update()` calls

---

## CORS Configuration

Default allowed origins:
- `http://localhost:3000`
- `http://localhost:8000`
- `http://localhost:8080`

Modify in `APIConfig.ALLOW_ORIGINS` as needed.

---

## Configuration

### Environment Variables

```bash
# Agent Environment
ENVIRONMENT=development  # development, demo, or production

# For Demo (GLM-4.7)
GLM_API_KEY=your_glm_api_key
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4

# For Production (OpenAI)
OPENAI_API_KEY=your_openai_api_key
```

### APIConfig Class

```python
class APIConfig:
    API_TITLE = "FraudTriage-Agent API"
    API_VERSION = "0.1.0"
    ALLOW_ORIGINS = [...]
    AGENT_ENVIRONMENT = "development"
    ALERT_STORE_MAX_SIZE = 1000
```

---

## Error Handling

| HTTP Code | Description |
|-----------|-------------|
| 202 Accepted | Alert submitted successfully |
| 400 Bad Request | Invalid request data or alert doesn't require review |
| 404 Not Found | Alert not found |
| 409 Conflict | Alert ID already exists |
| 500 Internal Server Error | Processing error |

**Error Response:**
```json
{
  "detail": "Alert ALERT-001 already exists. Use GET /triage/ALERT-001 to check status."
}
```

---

## Running the API

### Development

```bash
# Using Python
python -m src.api.main

# Using uvicorn directly
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
# Set environment
export ENVIRONMENT=production
export OPENAI_API_KEY=your_key

# Run with gunicorn
gunicorn src.api.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000
```

---

## Interactive Documentation

Once running, access:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## Usage Example

### Using curl

```bash
# 1. Submit alert
curl -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{
    "alert_id": "ALERT-001",
    "alert_type": "account_takeover",
    "transaction_amount": 7500.00,
    "customer_id": "CUST-004",
    "transaction_country": "NG",
    "transaction_device_id": "DEVICE-NEW-999"
  }'

# 2. Check status
curl http://localhost:8000/triage/ALERT-001

# 3. Submit human review (if required)
curl -X POST http://localhost:8000/triage/ALERT-001/approve \
  -H "Content-Type: application/json" \
  -d '{
    "reviewer_id": "ANALYST-001",
    "reviewer_name": "Jane Smith",
    "decision": "reject",
    "reasoning": "Customer confirmed unauthorized transaction"
  }'
```

### Using Python

```python
import requests
import time

BASE_URL = "http://localhost:8000"

# Submit alert
response = requests.post(f"{BASE_URL}/triage", json={
    "alert_id": "ALERT-001",
    "alert_type": "account_takeover",
    "transaction_amount": 7500.00,
    "customer_id": "CUST-004",
    "transaction_country": "NG",
})
print(f"Submitted: {response.json()}")

# Poll for completion
while True:
    response = requests.get(f"{BASE_URL}/triage/ALERT-001")
    data = response.json()

    if data["status"] in ["completed", "error"]:
        print(f"Final result: {data}")
        break

    print(f"Status: {data['status']}")
    time.sleep(1)

# Submit human review if needed
if data.get("requires_human_review"):
    response = requests.post(f"{BASE_URL}/triage/ALERT-001/approve", json={
        "reviewer_id": "ANALYST-001",
        "reviewer_name": "Jane Smith",
        "decision": "reject",
        "reasoning": "Customer confirmed unauthorized transaction",
    })
    print(f"Review submitted: {response.json()}")
```

---

## Production Considerations

1. **Database**: Replace `AlertStore` with PostgreSQL/MongoDB
2. **Queue**: Use Celery/RabbitMQ for background tasks
3. **Auth**: Add JWT authentication for reviewers
4. **Rate Limiting**: Add rate limiting middleware
5. **Monitoring**: Add Prometheus metrics
6. **Logging**: Use structured logging (JSON)
7. **Tracing**: Enable LangSmith for LLM tracing
8. **WebSockets**: Enable for real-time updates

---

## Next Steps

1. **Install dependencies**
   ```bash
   make install
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with API keys
   ```

3. **Start the API**
   ```bash
   python -m src.api.main
   ```

4. **Test with Swagger UI**
   Open http://localhost:8000/docs

5. **Run test client**
   ```bash
   python scripts/test_api_client.py
   ```
