# Fraud Detection Tools - Summary

## Overview

Created `src/tools/fraud_tools.py` with 5 comprehensive LangChain tools for fraud detection, all using the `@tool` decorator with realistic mock data.

## Files Created

### 1. `src/tools/fraud_tools.py` (~700 lines)
All tools use the `@tool` decorator and include type hints, docstrings, and detailed examples.

---

## Tool Descriptions

### Tool 1: `get_customer_profile`

**Purpose**: Retrieve customer profile information for fraud risk assessment

**Input**:
- `customer_id`: str (e.g., "CUST-001")

**Output**: Dictionary containing:
```python
{
    "customer_id": "CUST-001",
    "name": "John Smith",
    "account_age_months": 48,
    "verification_status": "verified",  # verified/pending/not_verified
    "risk_rating": "low",              # low/medium/high
    "total_transactions_ytd": 127,
    "average_transaction": 285.00,
    "country": "US",
    "occupation": "Software Engineer",
    "segment": "retail",               # retail/premium
    "kyc_completed": True,
    "previous_fraud_cases": 0,
    "false_positive_count": 2,
    "registered_devices": ["DEVICE-001-APPLE", "DEVICE-001-WINDOWS"],
    "typical_countries": ["US", "CA", "UK"],
}
```

**Mock Customers**:
- `CUST-001`: John Smith - Low risk, 4-year account, verified
- `CUST-002`: Maria Garcia - Medium risk, 6-month account, pending verification
- `CUST-003`: Chen Wei - Low risk, 10-year account, premium segment
- `CUST-004`: Ahmed Hassan - High risk, 2-month account, not verified
- `CUST-005`: Sarah Johnson - Low risk, 3-year account, high-value customer

---

### Tool 2: `get_transaction_history`

**Purpose**: Retrieve recent transactions for pattern analysis

**Input**:
- `customer_id`: str
- `days`: int = 30 (max 90)

**Output**: List of transaction dictionaries:
```python
[
    {
        "transaction_id": "TXN-CUST-001-00001",
        "customer_id": "CUST-001",
        "account_id": "ACC-001",
        "date": "2025-01-28",
        "timestamp": "2025-01-28T14:30:00Z",
        "amount": 285.50,
        "currency": "USD",
        "merchant": "Amazon",
        "category": "retail",
        "status": "completed",
        "location": "US",
        "device_id": "DEVICE-001-APPLE",
    },
    # ... more transactions
]
```

**Features**:
- Generates realistic transaction patterns based on customer profile
- Includes 15+ merchants (Amazon, Walmart, Target, Starbucks, etc.)
- Transaction amounts vary around customer's average (±50%)
- 90% of transactions from typical countries, 10% from others (anomalies)

---

### Tool 3: `check_watchlists`

**Purpose**: Screen customer and transaction against watchlists

**Input**:
- `customer_id`: str
- `transaction_details`: dict with:
  - `amount`: float
  - `country`: str (2-letter code)
  - `beneficiary_name`: str (optional)

**Output**: List of watchlist hits (empty if clean):
```python
[
    {
        "list_name": "OFAC Sanctions List (Country-Based)",
        "match_type": "partial",
        "match_confidence": 0.60,
        "details": {
            "reason": "Transaction to/from high-risk country: IR",
            "country": "IR",
            "sanctions_program": "Country-Based Sanctions",
        }
    }
]
```

**Watchlists Checked**:
1. **Internal Fraud Database** - Previous confirmed fraud cases
2. **OFAC Sanctions List** - High-risk countries (IR, KP, SY, CU, MM, RU, BY)
3. **Internal AML Monitoring** - Large transactions >$10,000
4. **Suspicious Beneficiary List** - Names matching suspicious patterns

**Mock Scenarios**:
- `CUST-SUSPICIOUS-001`: Previous confirmed fraud case
- `CUST-SANCTIONS-001`: OFAC SDN list match

---

### Tool 4: `calculate_risk_score`

**Purpose**: Calculate comprehensive risk score with contributing factors

**Input**:
- `customer_profile`: dict (from get_customer_profile)
- `transaction_amount`: float
- `transaction_history`: list (from get_transaction_history)
- `alert_type`: str

**Output**: Dictionary with:
```python
{
    "score": 0.625,              # 0.0 - 1.0
    "level": "high",             # low/medium/high/critical
    "factors": [
        {
            "reason": "Transaction amount 7.5x higher than average",
            "contribution": 0.30,
            "data_point": "ratio=7.5"
        },
        {
            "reason": "Customer KYC not verified",
            "contribution": 0.20,
            "data_point": "verification_status=not_verified"
        },
        # ... more factors
    ],
    "confidence": 0.80,          # 0.0 - 1.0
    "recommendations": [
        "Verify transaction legitimacy with customer",
        "Require KYC verification"
    ],
    "customer_base_risk": "high",
    "data_points": {
        "account_age_months": 2,
        "average_transaction": 75.00,
        "transaction_count": 3,
        "verification_status": "not_verified",
    }
}
```

**Risk Factors & Weights**:

| Factor | Weight | Condition |
|--------|--------|-----------|
| Very new account (< 3 months) | +0.30 | account_age_months < 3 |
| Relatively new (< 12 months) | +0.10 | account_age_months < 12 |
| Amount > 10x average | +0.40 | deviation_ratio > 10 |
| Amount > 5x average | +0.30 | deviation_ratio > 5 |
| Amount > 3x average | +0.15 | deviation_ratio > 3 |
| Not verified | +0.20 | verification_status == "not_verified" |
| Verification pending | +0.10 | verification_status == "pending" |
| High-risk alert type | +0.25 | account_takeover, velocity, device_change |
| Medium-risk alert type | +0.15 | location_mismatch, unusual_amount |
| No transaction history | +0.15 | len(history) == 0 |
| High velocity | +0.15 | >20 transactions in period |
| Previous fraud victim | +0.10 | previous_fraud_cases > 0 |
| High risk profile | +0.20 | risk_rating == "high" |
| Medium risk profile | +0.10 | risk_rating == "medium" |

**Risk Levels**:
- **0.00 - 0.25**: LOW - Clear indicators of legitimate activity
- **0.26 - 0.50**: MEDIUM - Some concerns but likely legitimate
- **0.51 - 0.75**: HIGH - Significant risk indicators
- **0.76 - 1.00**: CRITICAL - Strong fraud probability

---

### Tool 5: `get_similar_alerts`

**Purpose**: Find historical alerts with similar patterns

**Input**:
- `alert_type`: str (e.g., "location_mismatch")
- `customer_id`: str

**Output**: List of similar alerts:
```python
[
    {
        "alert_id": "ALERT-2024-0891",
        "date": "2024-11-15",
        "customer_id": "CUST-001",
        "outcome": "false_positive",
        "reason": "Customer was traveling",
        "risk_score": 65,
        "similarity": 0.85,  # 0.0 - 1.0
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
]
```

**Available Alert Types**:
- `location_mismatch` - 3 similar alerts
- `unusual_amount` - 2 similar alerts
- `velocity` - 2 similar alerts
- `device_change` - 2 similar alerts

---

## Usage Examples

### Basic Usage

```python
from src.tools.fraud_tools import (
    get_customer_profile,
    get_transaction_history,
    calculate_risk_score,
)

# Get customer profile
profile = get_customer_profile.invoke({"customer_id": "CUST-001"})
print(f"Customer: {profile['name']}, Risk: {profile['risk_rating']}")

# Get transaction history
history = get_transaction_history.invoke({
    "customer_id": "CUST-001",
    "days": 30
})
print(f"Found {len(history)} transactions")

# Calculate risk score
risk = calculate_risk_score.invoke({
    "customer_profile": profile,
    "transaction_amount": 5000.00,
    "transaction_history": history,
    "alert_type": "unusual_amount",
})
print(f"Risk Score: {risk['score']:.2f} ({risk['level']})")
```

### Complete Fraud Assessment

```python
from src.tools.fraud_tools import (
    get_customer_profile,
    get_transaction_history,
    check_watchlists,
    calculate_risk_score,
    get_similar_alerts,
)

# Gather all information
customer_id = "CUST-004"
alert_type = "account_takeover"
transaction_amount = 7500.00
transaction_details = {
    "amount": transaction_amount,
    "country": "NG",
    "beneficiary_name": "Unknown Person"
}

# 1. Get profile
profile = get_customer_profile.invoke({"customer_id": customer_id})

# 2. Get history
history = get_transaction_history.invoke({"customer_id": customer_id, "days": 30})

# 3. Check watchlists
hits = check_watchlists.invoke({
    "customer_id": customer_id,
    "transaction_details": transaction_details
})

# 4. Calculate risk
risk = calculate_risk_score.invoke({
    "customer_profile": profile,
    "transaction_amount": transaction_amount,
    "transaction_history": history,
    "alert_type": alert_type,
})

# 5. Get similar alerts
similar = get_similar_alerts.invoke({"alert_type": alert_type, "customer_id": customer_id})

# Display results
print(f"Risk Score: {risk['score']:.2f} / 1.0")
print(f"Risk Level: {risk['level'].upper()}")
print(f"\nRisk Factors:")
for factor in risk['factors']:
    print(f"  +{factor['contribution']:.2f}: {factor['reason']}")

if hits:
    print(f"\n⚠️  Watchlist Hits: {len(hits)}")
    for hit in hits:
        print(f"  - {hit['list_name']} ({hit['match_confidence']:.0%})")

if similar:
    fraud_count = sum(1 for a in similar if a['outcome'] == 'confirmed_fraud')
    print(f"\n📊 Similar Alerts: {len(similar)} ({fraud_count} confirmed fraud)")
```

---

## Test Script

### `scripts/test_fraud_tools.py`

Comprehensive test suite demonstrating all tools with realistic scenarios:

```bash
python scripts/test_fraud_tools.py
```

**Test Coverage**:
1. Customer Profile Retrieval - Different risk levels
2. Transaction History - Pattern analysis
3. Watchlist Screening - Normal, high-risk, and flagged customers
4. Risk Score Calculation - 3 scenarios (low, high, medium risk)
5. Similar Alerts Lookup - Historical pattern matching
6. Tool Descriptions - Documentation

---

## Integration with LangGraph

These tools integrate seamlessly with the LangGraph workflow:

```python
from src.tools.fraud_tools import get_all_fraud_tools

# Get all tools as LangChain tool objects
tools = get_all_fraud_tools()

# Bind to LLM
llm_with_tools = llm.bind_tools(tools)

# Use in agent nodes
@tool
def gather_context_node(state: FraudTriageState) -> FraudTriageState:
    # Tools are automatically available to the LLM
    profile = get_customer_profile.invoke({"customer_id": state["customer_id"]})
    state["customer_profile"] = profile
    return state
```

---

## Realistic Fraud Scenarios Covered

### Scenario 1: Account Takeover
- **Customer**: CUST-004 (2 months old, not verified)
- **Transaction**: $7,500 to Nigeria
- **Risk Score**: ~0.80 (CRITICAL)
- **Factors**: New account, unverified, unusual location, high amount

### Scenario 2: Legitimate Large Purchase
- **Customer**: CUST-005 (3 years, verified, high-value)
- **Transaction**: $4,200 in US
- **Risk Score**: ~0.15 (LOW)
- **Factors**: None significant - customer profile strong

### Scenario 3: Suspicious Velocity
- **Customer**: CUST-001 (4 years, verified)
- **Transaction**: 25 transactions in 1 hour
- **Risk Score**: ~0.60 (HIGH)
- **Factors**: Unusual velocity pattern, possible bot activity

---

## Next Steps

1. **Install dependencies**:
   ```bash
   cd FraudTriage-Agent
   make install
   ```

2. **Run test suite**:
   ```bash
   python scripts/test_fraud_tools.py
   ```

3. **Integrate with LangGraph workflow**:
   - Tools already imported in `src/agents/triage_nodes.py`
   - Used by `gather_context_node()`
   - Results fed to `assess_risk_node()` for LLM analysis

4. **Extend with real data sources**:
   - Replace mock data with API calls
   - Connect to transaction databases
   - Integrate with real watchlist services (OFAC API, etc.)
