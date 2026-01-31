# FraudTriageAgent - Complete LangGraph Agent

## Overview

The `FraudTriageAgent` is a production-ready LangGraph workflow for automated fraud alert triage. It integrates state definitions, fraud tools, and LLM analysis into a cohesive system.

## File Created

**`src/agents/fraud_triage_agent.py`** (~900 lines)

---

## Architecture

### Workflow Graph

```
┌─────────────────────────────────────────────────────────────────┐
│                        PARSE_ALERT                              │
│  - Validate alert data                                          │
│  - Extract key fields                                           │
│  - Initialize processing                                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      GATHER_CONTEXT                             │
│  Tool Calls:                                                    │
│  ✓ get_customer_profile()                                       │
│  ✓ get_transaction_history(days=30)                             │
│  ✓ check_watchlists(transaction)                                │
│  ✓ calculate_risk_score(profile, txn, history)                  │
│  ✓ get_similar_alerts(alert_type)                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ANALYZE_RISK                              │
│  - Invoke LLM with gathered context                             │
│  - Parse LLM analysis                                           │
│  - Determine final decision                                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                    ┌───────┴────────┐
                    │   route_decision()  │
                    └───────┬────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
  risk > 0.8      0.4 < risk ≤ 0.8     risk ≤ 0.4
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  ESCALATE     │  │  RECOMMEND    │  │  AUTO_CLOSE   │
│  🚨 High Risk │  │  📋 Medium    │  │  ✅ Low Risk  │
│  - Block txn  │  │  - Review     │  │  - Close      │
│  - Escalate   │  │  - Monitor    │  │  - Archive    │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                         END
```

---

## LLM Factory Function

### `create_llm(environment, model) -> BaseChatModel`

Supports three environments:

| Environment | LLM | Use Case | API Key Required |
|-------------|-----|----------|------------------|
| **development** | Ollama (llama3.2) | Local testing | ❌ No |
| **demo** | GLM-4.7 | Demos/presentations | ✅ GLM_API_KEY |
| **production** | OpenAI GPT-4o | Production | ✅ OPENAI_API_KEY |

**Environment Variables:**

```bash
# Development (default)
ENVIRONMENT=development
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Demo
ENVIRONMENT=demo
GLM_API_KEY=your_glm_api_key
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
GLM_MODEL=glm-4-plus

# Production
ENVIRONMENT=production
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o
```

**Usage:**

```python
# Auto-detect from environment
llm = create_llm()

# Explicit environment
llm = create_llm(environment="development")  # Ollama
llm = create_llm(environment="demo")         # GLM-4.7
llm = create_llm(environment="production")   # OpenAI
```

---

## Node Functions

### 1. `parse_alert_node(state) -> state`

**Purpose**: Validate and extract alert information

**Input**: Initial state with alert_id, alert_type, transaction_amount, customer_id

**Output**: Validated state with processing_started timestamp

**Validates**:
- customer_id exists
- transaction_amount > 0

**On Error**: Sets error_message, stops workflow

---

### 2. `gather_context_node(state) -> state`

**Purpose**: Call all fraud tools to gather comprehensive context

**Tool Calls**:
1. `get_customer_profile(customer_id)`
2. `get_transaction_history(customer_id, days=30)`
3. `check_watchlists(customer_id, transaction_details)`
4. `calculate_risk_score(profile, amount, history, alert_type)`
5. `get_similar_alerts(alert_type, customer_id)`

**Output**: State with all context fields populated

**State Updates**:
- `customer_profile`: dict
- `transaction_history`: list
- `watchlist_hits`: list
- `risk_score`: float (0-1)
- `similar_alerts`: list
- `tools_used`: list[str]
- `confidence`: float

---

### 3. `analyze_risk_node(state) -> state`

**Purpose**: Use LLM to synthesize context and produce final decision

**Process**:
1. Format all context for LLM
2. Invoke LLM with risk analysis prompt
3. Parse LLM response for decision
4. Update state with final analysis

**LLM Prompt Includes**:
- Alert details
- Customer profile summary
- Transaction history summary
- Risk assessment score
- Watchlist hits
- Similar alerts
- Account age and verification status

**Output**: State with:
- `risk_level`: RiskLevel enum
- `risk_factors`: list[str]
- `decision`: AlertDecision enum
- `recommendation`: str
- `requires_human_review`: bool

**Fallback**: If LLM fails, uses rule-based decision from risk_score

---

### 4. `escalate_alert_node(state) -> state`

**Purpose**: Handle high-risk alerts (risk_score > 0.8)

**Actions**:
- Sets `decision = ESCALATE`
- Sets `requires_human_review = True`
- Calculates processing_duration_ms
- Builds escalation recommendation

**Recommendation Example**:
```
Alert TEST-001 escalated due to high risk score (0.875).
Risk factors: Very new account, Transaction amount 100x higher,
Customer KYC not verified, High-risk alert type.
Immediate review required.
```

---

### 5. `recommend_action_node(state) -> state`

**Purpose**: Handle medium-risk alerts (0.4 < risk_score ≤ 0.8)

**Actions**:
- Sets `decision = REVIEW_REQUIRED`
- Sets `requires_human_review = True`
- Calculates processing_duration_ms
- Builds action recommendation

**Recommendation Example**:
```
Alert TEST-002 requires review (risk score: 0.625).
Recommend: Monitor for additional indicators.
Factors: Relatively new account, Transaction amount 5x higher.
```

---

### 6. `auto_close_alert_node(state) -> state`

**Purpose**: Handle low-risk alerts (risk_score ≤ 0.4)

**Actions**:
- Sets `decision = AUTO_CLOSE`
- Sets `requires_human_review = False`
- Calculates processing_duration_ms
- Builds closure reason

**Recommendation Example**:
```
Alert TEST-003 auto-closed as false positive (risk score: 0.185).
No significant fraud indicators detected. Transaction appears legitimate.
```

---

## Routing Function

### `route_decision(state) -> "escalate" | "recommend" | "auto_close"`

Routes alerts based on risk score thresholds:

```python
if risk_score > 0.8:
    return "escalate"     # High risk
elif risk_score > 0.4:
    return "recommend"    # Medium risk
else:
    return "auto_close"   # Low risk
```

**Thresholds** (configurable via `AgentConfig`):
- `ESCALATE_THRESHOLD = 0.8`
- `RECOMMEND_THRESHOLD = 0.4`

---

## Main Agent Class

### `FraudTriageAgent`

```python
class FraudTriageAgent:
    def __init__(self, environment: str | None = None)

    async def arun(
        alert_id: str,
        alert_type: str | AlertType,
        transaction_amount: float,
        customer_id: str,
        **kwargs
    ) -> FraudTriageState

    def run(
        alert_id: str,
        alert_type: str | AlertType,
        transaction_amount: float,
        customer_id: str,
        **kwargs
    ) -> FraudTriageState
```

**Usage Examples:**

```python
# Initialize agent
agent = FraudTriageAgent(environment="demo")

# Async execution
result = await agent.arun(
    alert_id="ALERT-001",
    alert_type=AlertType.ACCOUNT_TAKEOVER,
    transaction_amount=7500.00,
    customer_id="CUST-004",
    transaction_country="NG",
    transaction_device_id="DEVICE-NEW-999",
)

# Check results
print(f"Risk Score: {result['risk_score']}")
print(f"Decision: {result['decision'].value}")
print(f"Recommendation: {result['recommendation']}")

# Sync execution
result = agent.run(
    alert_id="ALERT-002",
    alert_type="unusual_amount",
    transaction_amount=350.00,
    customer_id="CUST-001",
)
```

---

## Helper Functions

### Formatting Functions

- `_format_customer_profile(profile)`: Customer profile → markdown
- `_format_transaction_history(history)`: Transaction summary
- `_format_risk_assessment(score, profile)`: Risk analysis
- `_format_watchlist_hits(hits)`: Watchlist results
- `_format_similar_alerts(alerts)`: Similar alerts list

### Parsing Function

- `_parse_llm_analysis(text, fallback_score)`: Extract decision from LLM response
  - Parses risk score, risk level, decision
  - Extracts risk factors (bullet points)
  - Falls back to rule-based if parsing fails

---

## Example Output

### High Risk Alert

```
🚨 ALERT ESCALATED

📊 RESULTS:
  Alert ID: TEST-ALERT-001
  Risk Score: 0.875 / 1.0
  Risk Level: CRITICAL
  Decision: escalate
  Requires Human Review: True
  Processing Duration: 2340ms
  Iterations: 4

⚠️  Risk Factors:
  • Very new account (2 months old)
  • Transaction amount 100.0x higher than average
  • Customer KYC not verified
  • High-risk alert type: account_takeover
  • High-risk country transaction: Nigeria

💡 Recommendation:
  Alert TEST-ALERT-001 escalated due to high risk score (0.875).
  Risk factors: Very new account, Transaction amount 100x higher,
  Customer KYC not verified, High-risk alert type.
  Immediate review required.
```

### Low Risk Alert

```
✅ ALERT CLOSED

📊 RESULTS:
  Alert ID: TEST-ALERT-002
  Risk Score: 0.185 / 1.0
  Risk Level: LOW
  Decision: auto_close
  Requires Human Review: False
  Processing Duration: 1890ms
  Iterations: 3

⚠️  Risk Factors:
  • Established customer profile

💡 Recommendation:
  Alert TEST-ALERT-002 auto-closed as false positive (0.185).
  No significant fraud indicators detected. Transaction appears legitimate.
```

---

## Running the Agent

### 1. Test with Built-in Scenarios

```bash
# Set environment (optional)
export ENVIRONMENT=development  # Uses Ollama

# Run the agent
python src/agents/fraud_triage_agent.py
```

### 2. Use as a Module

```python
import asyncio
from src.agents.fraud_triage_agent import FraudTriageAgent
from src.models.state import AlertType

async def main():
    agent = FraudTriageAgent(environment="demo")

    result = await agent.arun(
        alert_id="ALERT-001",
        alert_type=AlertType.LOCATION_MISMATCH,
        transaction_amount=4250.00,
        customer_id="CUST-001",
        transaction_country="NG",
    )

    print(f"Decision: {result['decision'].value}")
    print(f"Risk: {result['risk_score']:.2f}")

asyncio.run(main())
```

### 3. API Integration

```python
from fastapi import FastAPI
from src.agents.fraud_triage_agent import FraudTriageAgent

app = FastAPI()
agent = FraudTriageAgent(environment="production")

@app.post("/triage")
async def triage_alert(request: FraudAlertRequest):
    result = await agent.arun(
        alert_id=request.alert_id,
        alert_type=request.alert_type,
        transaction_amount=request.transaction_amount,
        customer_id=request.customer_id,
    )
    return result
```

---

## Configuration

### Risk Thresholds

```python
class AgentConfig:
    ESCALATE_THRESHOLD = 0.8   # High risk
    RECOMMEND_THRESHOLD = 0.4  # Medium risk
```

### LLM Settings

```python
class AgentConfig:
    MAX_TOKENS = 4096
    TEMPERATURE = 0.1  # Low for consistent analysis
```

---

## Checkpointing & State Persistence

The agent uses `MemorySaver` for state persistence:

```python
# Compiled with checkpointer
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Each alert has a thread_id
config = {"configurable": {"thread_id": alert_id}}
result = await app.ainvoke(initial_state, config=config)
```

**Benefits**:
- Resume workflow after interruption
- Human-in-the-loop support
- Debugging with full state history

---

## Next Steps

1. **Install dependencies**
   ```bash
   make install
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run test scenarios**
   ```bash
   python src/agents/fraud_triage_agent.py
   ```

4. **Integrate with API**
   - Update `src/api/main.py` to use `FraudTriageAgent`
   - Add endpoints for alert submission
   - Add human review endpoints

5. **Deploy**
   - Set `ENVIRONMENT=production`
   - Use OpenAI GPT-4
   - Enable LangSmith tracing
