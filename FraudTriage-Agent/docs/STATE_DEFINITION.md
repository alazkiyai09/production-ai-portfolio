# FraudTriage-Agent State Definition Guide

## Overview

The `FraudTriageState` TypedDict is the core state object that flows through the LangGraph workflow. It contains all information needed to process a fraud alert from submission to final decision.

## State Schema

```python
class FraudTriageState(TypedDict, total=False):
    # Required Input Fields
    alert_id: str
    alert_type: AlertType
    transaction_amount: float
    customer_id: str

    # Message History (LangGraph managed with add_messages reducer)
    messages: Annotated[list[BaseMessage], add_messages]

    # Context Fields (populated by tools)
    customer_profile: CustomerProfile | dict | None
    transaction_history: list[TransactionInfo] | list[dict] | None
    watchlist_hits: list[WatchlistHit] | list[dict] | None
    similar_alerts: list[SimilarAlert] | list[dict] | None
    device_info: DeviceInfo | dict | None
    transaction_patterns: dict | None
    customer_risk_history: dict | None
    ip_reputation: dict | None

    # Analysis Fields (output from risk assessment)
    risk_score: float  # 0-100
    risk_level: RiskLevel  # LOW, MEDIUM, HIGH, CRITICAL
    risk_factors: list[str]
    confidence: float  # 0-1

    # Decision Fields
    decision: AlertDecision
    recommendation: str
    requires_human_review: bool

    # Human Review Fields
    human_review_input: str | None
    human_review_decision: AlertDecision | None
    human_review_reasoning: str | None
    human_reviewer_id: str | None
    human_reviewer_name: str | None
    human_review_timestamp: datetime | None

    # Metadata
    processing_started: datetime | None
    processing_completed: datetime | None
    processing_duration_ms: int | None
    iteration_count: int
    error_message: str | None
    model_used: str | None
```

## Enums

### RiskLevel
- `LOW` (0-25): Clear indicators of legitimate activity
- `MEDIUM` (26-50): Some concerns but likely legitimate
- `HIGH` (51-75): Significant risk indicators
- `CRITICAL` (76-100): Strong fraud probability

### AlertDecision
- `AUTO_CLOSE`: False positive, close automatically
- `REVIEW_REQUIRED`: Needs human analyst review
- `ESCALATE`: Escalate to fraud investigation team
- `BLOCK_TRANSACTION`: Block the transaction immediately

### AlertType
- `UNUSUAL_AMOUNT`: Amount differs from typical pattern
- `VELOCITY`: Too many transactions in short time
- `LOCATION_MISMATCH`: Transaction from unusual location
- `DEVICE_CHANGE`: Transaction from new device
- `ACCOUNT_TAKEOVER`: Signs of account compromise

## Workflow Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        PARSE_ALERT                              │
│  - Validate alert data                                          │
│  - Extract key fields                                           │
│  - Log processing start                                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      GATHER_CONTEXT                             │
│  - Fetch customer profile                                       │
│  - Get transaction history                                      │
│  - Analyze transaction patterns                                 │
│  - Check device fingerprint                                     │
│  - Check IP reputation                                          │
│  - Get customer risk history                                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ASSESS_RISK                               │
│  - Invoke LLM with alert + context                              │
│  - Calculate risk score (0-100)                                 │
│  - Identify risk factors                                        │
│  - Determine risk level                                         │
│  - Make initial decision                                        │
│  - Set requires_human_review flag                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                    ┌───────┴───────┐
                    │               │
              requires_review?  no
                    │               │
                    ▼               ▼
    ┌───────────────────────┐   ┌────────────┐
    │    HUMAN_REVIEW       │   │  FINALIZE  │
    │  - Wait for analyst   │   │  - Compute │
    │    decision           │   │    metrics │
    │  - Process decision   │   └────────────┘
    │  - Update state       │          │
    └───────────┬───────────┘          │
                │                       │
                └───────────┬───────────┘
                            ▼
                  ┌────────────────┐
                  │    RETURN      │
                  │  Final State   │
                  └────────────────┘
```

## Usage Examples

### Creating Initial State

```python
from src.models.state import create_initial_state, AlertType

state = create_initial_state(
    alert_id="ALERT-001",
    alert_type=AlertType.LOCATION_MISMATCH,
    transaction_amount=4250.00,
    customer_id="CUST-12345",
    account_id="ACC-67890",
    transaction_country="NG",
    transaction_device_id="DEVICE-NEW-999",
)
```

### Running Workflow

```python
from src.agents.workflow import triage_alert, AlertType

result = await triage_alert(
    alert_id="ALERT-001",
    alert_type=AlertType.LOCATION_MISMATCH,
    transaction_amount=4250.00,
    customer_id="CUST-12345",
    account_id="ACC-67890",
    transaction_country="NG",
)

print(f"Risk Score: {result['risk_score']}")
print(f"Decision: {result['decision'].value}")
print(f"Recommendation: {result['recommendation']}")
```

### Submitting Human Review

```python
from src.models.state import AlertDecision

# Get current state
workflow = get_workflow()
state = workflow.get_state("ALERT-001")

# Update with human review
workflow.update_state(
    alert_id="ALERT-001",
    updates={
        "human_review_decision": AlertDecision.ESCALATE,
        "human_reviewer_id": "ANALYST-001",
        "human_reviewer_name": "Jane Smith",
        "human_review_reasoning": "Customer confirmed unauthorized transaction",
        "agreed_with_agent": True,
    }
)
```

## State Transitions

| Node | Input State | Output State | Key Changes |
|------|-------------|--------------|-------------|
| parse_alert | Initial alert data | Parsed alert | iteration_count++, messages added |
| gather_context | Parsed alert | Alert + context | All context fields populated |
| assess_risk | Alert + context | Complete analysis | risk_score, decision, requires_human_review set |
| human_review | Analysis awaiting review | Final decision | human_review_* fields populated |
| finalize | Any state | Complete | processing_completed, duration computed |

## Error Handling

If an error occurs during processing:
- `error_message` field is populated
- Processing stops at current node
- State is preserved for debugging
- Workflow can be resumed after fixing the issue

## Best Practices

1. **Type Safety**: Use the enum types instead of strings
   ```python
   # Good
   state["decision"] = AlertDecision.ESCALATE

   # Bad
   state["decision"] = "escalate"
   ```

2. **Optional Fields**: Check for None before accessing
   ```python
   if state.get("customer_profile"):
       profile = state["customer_profile"]
   ```

3. **Messages**: LangGraph manages message reduction automatically
   ```python
   # Just append, don't worry about size
   state["messages"].append(AIMessage(content="Analysis complete"))
   ```

4. **Human Review**: Always set both decision and reasoning
   ```python
   state["human_review_decision"] = AlertDecision.AUTO_CLOSE
   state["human_review_reasoning"] = "Customer verified transaction"
   state["human_reviewer_id"] = "ANALYST-001"
   ```
