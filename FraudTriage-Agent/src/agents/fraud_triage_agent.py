"""
Main LangGraph agent for fraud alert triage.

This module creates a complete fraud triage workflow using LangGraph,
integrating the state definitions and fraud detection tools.

Workflow:
    parse_alert → gather_context → analyze_risk → [conditional routing]
                                                          │
                                    ┌─────────────────────┼─────────────────────┐
                                    ▼                     ▼                     ▼
                               escalate_alert      recommend_action      auto_close_alert
                                    │                     │                     │
                                    └─────────────────────┴─────────────────────┘
                                                          ▼
                                                         END
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Import state and tools
from src.models.state import (
    AlertDecision,
    AlertType,
    FraudTriageState,
    RiskLevel,
    create_initial_state,
)
from src.tools.fraud_tools import (
    calculate_risk_score,
    check_watchlists,
    get_customer_profile,
    get_similar_alerts,
    get_transaction_history,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class AgentConfig:
    """Configuration for the fraud triage agent."""

    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")  # development, demo, production

    # Risk thresholds
    ESCALATE_THRESHOLD = 0.8  # Risk score > 0.8 escalates
    RECOMMEND_THRESHOLD = 0.4  # Risk score > 0.4 recommends action

    # LLM settings
    MAX_TOKENS = 4096
    TEMPERATURE = 0.1  # Low temperature for consistent fraud analysis


# =============================================================================
# LLM Factory
# =============================================================================

def create_llm(
    model: str | None = None,
    environment: str | None = None,
) -> BaseChatModel:
    """
    Create an LLM instance based on the environment.

    Environments:
        - development: Uses Ollama (local, free, no API key needed)
        - demo: Uses GLM-4.7 via OpenAI-compatible API
        - production: Uses OpenAI GPT-4

    Args:
        model: Optional model name override
        environment: Environment override (defaults to AGENT_CONFIG.ENVIRONMENT)

    Returns:
        Configured chat model instance

    Raises:
        ValueError: If environment is invalid or required API keys are missing

    Examples:
        >>> llm = create_llm()  # Uses ENVIRONMENT variable
        >>> llm = create_llm(environment="development")  # Force Ollama
        >>> llm = create_llm(environment="demo")  # Force GLM-4.7
        >>> llm = create_llm(environment="production")  # Force OpenAI
    """
    env = environment or AgentConfig.ENVIRONMENT

    if env == "development":
        # Use Ollama for local development
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model_name = model or os.getenv("OLLAMA_MODEL", "llama3.2")

        logger.info(f"🔧 Using Ollama (development): {model_name} at {base_url}")

        return ChatOpenAI(
            base_url=base_url,
            api_key="ollama",  # Ollama doesn't need a real API key
            model=model_name,
            temperature=AgentConfig.TEMPERATURE,
            max_tokens=AgentConfig.MAX_TOKENS,
        )

    elif env == "demo":
        # Use GLM-4.7 for demos (OpenAI-compatible API)
        api_key = os.getenv("GLM_API_KEY")
        if not api_key:
            raise ValueError(
                "GLM_API_KEY environment variable required for demo environment. "
                "Get one at: https://open.bigmodel.cn/"
            )

        base_url = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
        model_name = model or os.getenv("GLM_MODEL", "glm-4-plus")

        logger.info(f"🎯 Using GLM-4.7 (demo): {model_name}")

        return ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model_name,
            temperature=AgentConfig.TEMPERATURE,
            max_tokens=AgentConfig.MAX_TOKENS,
        )

    elif env == "production":
        # Use OpenAI GPT-4 for production
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable required for production environment. "
                "Get one at: https://platform.openai.com/"
            )

        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o")

        logger.info(f"🚀 Using OpenAI (production): {model_name}")

        return ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model_name,
            temperature=AgentConfig.TEMPERATURE,
            max_tokens=AgentConfig.MAX_TOKENS,
        )

    else:
        raise ValueError(
            f"Invalid environment: {env}. "
            f"Must be 'development', 'demo', or 'production'"
        )


# =============================================================================
# LLM Prompts
# =============================================================================

RISK_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert fraud analyst with 15+ years of experience in banking fraud detection.

Your role is to analyze fraud alerts and provide comprehensive risk assessments.

**Assessment Framework:**

1. **Risk Score (0.0-1.0):**
   - 0.00-0.25: LOW - Clear indicators of legitimate activity
   - 0.26-0.50: MEDIUM-LOW - Some concerns but likely legitimate
   - 0.51-0.75: MEDIUM-HIGH - Significant risk indicators
   - 0.76-1.00: HIGH - Strong fraud probability

2. **Risk Level:** Categorize as LOW, MEDIUM, HIGH, or CRITICAL

3. **Risk Factors:** List specific indicators (unusual location, new device, velocity, etc.)

4. **Decision:** Choose one:
   - auto_close: Clear false positive, no action needed
   - recommend_action: Needs monitoring or additional verification
   - escalate: Escalate to fraud team or block transaction

**Analysis Guidelines:**
- Consider customer history and patterns
- Evaluate transaction deviation from normal behavior
- Assess geographic and device anomalies
- Review velocity and timing patterns
- Weigh all factors before deciding

Provide a clear, well-reasoned assessment."""),
    ("human", """**FRAUD ALERT ANALYSIS**

**Alert Information:**
- Alert ID: {alert_id}
- Alert Type: {alert_type}
- Transaction Amount: ${transaction_amount:.2f}
- Customer ID: {customer_id}
- Alert Reason: {alert_reason}

**Customer Profile:**
{customer_profile}

**Transaction History:**
{transaction_summary}

**Risk Assessment:**
{risk_assessment}

**Watchlist Hits:**
{watchlist_results}

**Similar Alerts:**
{similar_alerts}

**Additional Context:**
- Transaction Country: {transaction_country}
- Transaction Device: {transaction_device}
- Account Age: {account_age_months} months
- Verification Status: {verification_status}

Please provide:
1. Risk Score (0.0-1.0)
2. Risk Level (LOW/MEDIUM/HIGH/CRITICAL)
3. Risk Factors (list specific indicators)
4. Decision (auto_close/recommend_action/escalate)
5. Detailed reasoning"""),
])


# =============================================================================
# Node Functions
# =============================================================================

async def parse_alert_node(state: FraudTriageState) -> FraudTriageState:
    """
    Parse and validate the incoming fraud alert.

    This is the entry point node. It extracts and validates information
    from the raw alert data and initializes the workflow.

    Args:
        state: Current workflow state

    Returns:
        Updated state with parsed information
    """
    alert_id = state["alert_id"]
    logger.info(f"[{alert_id}] 🔍 Parsing alert")

    # Increment iteration count
    state["iteration_count"] = state.get("iteration_count", 0) + 1

    # Extract and validate alert information
    alert_type = state.get("alert_type", AlertType.OTHER)
    transaction_amount = state.get("transaction_amount", 0.0)
    customer_id = state.get("customer_id", "")

    # Validate required fields
    if not customer_id:
        state["error_message"] = "Missing customer_id"
        logger.error(f"[{alert_id}] Missing customer_id in alert")
        return state

    if transaction_amount <= 0:
        state["error_message"] = f"Invalid transaction amount: {transaction_amount}"
        logger.error(f"[{alert_id}] Invalid transaction amount: {transaction_amount}")
        return state

    # Initialize processing timestamp
    state["processing_started"] = datetime.utcnow()

    # Add system message to track workflow
    state["messages"].append(
        SystemMessage(
            content=f"Alert {alert_id} received for processing. "
            f"Type: {alert_type.value}, Amount: ${transaction_amount:.2f}, Customer: {customer_id}"
        )
    )

    logger.info(
        f"[{alert_id}] ✅ Alert parsed successfully - "
        f"Type: {alert_type.value}, Amount: ${transaction_amount:.2f}"
    )

    return state


async def gather_context_node(state: FraudTriageState) -> FraudTriageState:
    """
    Gather context from multiple data sources using fraud tools.

    This node invokes tools to collect comprehensive information:
    - Customer profile and account details
    - Transaction history for pattern analysis
    - Watchlist and sanctions screening
    - Risk score calculation
    - Similar historical alerts

    Args:
        state: Current workflow state

    Returns:
        Updated state with gathered context
    """
    alert_id = state["alert_id"]
    logger.info(f"[{alert_id}] 📚 Gathering context")

    state["iteration_count"] = state.get("iteration_count", 0) + 1
    state["tools_used"] = state.get("tools_used", [])

    # Extract parameters
    customer_id = state.get("customer_id", "")
    transaction_amount = state.get("transaction_amount", 0.0)
    alert_type = state.get("alert_type", AlertType.OTHER)

    # Transaction details for watchlist check
    transaction_details = {
        "amount": transaction_amount,
        "country": state.get("transaction_country", "US"),
        "beneficiary_name": state.get("merchant_name"),
    }

    try:
        # 1. Get customer profile
        logger.debug(f"[{alert_id}] Fetching customer profile...")
        profile = get_customer_profile.invoke({"customer_id": customer_id})
        state["customer_profile"] = profile
        state["tools_used"].append("get_customer_profile")

        # 2. Get transaction history
        logger.debug(f"[{alert_id}] Fetching transaction history...")
        history = get_transaction_history.invoke({"customer_id": customer_id, "days": 30})
        state["transaction_history"] = history
        state["tools_used"].append("get_transaction_history")

        # 3. Check watchlists
        logger.debug(f"[{alert_id}] Checking watchlists...")
        watchlist_hits = check_watchlists.invoke({
            "customer_id": customer_id,
            "transaction_details": transaction_details
        })
        state["watchlist_hits"] = watchlist_hits
        state["tools_used"].append("check_watchlists")

        # 4. Calculate risk score
        logger.debug(f"[{alert_id}] Calculating risk score...")
        risk_assessment = calculate_risk_score.invoke({
            "customer_profile": profile,
            "transaction_amount": transaction_amount,
            "transaction_history": history,
            "alert_type": alert_type.value,
        })
        state["risk_score"] = risk_assessment["score"]
        state["confidence"] = risk_assessment["confidence"]
        state["tools_used"].append("calculate_risk_score")

        # 5. Get similar alerts
        logger.debug(f"[{alert_id}] Finding similar alerts...")
        similar = get_similar_alerts.invoke({
            "alert_type": alert_type.value,
            "customer_id": customer_id
        })
        state["similar_alerts"] = similar
        state["tools_used"].append("get_similar_alerts")

        # Add summary message
        state["messages"].append(
            AIMessage(
                content=f"Context gathered from {len(state['tools_used'])} sources. "
                f"Risk score: {state['risk_score']:.3f}, "
                f"Watchlist hits: {len(watchlist_hits)}, "
                f"Similar alerts: {len(similar)}"
            )
        )

        logger.info(
            f"[{alert_id}] ✅ Context gathered - "
            f"Risk Score: {state['risk_score']:.3f}, "
            f"Tools: {', '.join(state['tools_used'])}"
        )

    except Exception as e:
        logger.error(f"[{alert_id}] ❌ Error gathering context: {e}")
        state["error_message"] = str(e)
        # Set conservative defaults
        state["risk_score"] = 0.5
        state["confidence"] = 0.0

    return state


async def analyze_risk_node(state: FraudTriageState) -> FraudTriageState:
    """
    Analyze risk using LLM with gathered context.

    This node uses the LLM to synthesize all gathered information
    and produce a comprehensive risk assessment with decision.

    Args:
        state: Current workflow state

    Returns:
        Updated state with LLM analysis and decision
    """
    alert_id = state["alert_id"]
    logger.info(f"[{alert_id}] 🤖 Analyzing risk with LLM")

    state["iteration_count"] = state.get("iteration_count", 0) + 1

    # Format context for LLM
    customer_profile = state.get("customer_profile", {})
    transaction_history = state.get("transaction_history", [])
    risk_assessment = state.get("risk_score", 0.0)
    watchlist_hits = state.get("watchlist_hits", [])
    similar_alerts = state.get("similar_alerts", [])

    # Build formatted strings
    customer_summary = _format_customer_profile(customer_profile)
    transaction_summary = _format_transaction_history(transaction_history)
    risk_summary = _format_risk_assessment(risk_assessment, customer_profile)
    watchlist_summary = _format_watchlist_hits(watchlist_hits)
    similar_summary = _format_similar_alerts(similar_alerts)

    # Get additional context
    transaction_country = state.get("transaction_country", "Unknown")
    transaction_device = state.get("transaction_device_id", "Unknown")
    account_age_months = customer_profile.get("account_age_months", 0)
    verification_status = customer_profile.get("verification_status", "unknown")
    alert_reason = state.get("alert_reason", "Not provided")

    # Create LLM
    try:
        llm = create_llm()

        # Generate prompt
        messages = RISK_ANALYSIS_PROMPT.format_messages(
            alert_id=alert_id,
            alert_type=state.get("alert_type", AlertType.OTHER).value,
            transaction_amount=state.get("transaction_amount", 0.0),
            customer_id=state.get("customer_id", ""),
            alert_reason=alert_reason,
            customer_profile=customer_summary,
            transaction_summary=transaction_summary,
            risk_assessment=risk_summary,
            watchlist_results=watchlist_summary,
            similar_alerts=similar_summary,
            transaction_country=transaction_country,
            transaction_device=transaction_device,
            account_age_months=account_age_months,
            verification_status=verification_status,
        )

        # Invoke LLM
        response = await llm.ainvoke(messages)
        response_text = response.content

        # Parse LLM response
        analysis = _parse_llm_analysis(response_text, state["risk_score"])

        # Update state with LLM analysis
        state["risk_level"] = analysis["risk_level"]
        state["risk_factors"] = analysis["risk_factors"]
        state["decision"] = analysis["decision"]
        state["recommendation"] = analysis["recommendation"]
        state["requires_human_review"] = analysis["requires_human_review"]

        # Add AI message
        state["messages"].append(AIMessage(content=response_text))

        logger.info(
            f"[{alert_id}] ✅ LLM analysis complete - "
            f"Risk Level: {state['risk_level'].value}, "
            f"Decision: {state['decision'].value}, "
            f"Human Review: {state['requires_human_review']}"
        )

    except Exception as e:
        logger.error(f"[{alert_id}] ❌ Error in LLM analysis: {e}")
        state["error_message"] = str(e)

        # Fallback to rule-based decision
        risk_score = state.get("risk_score", 0.0)
        if risk_score > AgentConfig.ESCALATE_THRESHOLD:
            state["decision"] = AlertDecision.ESCALATE
            state["risk_level"] = RiskLevel.HIGH
        elif risk_score > AgentConfig.RECOMMEND_THRESHOLD:
            state["decision"] = AlertDecision.REVIEW_REQUIRED
            state["risk_level"] = RiskLevel.MEDIUM
        else:
            state["decision"] = AlertDecision.AUTO_CLOSE
            state["risk_level"] = RiskLevel.LOW

        state["requires_human_review"] = state["decision"] != AlertDecision.AUTO_CLOSE
        state["recommendation"] = f"Rule-based decision (LLM error: {str(e)[:50]})"

    return state


async def escalate_alert_node(state: FraudTriageState) -> FraudTriageState:
    """
    Handle high-risk alerts that require escalation.

    This node is called for alerts with risk score > 0.8.
    It prepares the alert for escalation to the fraud team.

    Args:
        state: Current workflow state

    Returns:
        Updated state with escalation details
    """
    alert_id = state["alert_id"]
    logger.info(f"[{alert_id}] 🚨 Escalating alert (HIGH RISK)")

    state["iteration_count"] = state.get("iteration_count", 0) + 1
    state["processing_completed"] = datetime.utcnow()

    # Calculate duration
    if state.get("processing_started"):
        duration = (state["processing_completed"] - state["processing_started"]).total_seconds()
        state["processing_duration_ms"] = int(duration * 1000)

    # Set escalation details
    state["decision"] = AlertDecision.ESCALATE
    state["requires_human_review"] = True

    # Build escalation message
    escalation_reason = (
        f"Alert {alert_id} escalated due to high risk score ({state['risk_score']:.3f}). "
        f"Risk factors: {', '.join(state.get('risk_factors', ['Multiple high-risk indicators'])[:5])}. "
        f"Immediate review required."
    )

    state["recommendation"] = escalation_reason

    # Add message
    state["messages"].append(
        AIMessage(
            content=f"🚨 ALERT ESCALATED: {escalation_reason}"
        )
    )

    logger.warning(f"[{alert_id}] ⚠️  Alert escalated to fraud team")

    return state


async def recommend_action_node(state: FraudTriageState) -> FraudTriageState:
    """
    Handle medium-risk alerts that require action or monitoring.

    This node is called for alerts with risk score between 0.4 and 0.8.
    It recommends specific actions for the fraud team.

    Args:
        state: Current workflow state

    Returns:
        Updated state with action recommendations
    """
    alert_id = state["alert_id"]
    logger.info(f"[{alert_id}] 📋 Recommending action (MEDIUM RISK)")

    state["iteration_count"] = state.get("iteration_count", 0) + 1
    state["processing_completed"] = datetime.utcnow()

    # Calculate duration
    if state.get("processing_started"):
        duration = (state["processing_completed"] - state["processing_started"]).total_seconds()
        state["processing_duration_ms"] = int(duration * 1000)

    # Set recommendation details
    state["decision"] = AlertDecision.REVIEW_REQUIRED
    state["requires_human_review"] = True

    # Build recommendation message
    recommendation = (
        f"Alert {alert_id} requires review (risk score: {state['risk_score']:.3f}). "
        f"Recommend: {state.get('recommendation', 'Monitor for additional indicators')}. "
        f"Factors: {', '.join(state.get('risk_factors', [])[:3])}."
    )

    state["recommendation"] = recommendation

    # Add message
    state["messages"].append(
        AIMessage(
            content=f"📋 ACTION RECOMMENDED: {recommendation}"
        )
    )

    logger.info(f"[{alert_id}] 📝 Action recommended for alert")

    return state


async def auto_close_alert_node(state: FraudTriageState) -> FraudTriageState:
    """
    Handle low-risk alerts that can be automatically closed.

    This node is called for alerts with risk score <= 0.4.
    It closes the alert as a false positive.

    Args:
        state: Current workflow state

    Returns:
        Updated state with auto-close details
    """
    alert_id = state["alert_id"]
    logger.info(f"[{alert_id}] ✅ Auto-closing alert (LOW RISK)")

    state["iteration_count"] = state.get("iteration_count", 0) + 1
    state["processing_completed"] = datetime.utcnow()

    # Calculate duration
    if state.get("processing_started"):
        duration = (state["processing_completed"] - state["processing_started"]).total_seconds()
        state["processing_duration_ms"] = int(duration * 1000)

    # Set auto-close details
    state["decision"] = AlertDecision.AUTO_CLOSE
    state["requires_human_review"] = False

    # Build closure message
    closure_reason = (
        f"Alert {alert_id} auto-closed as false positive (risk score: {state['risk_score']:.3f}). "
        f"No significant fraud indicators detected. Transaction appears legitimate."
    )

    state["recommendation"] = closure_reason

    # Add message
    state["messages"].append(
        AIMessage(
            content=f"✅ ALERT CLOSED: {closure_reason}"
        )
    )

    logger.info(f"[{alert_id}] ✅ Alert automatically closed")

    return state


# =============================================================================
# Routing Function
# =============================================================================

def route_decision(state: FraudTriageState) -> Literal["escalate", "recommend", "auto_close"]:
    """
    Route the alert based on risk score.

    Routing logic:
        - Risk score > 0.8 → escalate (high risk, immediate action)
        - Risk score > 0.4 → recommend (medium risk, needs review)
        - Risk score <= 0.4 → auto_close (low risk, false positive)

    Args:
        state: Current workflow state

    Returns:
        Next node name: "escalate", "recommend", or "auto_close"
    """
    alert_id = state["alert_id"]
    risk_score = state.get("risk_score", 0.0)

    logger.debug(f"[{alert_id}] 🔀 Routing decision: risk_score={risk_score:.3f}")

    if risk_score > AgentConfig.ESCALATE_THRESHOLD:
        logger.info(f"[{alert_id}] → Routing to escalate (risk: {risk_score:.3f} > {AgentConfig.ESCALATE_THRESHOLD})")
        return "escalate"

    elif risk_score > AgentConfig.RECOMMEND_THRESHOLD:
        logger.info(f"[{alert_id}] → Routing to recommend (risk: {risk_score:.3f} > {AgentConfig.RECOMMEND_THRESHOLD})")
        return "recommend"

    else:
        logger.info(f"[{alert_id}] → Routing to auto_close (risk: {risk_score:.3f} <= {AgentConfig.RECOMMEND_THRESHOLD})")
        return "auto_close"


# =============================================================================
# Helper Functions
# =============================================================================

def _format_customer_profile(profile: dict[str, Any]) -> str:
    """Format customer profile for LLM."""
    if not profile or isinstance(profile, dict) and profile.get("error"):
        return "No customer profile available"

    return f"""
Name: {profile.get('name', 'Unknown')}
Account Age: {profile.get('account_age_months', 0)} months
Verification Status: {profile.get('verification_status', 'unknown')}
Risk Rating: {profile.get('risk_rating', 'unknown')}
Average Transaction: ${profile.get('average_transaction', 0):.2f}
Total Transactions YTD: {profile.get('total_transactions_ytd', 0)}
Country: {profile.get('country', 'Unknown')}
Occupation: {profile.get('occupation', 'Unknown')}
Previous Fraud Cases: {profile.get('previous_fraud_cases', 0)}
False Positives: {profile.get('false_positive_count', 0)}
""".strip()


def _format_transaction_history(history: list[dict[str, Any]]) -> str:
    """Format transaction history for LLM."""
    if not history:
        return "No transaction history available"

    total_amount = sum(t.get("amount", 0) for t in history)
    unique_merchants = len(set(t.get("merchant", "") for t in history))
    unique_locations = len(set(t.get("location", "") for t in history))

    # Show recent transactions
    recent_txns = "\n".join(
        f"  - {t.get('date', 'Unknown')}: {t.get('merchant', 'Unknown')} - ${t.get('amount', 0):.2f} ({t.get('location', 'Unknown')})"
        for t in history[:5]
    )

    return f"""
Total Transactions: {len(history)}
Total Volume: ${total_amount:.2f}
Unique Merchants: {unique_merchants}
Unique Locations: {unique_locations}

Recent Transactions:
{recent_txns}
""".strip()


def _format_risk_assessment(risk_score: float, profile: dict[str, Any]) -> str:
    """Format risk assessment for LLM."""
    avg_txn = profile.get("average_transaction", 0)
    current_txn = 0  # Will be filled from context

    if avg_txn > 0 and current_txn > 0:
        ratio = current_txn / avg_txn
        return f"""
Calculated Risk Score: {risk_score:.3f} (0.0-1.0)
Transaction Deviation: {ratio:.1f}x from average
""".strip()

    return f"Calculated Risk Score: {risk_score:.3f} (0.0-1.0)"


def _format_watchlist_hits(hits: list[dict[str, Any]]) -> str:
    """Format watchlist hits for LLM."""
    if not hits:
        return "No watchlist hits detected"

    formatted = [f"- {hit['list_name']} (confidence: {hit['match_confidence']:.0%})" for hit in hits]
    return "\n".join(formatted)


def _format_similar_alerts(alerts: list[dict[str, Any]]) -> str:
    """Format similar alerts for LLM."""
    if not alerts:
        return "No similar alerts found"

    formatted = []
    for alert in alerts[:5]:
        outcome_icon = "✅" if alert['outcome'] == "false_positive" else "🔴"
        formatted.append(
            f"- {alert['alert_id']} ({alert['date']}): {alert['outcome']} - {alert.get('reason', 'N/A')}"
        )

    return "\n".join(formatted)


def _parse_llm_analysis(text: str, fallback_risk_score: float) -> dict[str, Any]:
    """
    Parse LLM response into structured analysis.

    Args:
        text: LLM response text
        fallback_risk_score: Risk score to use if parsing fails

    Returns:
        Dictionary with parsed analysis
    """
    import re

    # Defaults
    analysis = {
        "risk_score": fallback_risk_score,
        "risk_level": RiskLevel.MEDIUM,
        "risk_factors": ["Analysis completed"],
        "decision": AlertDecision.REVIEW_REQUIRED,
        "recommendation": text[:500],  # Use first 500 chars as recommendation
        "requires_human_review": True,
    }

    # Extract risk score
    score_match = re.search(r'risk score[:\s]+([0-9.]+)', text, re.IGNORECASE)
    if score_match:
        try:
            score = float(score_match.group(1))
            # Normalize to 0-1 if needed
            if score > 1:
                score = score / 100
            analysis["risk_score"] = min(max(score, 0), 1)
        except ValueError:
            pass

    # Determine risk level
    score = analysis["risk_score"]
    if score >= 0.76:
        analysis["risk_level"] = RiskLevel.CRITICAL
    elif score >= 0.51:
        analysis["risk_level"] = RiskLevel.HIGH
    elif score >= 0.26:
        analysis["risk_level"] = RiskLevel.MEDIUM
    else:
        analysis["risk_level"] = RiskLevel.LOW

    # Extract decision
    if re.search(r'auto.?close|close|false positive|legitimate', text, re.IGNORECASE):
        analysis["decision"] = AlertDecision.AUTO_CLOSE
        analysis["requires_human_review"] = False
    elif re.search(r'escalate|block|critical|fraud', text, re.IGNORECASE):
        analysis["decision"] = AlertDecision.ESCALATE
        analysis["requires_human_review"] = True
    else:
        analysis["decision"] = AlertDecision.REVIEW_REQUIRED
        analysis["requires_human_review"] = True

    # Extract risk factors (look for lists)
    factors = re.findall(r'[-•]\s*(.+?)(?:\n|$)', text)
    if factors:
        analysis["risk_factors"] = [f.strip() for f in factors[:10]]

    return analysis


# =============================================================================
# Graph Builder
# =============================================================================

def create_fraud_triage_graph() -> StateGraph:
    """
    Create the LangGraph workflow for fraud alert triage.

    Builds the complete graph with all nodes, edges, and routing logic.

    Returns:
        Compiled StateGraph ready for execution
    """
    logger.info("🔨 Building fraud triage graph")

    # Create the graph
    workflow = StateGraph(FraudTriageState)

    # Add all nodes
    workflow.add_node("parse_alert", parse_alert_node)
    workflow.add_node("gather_context", gather_context_node)
    workflow.add_node("analyze_risk", analyze_risk_node)
    workflow.add_node("escalate", escalate_alert_node)
    workflow.add_node("recommend", recommend_action_node)
    workflow.add_node("auto_close", auto_close_alert_node)

    logger.info("✅ Added 6 nodes: parse_alert, gather_context, analyze_risk, escalate, recommend, auto_close")

    # Set entry point
    workflow.set_entry_point("parse_alert")

    # Define linear flow
    workflow.add_edge("parse_alert", "gather_context")
    workflow.add_edge("gather_context", "analyze_risk")

    # Define conditional routing after risk analysis
    workflow.add_conditional_edges(
        source="analyze_risk",
        path=route_decision,
        path_map={
            "escalate": "escalate",
            "recommend": "recommend",
            "auto_close": "auto_close",
        },
    )

    # All decision nodes lead to END
    workflow.add_edge("escalate", END)
    workflow.add_edge("recommend", END)
    workflow.add_edge("auto_close", END)

    # Compile with checkpointer for persistence
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    logger.info("✅ Fraud triage graph compiled with MemorySaver checkpointer")

    return app


# =============================================================================
# Main Agent Class
# =============================================================================

class FraudTriageAgent:
    """
    Main agent class for fraud alert triage.

    Provides a simple interface for running the fraud triage workflow.
    """

    def __init__(self, environment: str | None = None):
        """
        Initialize the fraud triage agent.

        Args:
            environment: Environment for LLM (development/demo/production)
        """
        if environment:
            AgentConfig.ENVIRONMENT = environment

        self.graph = create_fraud_triage_graph()
        self.environment = AgentConfig.ENVIRONMENT

        logger.info(f"🚀 FraudTriageAgent initialized (environment: {self.environment})")

    async def arun(
        self,
        alert_id: str,
        alert_type: str | AlertType,
        transaction_amount: float,
        customer_id: str,
        **kwargs: Any,
    ) -> FraudTriageState:
        """
        Run the fraud triage workflow asynchronously.

        Args:
            alert_id: Unique alert identifier
            alert_type: Type of fraud alert (string or AlertType enum)
            transaction_amount: Amount of the flagged transaction
            customer_id: Customer identifier
            **kwargs: Additional optional state fields

        Returns:
            Final workflow state after completion
        """
        # Convert alert_type to enum if needed
        if isinstance(alert_type, str):
            alert_type = AlertType(alert_type)

        # Create initial state
        initial_state = create_initial_state(
            alert_id=alert_id,
            alert_type=alert_type,
            transaction_amount=transaction_amount,
            customer_id=customer_id,
            **kwargs,
        )

        logger.info(f"🚀 Starting workflow for alert {alert_id}")

        # Configure run
        config = {
            "configurable": {
                "thread_id": alert_id,
            }
        }

        # Run workflow
        try:
            result = await self.graph.ainvoke(initial_state, config=config)
            logger.info(f"✅ Workflow completed for alert {alert_id}")
            return result
        except Exception as e:
            logger.error(f"❌ Workflow failed for alert {alert_id}: {e}")
            initial_state["error_message"] = str(e)
            return initial_state

    def run(
        self,
        alert_id: str,
        alert_type: str | AlertType,
        transaction_amount: float,
        customer_id: str,
        **kwargs: Any,
    ) -> FraudTriageState:
        """
        Run the fraud triage workflow synchronously.

        Args:
            alert_id: Unique alert identifier
            alert_type: Type of fraud alert
            transaction_amount: Amount of the flagged transaction
            customer_id: Customer identifier
            **kwargs: Additional optional fields

        Returns:
            Final workflow state
        """
        # Convert alert_type to enum if needed
        if isinstance(alert_type, str):
            alert_type = AlertType(alert_type)

        # Create initial state
        initial_state = create_initial_state(
            alert_id=alert_id,
            alert_type=alert_type,
            transaction_amount=transaction_amount,
            customer_id=customer_id,
            **kwargs,
        )

        logger.info(f"🚀 Starting workflow (sync) for alert {alert_id}")

        # Configure run
        config = {
            "configurable": {
                "thread_id": alert_id,
            }
        }

        # Run workflow
        try:
            result = self.graph.invoke(initial_state, config=config)
            logger.info(f"✅ Workflow completed for alert {alert_id}")
            return result
        except Exception as e:
            logger.error(f"❌ Workflow failed for alert {alert_id}: {e}")
            initial_state["error_message"] = str(e)
            return initial_state


# =============================================================================
# Main Block for Testing
# =============================================================================

async def main():
    """
    Main block for testing the fraud triage agent.

    Runs a sample alert through the workflow and prints results.
    """
    from src.models.state import AlertType

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("\n" + "=" * 70)
    print("  FRAUD TRIAGE AGENT - TEST RUN")
    print("=" * 70 + "\n")

    # Test scenarios
    scenarios = [
        {
            "name": "HIGH RISK - Account Takeover",
            "alert_id": "TEST-ALERT-001",
            "alert_type": AlertType.ACCOUNT_TAKEOVER,
            "transaction_amount": 7500.00,
            "customer_id": "CUST-004",  # New, unverified customer
            "transaction_country": "NG",
            "transaction_device_id": "DEVICE-NEW-999",
            "alert_reason": "Transaction from high-risk country with new device",
        },
        {
            "name": "LOW RISK - Legitimate Transaction",
            "alert_id": "TEST-ALERT-002",
            "alert_type": AlertType.UNUSUAL_AMOUNT,
            "transaction_amount": 350.00,
            "customer_id": "CUST-001",  # Established, verified customer
            "transaction_country": "US",
            "transaction_device_id": "DEVICE-001-APPLE",
            "alert_reason": "Transaction slightly above average",
        },
    ]

    for scenario in scenarios:
        print(f"\n{'=' * 70}")
        print(f"  Scenario: {scenario['name']}")
        print(f"{'=' * 70}\n")

        # Create agent
        agent = FraudTriageAgent(environment="development")

        # Run workflow
        result = await agent.arun(
            alert_id=scenario["alert_id"],
            alert_type=scenario["alert_type"],
            transaction_amount=scenario["transaction_amount"],
            customer_id=scenario["customer_id"],
            transaction_country=scenario.get("transaction_country"),
            transaction_device_id=scenario.get("transaction_device_id"),
            alert_reason=scenario.get("alert_reason"),
        )

        # Print results
        print(f"\n📊 RESULTS:")
        print(f"  Alert ID: {result['alert_id']}")
        print(f"  Risk Score: {result.get('risk_score', 0):.3f} / 1.0")
        print(f"  Risk Level: {result.get('risk_level', 'UNKNOWN').value}")
        print(f"  Decision: {result.get('decision', 'UNKNOWN').value}")
        print(f"  Requires Human Review: {result.get('requires_human_review', False)}")
        print(f"  Processing Duration: {result.get('processing_duration_ms', 0)}ms")
        print(f"  Iterations: {result.get('iteration_count', 0)}")

        if result.get("risk_factors"):
            print(f"\n⚠️  Risk Factors:")
            for factor in result["risk_factors"][:5]:
                print(f"    • {factor}")

        print(f"\n💡 Recommendation:")
        print(f"  {result.get('recommendation', 'N/A')}")

        if result.get("error_message"):
            print(f"\n❌ Error: {result['error_message']}")

        print()


if __name__ == "__main__":
    # Run the test
    asyncio.run(main())
