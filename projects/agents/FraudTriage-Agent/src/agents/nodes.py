"""
LangGraph node definitions for the fraud triage workflow.

Each node represents a step in the fraud alert triage process.
"""

import logging
from datetime import datetime
from typing import Any, Literal, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config.settings import settings
from src.models.agent import RiskAssessment
from src.tools import create_tool_registry
from src.tools.customer_tools import get_customer_profile, get_customer_risk_history
from src.tools.device_tools import check_ip_reputation, get_device_fingerprint
from src.tools.transaction_tools import (
    analyze_transaction_patterns,
    get_transaction_by_id,
    get_transaction_history,
)

from .state import AgentState

logger = logging.getLogger(__name__)

# Initialize LLM
def create_llm():
    """Create LLM instance based on configuration."""
    # Try GLM first (via OpenAI-compatible API)
    if settings.glm_api_key:
        return ChatOpenAI(
            base_url=settings.glm_base_url,
            api_key=settings.glm_api_key,
            model=settings.glm_model,
            temperature=settings.glm_temperature,
            max_tokens=settings.glm_max_tokens,
        )
    # Fallback to OpenAI
    elif settings.openai_api_key:
        return ChatOpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens,
        )
    else:
        # For development mode, allow Ollama without API key
        if os.getenv("ENVIRONMENT") == "development" or settings.mock_external_apis:
            return ChatOpenAI(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                api_key="ollama",  # Ollama doesn't need a real API key
                model=os.getenv("OLLAMA_MODEL", "llama3.2"),
                temperature=0.1,
            )
        raise ValueError("No LLM API key configured. Set GLM_API_KEY, OPENAI_API_KEY, or ENVIRONMENT=development")


def get_llm():
    """Get or create the LLM instance (lazy loading)."""
    global _llm
    if _llm is None:
        _llm = create_llm()
    return _llm


_llm = None

# Risk assessment prompt
RISK_ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a fraud analyst AI assistant. Analyze the provided fraud alert and context to assess risk.

Provide your assessment in the following format:
- Risk Score: 0-100 (higher = more risky)
- Risk Factors: List specific indicators of fraud or legitimacy
- Reasoning: Explain your analysis
- Confidence: 0.0-1.0 (how confident are you in this assessment)
- Suggested Action: One of: auto_close, monitor, escalate_for_review, create_case

Risk Assessment Guidelines:
- 0-30: Low risk - likely legitimate, can auto-close
- 31-60: Medium risk - monitor for additional alerts
- 61-80: High risk - escalate for human review
- 81-100: Very high risk - create case immediately

Consider these factors:
- Transaction amount vs. customer history
- Location (unusual country, high-risk region)
- Device (new device, known compromised device)
- Customer risk history (previous fraud, false positives)
- Transaction patterns (velocity, timing, merchant types)"""),
    ("human", """Alert Details:
{alert_data}

Context:
{context_summary}

Provide your risk assessment."""),
])


async def parse_alert_node(state: AgentState) -> AgentState:
    """
    Parse and validate the incoming fraud alert.

    This is the first node in the workflow. It extracts relevant
    information from the raw alert data.
    """
    logger.info(f"Parsing alert {state['alert_id']}")

    alert_data = state["alert_data"]

    # Extract key information
    parsed = {
        "parsed_alert": {
            "alert_id": state["alert_id"],
            "alert_type": alert_data.get("alert_type", "unknown"),
            "account_id": alert_data.get("account_id", ""),
            "customer_id": alert_data.get("customer_id", ""),
            "transaction_id": alert_data.get("transaction", {}).get("transaction_id", ""),
            "transaction_amount": alert_data.get("transaction", {}).get("amount", 0),
            "transaction_location": alert_data.get("transaction", {}).get("location_country", ""),
            "created_at": alert_data.get("created_at", datetime.now().isoformat()),
            "rule_id": alert_data.get("rule_id"),
            "alert_reason": alert_data.get("alert_reason", ""),
        }
    }

    # Add system message
    state["messages"].append(SystemMessage(content=f"Alert {state['alert_id']} received for processing."))

    # Update state
    state["iteration_count"] += 1
    state["alert_data"].update(parsed)

    return state


async def gather_context_node(state: AgentState) -> AgentState:
    """
    Gather context from multiple data sources.

    This node fetches transaction history, customer profile,
    device fingerprint, and other relevant information.
    """
    logger.info(f"Gathering context for alert {state['alert_id']}")

    alert_data = state["alert_data"]
    parsed = alert_data.get("parsed_alert", {})

    account_id = parsed.get("account_id") or alert_data.get("account_id", "")
    customer_id = parsed.get("customer_id") or alert_data.get("customer_id", "")
    transaction_id = parsed.get("transaction_id") or alert_data.get("transaction", {}).get("transaction_id", "")

    # Gather context in parallel
    context = {}

    # Transaction history
    if account_id:
        txn_history = await get_transaction_history(account_id, days=90, limit=100)
        context["transaction_history"] = txn_history

        # Transaction patterns
        if transaction_id:
            patterns = await analyze_transaction_patterns(account_id, transaction_id)
            context["transaction_patterns"] = patterns

    # Customer profile
    if customer_id:
        profile = await get_customer_profile(customer_id)
        context["customer_profile"] = profile

        # Risk history
        risk_history = await get_customer_risk_history(customer_id, months=12)
        context["customer_risk_history"] = risk_history

    # Device and IP
    transaction = alert_data.get("transaction", {})
    device_id = transaction.get("device_id")
    ip_address = transaction.get("ip_address")

    if device_id:
        device_info = await get_device_fingerprint(device_id)
        context["device_fingerprint"] = device_info

    if ip_address:
        ip_info = await check_ip_reputation(ip_address)
        context["ip_reputation"] = ip_info

    # Update state with gathered context
    state["transaction_history"] = context.get("transaction_history", {}).get("transactions", [])
    state["customer_profile"] = context.get("customer_profile", {})
    state["device_fingerprint"] = context.get("device_fingerprint", {})
    state["similar_alerts"] = context.get("customer_risk_history", {}).get("alerts", [])

    # Add human-readable summary
    context_summary = _format_context_summary(context)
    state["messages"].append(AIMessage(content=f"Context gathered:\n{context_summary}"))

    state["iteration_count"] += 1

    return state


async def assess_risk_node(state: AgentState) -> AgentState:
    """
    Assess risk using LLM analysis.

    This node uses the LLM to analyze the alert and gathered context,
    producing a risk score and recommendation.
    """
    logger.info(f"Assessing risk for alert {state['alert_id']}")

    # Prepare context summary
    context_summary = _format_context_summary_for_llm(state)

    # Generate prompt
    prompt = RISK_ASSESSMENT_PROMPT.format_messages(
        alert_data=state["alert_data"],
        context_summary=context_summary,
    )

    # Invoke LLM
    try:
        llm_instance = get_llm()
        response = await llm_instance.ainvoke(prompt)
        response_text = response.content

        # Parse structured response
        risk_assessment = _parse_risk_assessment(response_text)

        # Update state
        state["risk_score"] = risk_assessment.risk_score
        state["risk_factors"] = risk_assessment.risk_factors
        state["confidence"] = risk_assessment.confidence
        state["recommendation"] = risk_assessment.reasoning
        state["next_action"] = risk_assessment.suggested_action

        # Determine if human review is needed
        state["requires_human_review"] = state["risk_score"] >= settings.high_risk_threshold
        state["human_review_required"] = state["requires_human_review"]

        logger.info(
            f"Risk assessment complete: score={state['risk_score']}, "
            f"action={state['next_action']}, human_review={state['requires_human_review']}"
        )

    except Exception as e:
        logger.error(f"Error during risk assessment: {e}")
        state["error_message"] = str(e)
        state["next_action"] = "escalate_for_review"  # Conservative fallback
        state["risk_score"] = 50  # Default to medium risk

    state["messages"].append(AIMessage(content=response_text))
    state["iteration_count"] += 1

    return state


async def human_review_node(state: AgentState) -> AgentState:
    """
    Handle human-in-the-loop review for high-risk alerts.

    This node waits for human analyst input before proceeding.
    """
    logger.info(f"Human review required for alert {state['alert_id']}")

    # In a real implementation, this would:
    # 1. Send notification to analysts
    # 2. Wait for human decision via API
    # 3. Process the human decision

    # For now, we'll set the state to await review
    state["messages"].append(
        HumanMessage(
            content=f"Human review required for alert {state['alert_id']} with risk score {state['risk_score']}. "
            f"Waiting for analyst decision..."
        )
    )

    # Check if human decision has been provided
    if state.get("human_decision"):
        logger.info(f"Human decision received: {state['human_decision']}")

        # Update next action based on human decision
        decision = state["human_decision"].lower()
        if "confirm_fraud" in decision or "fraud" in decision:
            state["next_action"] = "create_case"
        elif "legitimate" in decision or "false_positive" in decision:
            state["next_action"] = "auto_close"
        else:
            state["next_action"] = "escalate"
    else:
        # Still waiting for human review
        state["next_action"] = "awaiting_human_review"

    state["iteration_count"] += 1

    return state


def route_alert(state: AgentState) -> Literal["auto_close", "escalate", "human_review", "end"]:
    """
    Route the alert based on risk assessment and decisions.

    This conditional edge determines which path the workflow takes.
    """
    next_action = state.get("next_action", "")

    logger.info(f"Routing alert {state['alert_id']}: action={next_action}")

    # Check if awaiting human review
    if next_action == "awaiting_human_review":
        return "human_review"

    # Check if human review has completed
    if state.get("human_decision"):
        decision = state["human_decision"].lower()
        if "confirm_fraud" in decision or "fraud" in decision:
            return "escalate"
        elif "legitimate" in decision or "false_positive" in decision:
            return "auto_close"

    # Route based on risk score and action
    risk_score = state.get("risk_score", 0)

    if risk_score >= settings.high_risk_threshold or next_action == "create_case":
        return "escalate"
    elif risk_score <= settings.medium_risk_threshold or next_action == "auto_close":
        return "auto_close"
    else:
        return "escalate"


# Helper functions
def _format_context_summary(context: dict[str, Any]) -> str:
    """Format context data into a readable summary."""
    summary_parts = []

    if "transaction_history" in context:
        txn = context["transaction_history"]
        if txn.get("transactions"):
            count = len(txn["transactions"])
            summary_parts.append(f"- {count} historical transactions found")

    if "customer_profile" in context:
        profile = context["customer_profile"]
        if not profile.get("error"):
            name = profile.get("name", "Unknown")
            account_age = profile.get("account_age_years", 0)
            summary_parts.append(f"- Customer: {name}, Account age: {account_age} years")

    if "device_fingerprint" in context:
        device = context["device_fingerprint"]
        if device.get("is_new_device"):
            summary_parts.append(f"- NEW DEVICE detected: {device.get('device_type', 'Unknown')}")

    if "ip_reputation" in context:
        ip_info = context["ip_reputation"]
        if ip_info.get("risk_score", 0) > 50:
            summary_parts.append(f"- High-risk IP: {ip_info.get('country', 'Unknown')} (proxy/VPN)")

    return "\n".join(summary_parts) if summary_parts else "No additional context available."


def _format_context_summary_for_llm(state: AgentState) -> str:
    """Format context for LLM consumption."""
    parts = []

    # Transaction history
    if state.get("transaction_history"):
        txns = state["transaction_history"]
        parts.append(f"Transaction History: {len(txns)} transactions")

    # Customer profile
    if state.get("customer_profile"):
        profile = state["customer_profile"]
        if not profile.get("error"):
            parts.append(
                f"Customer: {profile.get('name', 'Unknown')}, "
                f"Account Age: {profile.get('account_age_years', 0)} years, "
                f"Risk Level: {profile.get('risk_level', 'unknown')}"
            )

    # Device info
    if state.get("device_fingerprint"):
        device = state["device_fingerprint"]
        parts.append(
            f"Device: {device.get('device_type', 'Unknown')}, "
            f"New Device: {device.get('is_new_device', False)}"
        )

    # Similar alerts
    if state.get("similar_alerts"):
        similar = state["similar_alerts"]
        parts.append(f"Previous Alerts: {len(similar)} in past 12 months")

    return "\n".join(parts) if parts else "No context available."


def _parse_risk_assessment(text: str) -> RiskAssessment:
    """Parse LLM response into RiskAssessment model."""
    # Try to extract structured data from text
    import re

    # Extract risk score
    score_match = re.search(r'risk score[:\s]+(\d+)', text, re.IGNORECASE)
    risk_score = int(score_match.group(1)) if score_match else 50

    # Extract confidence
    conf_match = re.search(r'confidence[:\s]+([0-9.]+)', text, re.IGNORECASE)
    confidence = float(conf_match.group(1)) if conf_match else 0.5

    # Extract risk factors (look for lists)
    factors = []
    factor_matches = re.findall(r'[-*]\s*(.+?)(?:\n|$)', text)
    factors.extend([f.strip() for f in factor_matches if f.strip()])

    # Determine action based on risk score
    if risk_score >= settings.high_risk_threshold:
        suggested_action = "create_case"
    elif risk_score <= settings.medium_risk_threshold:
        suggested_action = "auto_close"
    else:
        suggested_action = "escalate_for_review"

    return RiskAssessment(
        risk_score=min(100, max(0, risk_score)),
        risk_factors=factors if factors else ["Analysis based on provided context"],
        reasoning=text,
        confidence=min(1.0, max(0.0, confidence)),
        suggested_action=suggested_action,
    )
