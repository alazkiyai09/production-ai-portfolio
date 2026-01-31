"""
LangGraph nodes for fraud alert triage workflow.

Each node is a function that takes the current state and returns
updated state fields. Nodes should be pure functions that only
update the fields they are responsible for.
"""

import logging
from datetime import datetime
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config.settings import settings
from src.models.state import (
    AlertDecision,
    AlertType,
    FraudTriageState,
    RiskLevel,
    WorkflowStage,
    StateTransitionError,
    create_initial_state,
    transition_to_stage,
    validate_state_transition,
)

logger = logging.getLogger(__name__)

# =============================================================================
# LLM Setup
# =============================================================================

def create_llm() -> ChatOpenAI:
    """
    Create LLM instance based on configuration.

    Returns:
        Configured ChatOpenAI instance (supports GLM-4.7 via OpenAI-compatible API)
    """
    if settings.glm_api_key:
        logger.info(f"Using GLM model: {settings.glm_model}")
        return ChatOpenAI(
            base_url=settings.glm_base_url,
            api_key=settings.glm_api_key,
            model=settings.glm_model,
            temperature=settings.glm_temperature,
            max_tokens=settings.glm_max_tokens,
        )
    elif settings.openai_api_key:
        logger.info(f"Using OpenAI model: {settings.openai_model}")
        return ChatOpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens,
        )
    else:
        raise ValueError(
            "No LLM API key configured. Please set GLM_API_KEY or OPENAI_API_KEY in .env"
        )


# Global LLM instance
llm = create_llm()

# =============================================================================
# Prompts
# =============================================================================

RISK_ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert fraud analyst with 15 years of experience in banking fraud detection.

Your task is to analyze fraud alerts and provide comprehensive risk assessments.

**Risk Assessment Framework:**

1. **Risk Score (0-100):**
   - 0-25: LOW - Clear indicators of legitimate activity
   - 26-50: MEDIUM-LOW - Some concerns but likely legitimate
   - 51-75: MEDIUM-HIGH - Significant risk indicators
   - 76-100: HIGH - Strong fraud probability

2. **Risk Factors:** List specific indicators (unusual location, new device, velocity, etc.)

3. **Confidence (0.0-1.0):** How certain are you in this assessment?

4. **Decision:**
   - auto_close: Clear false positive, no action needed
   - review_required: Needs human analyst review
   - escalate: Escalate to fraud investigation team
   - block_transaction: Block the transaction immediately

**Analysis Considerations:**
- Customer history and behavior patterns
- Transaction deviation from normal patterns
- Geographic and device anomalies
- Velocity and timing patterns
- Known fraud indicators

Provide your assessment in a clear, structured format."""),
    ("human", """**FRAUD ALERT ANALYSIS REQUEST**

Alert ID: {alert_id}
Alert Type: {alert_type}
Transaction Amount: ${transaction_amount}
Customer ID: {customer_id}

**Alert Context:**
{alert_context}

**Gathered Information:**
{gathered_info}

Please provide a comprehensive risk assessment with:
1. Risk Score (0-100)
2. Risk Level (LOW/MEDIUM/HIGH/CRITICAL)
3. Risk Factors (list specific indicators)
4. Confidence (0.0-1.0)
5. Decision (auto_close/review_required/escalate/block_transaction)
6. Detailed reasoning""")
])


HUMAN_REVIEW_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are processing a human fraud analyst's review decision.

The analyst has provided their decision and reasoning. Update the assessment
accordingly and provide a final recommendation."""),
    ("human", """**HUMAN REVIEW RECEIVED**

Alert ID: {alert_id}
Original Risk Score: {original_risk_score}
Original Decision: {original_decision}

**Human Reviewer:** {reviewer_name} ({reviewer_id})
**Review Decision:** {review_decision}
**Review Reasoning:**
{review_reasoning}

**Agreement with Agent:** {agreed_with_agent}
**Suggested Risk Score:** {suggested_risk_score}

Please update the final assessment based on the human review."""),
])


# =============================================================================
# Node Functions
# =============================================================================

async def parse_alert_node(state: FraudTriageState) -> FraudTriageState:
    """
    Parse and validate the incoming fraud alert.

    This is the entry point node. It extracts and validates information
    from the raw alert data.

    Args:
        state: Current workflow state

    Returns:
        Updated state with parsed information
    """
    logger.info(f"[{state['alert_id']}] Parsing alert")

    # Validate state transition
    try:
        state = transition_to_stage(state, WorkflowStage.PARSING)
    except StateTransitionError as e:
        logger.error(f"[{state['alert_id']}] State transition error: {e}")
        state["error_message"] = str(e)
        return state

    # Increment iteration count
    state["iteration_count"] = state.get("iteration_count", 0) + 1

    # Add system message
    state["messages"].append(
        SystemMessage(content=f"Alert {state['alert_id']} received for processing. Type: {state['alert_type'].value}")
    )

    # Validate required fields
    if state["transaction_amount"] <= 0:
        state["error_message"] = "Invalid transaction amount: must be positive"
        logger.error(f"[{state['alert_id']}] Invalid transaction amount")
        # Transition to error state
        state = transition_to_stage(state, WorkflowStage.ERROR)
        return state

    # Log alert details
    logger.info(
        f"[{state['alert_id']}] Alert type: {state['alert_type'].value}, "
        f"Amount: ${state['transaction_amount']:.2f}, "
        f"Customer: {state['customer_id']}"
    )

    return state


async def gather_context_node(state: FraudTriageState) -> FraudTriageState:
    """
    Gather context from multiple data sources.

    This node invokes tools to collect:
    - Customer profile and history
    - Transaction patterns
    - Device fingerprint
    - IP reputation
    - Similar historical alerts

    Args:
        state: Current workflow state

    Returns:
        Updated state with gathered context
    """
    logger.info(f"[{state['alert_id']}] Gathering context")

    # Validate state transition
    try:
        state = transition_to_stage(state, WorkflowStage.GATHERING_CONTEXT)
    except StateTransitionError as e:
        logger.error(f"[{state['alert_id']}] State transition error: {e}")
        state["error_message"] = str(e)
        return state

    state["iteration_count"] = state.get("iteration_count", 0) + 1
    state["tools_used"] = state.get("tools_used", [])

    # Import tools (avoid circular import)
    from src.tools import (
        analyze_transaction_patterns,
        check_ip_reputation,
        get_customer_profile,
        get_customer_risk_history,
        get_device_fingerprint,
        get_transaction_history,
    )

    # Gather context in parallel where possible
    context_gathered = []

    # 1. Customer profile
    try:
        if state.get("customer_id"):
            profile = await get_customer_profile(state["customer_id"])
            state["customer_profile"] = profile
            context_gathered.append("customer_profile")
            logger.debug(f"[{state['alert_id']}] Customer profile retrieved")
    except Exception as e:
        logger.warning(f"[{state['alert_id']}] Error fetching customer profile: {e}")
        state["customer_profile"] = {"error": str(e)}

    # 2. Transaction history
    try:
        if state.get("account_id"):
            history = await get_transaction_history(
                account_id=state["account_id"],
                days=90,
                limit=100,
            )
            state["transaction_history"] = history.get("transactions", [])
            context_gathered.append("transaction_history")
            logger.debug(f"[{state['alert_id']}] Retrieved {len(state['transaction_history'])} transactions")
    except Exception as e:
        logger.warning(f"[{state['alert_id']}] Error fetching transaction history: {e}")
        state["transaction_history"] = []

    # 3. Transaction patterns
    try:
        if state.get("account_id") and state.get("transaction_id"):
            patterns = await analyze_transaction_patterns(
                account_id=state["account_id"],
                transaction_id=state["transaction_id"],
            )
            state["transaction_patterns"] = patterns
            context_gathered.append("transaction_patterns")
            logger.debug(f"[{state['alert_id']}] Transaction patterns analyzed")
    except Exception as e:
        logger.warning(f"[{state['alert_id']}] Error analyzing patterns: {e}")
        state["transaction_patterns"] = {"error": str(e)}

    # 4. Customer risk history
    try:
        if state.get("customer_id"):
            risk_history = await get_customer_risk_history(
                customer_id=state["customer_id"],
                months=12,
            )
            state["customer_risk_history"] = risk_history
            context_gathered.append("customer_risk_history")
            logger.debug(f"[{state['alert_id']}] Customer risk history retrieved")
    except Exception as e:
        logger.warning(f"[{state['alert_id']}] Error fetching risk history: {e}")
        state["customer_risk_history"] = {"error": str(e)}

    # 5. Device fingerprint
    try:
        if state.get("transaction_device_id"):
            device_info = await get_device_fingerprint(state["transaction_device_id"])
            state["device_info"] = device_info
            context_gathered.append("device_info")
            logger.debug(f"[{state['alert_id']}] Device fingerprint retrieved")
    except Exception as e:
        logger.warning(f"[{state['alert_id']}] Error fetching device info: {e}")
        state["device_info"] = {"error": str(e)}

    # 6. IP reputation
    try:
        if state.get("transaction_ip"):
            ip_info = await check_ip_reputation(state["transaction_ip"])
            state["ip_reputation"] = ip_info
            context_gathered.append("ip_reputation")
            logger.debug(f"[{state['alert_id']}] IP reputation checked")
    except Exception as e:
        logger.warning(f"[{state['alert_id']}] Error checking IP reputation: {e}")
        state["ip_reputation"] = {"error": str(e)}

    # Update tools used
    state["tools_used"] = list(set(state.get("tools_used", []) + context_gathered))

    # Add message about context gathering
    state["messages"].append(
        AIMessage(content=f"Gathered context from {len(context_gathered)} sources: {', '.join(context_gathered)}")
    )

    logger.info(f"[{state['alert_id']}] Context gathering complete: {len(context_gathered)} sources")

    return state


async def assess_risk_node(state: FraudTriageState) -> FraudTriageState:
    """
    Assess risk using LLM analysis.

    This node uses the LLM to analyze the alert and gathered context,
    producing a risk score, risk level, risk factors, and decision.

    Args:
        state: Current workflow state

    Returns:
        Updated state with risk assessment
    """
    logger.info(f"[{state['alert_id']}] Assessing risk with LLM")

    # Validate state transition
    try:
        state = transition_to_stage(state, WorkflowStage.ASSESSING_RISK)
    except StateTransitionError as e:
        logger.error(f"[{state['alert_id']}] State transition error: {e}")
        state["error_message"] = str(e)
        return state

    state["iteration_count"] = state.get("iteration_count", 0) + 1

    # Format alert context
    alert_context = _format_alert_context(state)

    # Format gathered information
    gathered_info = _format_gathered_info(state)

    # Generate prompt
    messages = RISK_ASSESSMENT_PROMPT.format_messages(
        alert_id=state["alert_id"],
        alert_type=state["alert_type"].value,
        transaction_amount=f"{state['transaction_amount']:.2f}",
        customer_id=state["customer_id"],
        alert_context=alert_context,
        gathered_info=gathered_info,
    )

    # Invoke LLM
    try:
        response = await llm.ainvoke(messages)
        response_text = response.content

        logger.debug(f"[{state['alert_id']}] LLM response: {response_text[:200]}...")

        # Parse structured response
        assessment = _parse_risk_assessment(response_text)

        # Update state with assessment
        state["risk_score"] = assessment["risk_score"]
        state["risk_level"] = assessment["risk_level"]
        state["risk_factors"] = assessment["risk_factors"]
        state["confidence"] = assessment["confidence"]
        state["decision"] = assessment["decision"]
        state["recommendation"] = assessment["recommendation"]
        state["requires_human_review"] = assessment["requires_human_review"]
        state["model_used"] = settings.glm_model if settings.glm_api_key else settings.openai_model

        # Add AI message
        state["messages"].append(AIMessage(content=response_text))

        logger.info(
            f"[{state['alert_id']}] Risk assessment complete: "
            f"Score={state['risk_score']:.1f}, "
            f"Level={state['risk_level'].value}, "
            f"Decision={state['decision'].value}, "
            f"Human Review={state['requires_human_review']}"
        )

    except Exception as e:
        logger.error(f"[{state['alert_id']}] Error during risk assessment: {e}")
        state["error_message"] = str(e)
        # Conservative fallback
        state["risk_score"] = 50.0
        state["risk_level"] = RiskLevel.MEDIUM
        state["decision"] = AlertDecision.REVIEW_REQUIRED
        state["requires_human_review"] = True
        state["confidence"] = 0.0

    return state


async def human_review_node(state: FraudTriageState) -> FraudTriageState:
    """
    Handle human-in-the-loop review for alerts requiring human decision.

    This node checks if human review has been provided and processes it.
    If no review is available yet, the workflow waits.

    Args:
        state: Current workflow state

    Returns:
        Updated state with human review processed
    """
    logger.info(f"[{state['alert_id']}] Processing human review")

    # Validate state transition (allow staying in same stage for waiting)
    try:
        current_stage = state.get("workflow_stage", WorkflowStage.INITIALIZED)
        if current_stage != WorkflowStage.HUMAN_REVIEW:
            state = transition_to_stage(state, WorkflowStage.HUMAN_REVIEW)
    except StateTransitionError as e:
        logger.error(f"[{state['alert_id']}] State transition error: {e}")
        state["error_message"] = str(e)
        return state

    state["iteration_count"] = state.get("iteration_count", 0) + 1

    # Check if human review has been submitted
    if not state.get("human_review_decision"):
        # Still waiting for human review
        logger.info(f"[{state['alert_id']}] Awaiting human review")
        state["messages"].append(
            HumanMessage(
                content=f"Alert {state['alert_id']} requires human review. "
                f"Risk Score: {state['risk_score']}, Risk Level: {state['risk_level'].value}. "
                f"Waiting for analyst decision..."
            )
        )
        return state

    # Process human review decision
    human_decision = state["human_review_decision"]
    reviewer_name = state.get("human_reviewer_name", "Unknown Analyst")
    reviewer_id = state.get("human_reviewer_id", "unknown")

    logger.info(
        f"[{state['alert_id']}] Human review received from {reviewer_name}: {human_decision.value}"
    )

    # Update decision based on human review
    state["decision"] = human_decision
    state["human_review_timestamp"] = datetime.utcnow()

    # Update recommendation based on human review
    if human_decision == AlertDecision.AUTO_CLOSE:
        state["requires_human_review"] = False
        state["recommendation"] = f"Human reviewer ({reviewer_name}) confirmed as legitimate. {state.get('human_review_reasoning', '')}"
    elif human_decision == AlertDecision.BLOCK_TRANSACTION:
        state["requires_human_review"] = False
        state["recommendation"] = f"Human reviewer ({reviewer_name}) confirmed fraud. Transaction should be blocked. {state.get('human_review_reasoning', '')}"
    elif human_decision == AlertDecision.ESCALATE:
        state["requires_human_review"] = False
        state["recommendation"] = f"Human reviewer ({reviewer_name}) escalated for investigation. {state.get('human_review_reasoning', '')}"
    else:  # REVIEW_REQUIRED (requesting additional review)
        state["requires_human_review"] = True
        state["recommendation"] = f"Human reviewer ({reviewer_name}) requests additional review. {state.get('human_review_reasoning', '')}"

    # Add message about human review
    state["messages"].append(
        HumanMessage(
            content=f"Human review completed by {reviewer_name} ({reviewer_id}). "
            f"Decision: {human_decision.value}. "
            f"Reasoning: {state.get('human_review_reasoning', 'N/A')}"
        )
    )

    logger.info(f"[{state['alert_id']}] Human review processed: {human_decision.value}")

    return state


async def finalize_node(state: FraudTriageState) -> FraudTriageState:
    """
    Finalize the workflow and compute final state.

    This node marks the workflow as complete and calculates final metrics.

    Args:
        state: Current workflow state

    Returns:
        Finalized state
    """
    logger.info(f"[{state['alert_id']}] Finalizing workflow")

    # Validate state transition
    try:
        state = transition_to_stage(state, WorkflowStage.FINALIZING)
    except StateTransitionError as e:
        logger.error(f"[{state['alert_id']}] State transition error: {e}")
        state["error_message"] = str(e)
        return state

    state["iteration_count"] = state.get("iteration_count", 0) + 1
    state["processing_completed"] = datetime.utcnow()

    # Mark as completed
    state = transition_to_stage(state, WorkflowStage.COMPLETED)

    # Calculate duration
    if state["processing_started"]:
        duration = (
            state["processing_completed"] - state["processing_started"]
        ).total_seconds()
        state["processing_duration_ms"] = int(duration * 1000)

    # Log final summary
    logger.info(
        f"[{state['alert_id']}] Workflow complete: "
        f"Decision={state['decision'].value}, "
        f"Risk Score={state['risk_score']:.1f}, "
        f"Duration={state['processing_duration_ms']}ms, "
        f"Iterations={state['iteration_count']}"
    )

    return state


# =============================================================================
# Routing Functions
# =============================================================================

def route_after_assessment(state: FraudTriageState) -> str:
    """
    Route alert after risk assessment.

    Args:
        state: Current workflow state

    Returns:
        Next node name: "human_review", "finalize", or "finalize"
    """
    risk_score = state.get("risk_score", 0)
    requires_review = state.get("requires_human_review", False)

    logger.debug(f"[{state['alert_id']}] Routing after assessment: score={risk_score}, review={requires_review}")

    # Route to human review if required
    if requires_review:
        return "human_review"

    # Otherwise finalize directly
    return "finalize"


def route_after_human_review(state: FraudTriageState) -> str:
    """
    Route alert after human review.

    Args:
        state: Current workflow state

    Returns:
        Next node name: "finalize" or "human_review" (if still waiting)
    """
    has_decision = state.get("human_review_decision") is not None

    if not has_decision:
        # Still waiting for human review
        logger.debug(f"[{state['alert_id']}] Still awaiting human review")
        return "human_review"

    # Human review complete, finalize
    logger.debug(f"[{state['alert_id']}] Human review complete, finalizing")
    return "finalize"


# =============================================================================
# Helper Functions
# =============================================================================

def _format_alert_context(state: FraudTriageState) -> str:
    """Format alert context for LLM prompt."""
    lines = [
        f"**Alert Type:** {state['alert_type'].value}",
        f"**Transaction Amount:** ${state['transaction_amount']:.2f}",
        f"**Customer ID:** {state['customer_id']}",
    ]

    if state.get("account_id"):
        lines.append(f"**Account ID:** {state['account_id']}")
    if state.get("transaction_country"):
        lines.append(f"**Transaction Country:** {state['transaction_country']}")
    if state.get("transaction_city"):
        lines.append(f"**Transaction City:** {state['transaction_city']}")
    if state.get("merchant_name"):
        lines.append(f"**Merchant:** {state['merchant_name']}")
    if state.get("alert_reason"):
        lines.append(f"**Alert Reason:** {state['alert_reason']}")

    return "\n".join(lines)


def _format_gathered_info(state: FraudTriageState) -> str:
    """Format gathered information for LLM prompt."""
    sections = []

    # Customer profile
    if state.get("customer_profile"):
        profile = state["customer_profile"]
        if isinstance(profile, dict) and not profile.get("error"):
            sections.append(
                f"**Customer Profile:**\n"
                f"- Name: {profile.get('name', 'Unknown')}\n"
                f"- Account Age: {profile.get('account_age_years', profile.get('account_age_days', 0) // 365)} years\n"
                f"- Segment: {profile.get('customer_segment', 'Unknown')}\n"
                f"- Risk Level: {profile.get('risk_level', 'Unknown')}\n"
                f"- KYC Verified: {profile.get('kyc_verified', False)}\n"
                f"- Previous Fraud Cases: {profile.get('previous_fraud_cases', 0)}"
            )

    # Transaction patterns
    if state.get("transaction_patterns"):
        patterns = state["transaction_patterns"]
        if isinstance(patterns, dict) and not patterns.get("error"):
            sections.append(
                f"**Transaction Patterns:**\n"
                f"- Transaction Count: {patterns.get('transaction_count', 'N/A')}\n"
                f"- Average Amount: ${patterns.get('average_amount', 0):.2f}\n"
                f"- Max Amount: ${patterns.get('max_amount', 0):.2f}\n"
                f"- Anomalies: {len(patterns.get('anomalies', []))} detected"
            )
            if patterns.get("anomalies"):
                for anomaly in patterns["anomalies"][:3]:  # Limit to 3
                    sections.append(f"  - {anomaly.get('description', anomaly)}")

    # Device info
    if state.get("device_info"):
        device = state["device_info"]
        if isinstance(device, dict) and not device.get("error"):
            new_device = device.get("is_new_device", False)
            sections.append(
                f"**Device Information:**\n"
                f"- Device Type: {device.get('device_type', 'Unknown')}\n"
                f"- New Device: {'YES ⚠️' if new_device else 'No (known device)'}\n"
                f"- Transaction Count: {device.get('transaction_count', 0)}\n"
                f"- Device Risk Score: {device.get('risk_score', 0)}/100"
            )

    # IP reputation
    if state.get("ip_reputation"):
        ip = state["ip_reputation"]
        if isinstance(ip, dict) and not ip.get("error"):
            sections.append(
                f"**IP Reputation:**\n"
                f"- Country: {ip.get('country', 'Unknown')}\n"
                f"- VPN/Proxy: {'Yes ⚠️' if ip.get('is_vpn') or ip.get('is_proxy') else 'No'}\n"
                f"- IP Risk Score: {ip.get('risk_score', 0)}/100"
            )

    # Customer risk history
    if state.get("customer_risk_history"):
        history = state["customer_risk_history"]
        if isinstance(history, dict) and not history.get("error"):
            sections.append(
                f"**Customer Risk History:**\n"
                f"- Total Alerts (12mo): {history.get('total_alerts', 0)}\n"
                f"- Confirmed Fraud: {history.get('confirmed_fraud_cases', 0)}\n"
                f"- False Positives: {history.get('false_positives', 0)}"
            )

    return "\n\n".join(sections) if sections else "No additional information available."


def _parse_risk_assessment(text: str) -> dict[str, Any]:
    """
    Parse LLM response into structured assessment.

    Args:
        text: LLM response text

    Returns:
        Dictionary with parsed assessment fields
    """
    import re

    # Default values
    assessment = {
        "risk_score": 50.0,
        "risk_level": RiskLevel.MEDIUM,
        "risk_factors": [],
        "confidence": 0.5,
        "decision": AlertDecision.REVIEW_REQUIRED,
        "recommendation": text,
        "requires_human_review": True,
    }

    # Extract risk score
    score_match = re.search(r'risk score[:\s]+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if score_match:
        assessment["risk_score"] = float(score_match.group(1))

    # Determine risk level from score
    score = assessment["risk_score"]
    if score >= 80:
        assessment["risk_level"] = RiskLevel.CRITICAL
    elif score >= 60:
        assessment["risk_level"] = RiskLevel.HIGH
    elif score >= 40:
        assessment["risk_level"] = RiskLevel.MEDIUM
    else:
        assessment["risk_level"] = RiskLevel.LOW

    # Extract risk factors
    factor_patterns = [
        r'risk factors?:?\s*[:\n](.*?)(?=\n\n|\n[A-Z]|$)',
        r'indicators?:?\s*[:\n](.*?)(?=\n\n|\n[A-Z]|$)',
    ]
    for pattern in factor_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            factors_text = match.group(1)
            # Extract bullet points or numbered items
            factors = re.findall(r'[-*•]\s*(.*?)(?:\n|$)|\d+\.\s*(.*?)(?:\n|$)', factors_text)
            assessment["risk_factors"] = [f[0] or f[1] for f in factors if (f[0] or f[1])]
            if assessment["risk_factors"]:
                break

    # If no factors found via patterns, try simple line extraction
    if not assessment["risk_factors"]:
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(('-', '*', '•')) or (len(line) > 10 and line[0].isdigit() and line[1] == '.'):
                assessment["risk_factors"].append(line.lstrip('-*•0123456789.'))

    # Extract confidence
    conf_match = re.search(r'confidence[:\s]+([0-9.]+)', text, re.IGNORECASE)
    if conf_match:
        assessment["confidence"] = float(conf_match.group(1))

    # Determine decision based on risk score and keywords
    decision_keywords = {
        AlertDecision.BLOCK_TRANSACTION: ['block', 'immediate', 'critical', 'confirmed fraud'],
        AlertDecision.ESCALATE: ['escalate', 'investigation', 'fraud team', 'high risk'],
        AlertDecision.AUTO_CLOSE: ['close', 'legitimate', 'false positive', 'low risk'],
        AlertDecision.REVIEW_REQUIRED: ['review', 'uncertain', 'analyst', 'manual'],
    }

    text_lower = text.lower()
    for decision, keywords in decision_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            assessment["decision"] = decision
            break

    # Determine if human review is required
    assessment["requires_human_review"] = (
        assessment["risk_score"] >= settings.high_risk_threshold or
        assessment["decision"] in [AlertDecision.REVIEW_REQUIRED, AlertDecision.ESCALATE]
    )

    # Build recommendation
    if assessment["risk_factors"]:
        recommendation = f"Risk Assessment: {assessment['risk_level'].value.upper()} ({assessment['risk_score']:.0f}/100)\n\n"
        recommendation += "Key Risk Factors:\n" + "\n".join(f"• {f}" for f in assessment["risk_factors"][:5])
        assessment["recommendation"] = recommendation
    else:
        assessment["recommendation"] = text[:500]  # Use first 500 chars of response

    return assessment
