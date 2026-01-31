"""
Visualization utilities for FraudTriage-Agent workflow.

This module provides functions to visualize the LangGraph workflow
in multiple formats: Mermaid diagrams, ASCII art, and PNG images.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Mermaid Diagram Generation
# =============================================================================

def get_mermaid_diagram() -> str:
    """
    Generate a Mermaid diagram of the fraud triage workflow.

    Returns:
        Mermaid diagram string that can be rendered in Markdown/HTML

    Example:
        >>> diagram = get_mermaid_diagram()
        >>> print(diagram)
        ```mermaid
        graph TD
            A[parse_alert] --> B[gather_context]
            ...
        ```
    """
    return """```mermaid
graph TD
    %% =============================================================
    %% FraudTriage-Agent Workflow Diagram
    %% =============================================================

    %% Nodes
    START([Start: Alert Received])
    PARSE[parse_alert]
    CONTEXT[gather_context]
    ANALYZE[analyze_risk]

    %% Decision nodes
    ESCALATE[escalate_alert]
    RECOMMEND[recommend_action]
    CLOSE[auto_close_alert]

    %% End points
    END_ESCALATE([End: Escalated to Fraud Team])
    END_RECOMMEND([End: Action Recommended])
    END_CLOSE([End: Auto-Closed as False Positive])

    %% =============================================================
    %% Main Flow
    %% =============================================================
    START --> PARSE

    PARSE -->|Validate Data| CONTEXT

    CONTEXT -->|Call Tools| ANALYZE

    %% =============================================================
    %% Conditional Routing
    %% =============================================================
    ANALYZE -->|risk_score > 0.8| ESCALATE
    ANALYZE -->|0.4 < risk_score ≤ 0.8| RECOMMEND
    ANALYZE -->|risk_score ≤ 0.4| CLOSE

    %% =============================================================
    %% End Points
    %% =============================================================
    ESCALATE --> END_ESCALATE
    RECOMMEND --> END_RECOMMEND
    CLOSE --> END_CLOSE

    %% =============================================================
    %% Styling
    %% =============================================================
    style START fill:#e1f5e1,stroke:#4caf50,stroke-width:3px
    style PARSE fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    style CONTEXT fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    style ANALYZE fill:#fff3e0,stroke:#ff9800,stroke-width:2px

    style ESCALATE fill:#ffebee,stroke:#f44336,stroke-width:3px
    style RECOMMEND fill:#fff8e1,stroke:#ffc107,stroke-width:2px
    style CLOSE fill:#e8f5e9,stroke:#4caf50,stroke-width:2px

    style END_ESCALATE fill:#ffebee,stroke:#f44336,stroke-width:3px
    style END_RECOMMEND fill:#fff8e1,stroke:#ffc107,stroke-width:2px
    style END_CLOSE fill:#e8f5e9,stroke:#4caf50,stroke-width:2px

    %% =============================================================
    %% Annotations
    %% =============================================================
    subgraph Tools_Called_by_Gather_Context
        T1[get_customer_profile]
        T2[get_transaction_history]
        T3[check_watchlists]
        T4[calculate_risk_score]
        T5[get_similar_alerts]
    end

    CONTEXT -.-> T1
    CONTEXT -.-> T2
    CONTEXT -.-> T3
    CONTEXT -.-> T4
    CONTEXT -.-> T5
```"""


def print_mermaid() -> None:
    """Print the Mermaid diagram to console."""
    print(get_mermaid_diagram())


# =============================================================================
# ASCII Art Generation
# =============================================================================

def get_ascii_diagram() -> str:
    """
    Generate an ASCII art representation of the fraud triage workflow.

    Returns:
        ASCII diagram as a string

    Example:
        >>> diagram = get_ascii_diagram()
        >>> print(diagram)
    """
    return r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    FraudTriage-Agent Workflow                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────────┐
│  1. PARSE_ALERT                                                              │
│  ────────────────                                                            │
│  • Validate alert data                                                        │
│  • Extract: alert_id, alert_type, transaction_amount, customer_id            │
│  • Initialize processing timestamp                                            │
│  • Add system message to workflow                                             │
│                                                                              │
│  Input: Raw alert data                                                       │
│  Output: Validated state                                                     │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  2. GATHER_CONTEXT                                                           │
│  ─────────────────                                                            │
│  Tool Calls (parallel when possible):                                        │
│                                                                              │
│  ✓ get_customer_profile()        → Customer profile, account age, risk       │
│  ✓ get_transaction_history()    → Recent transactions, patterns            │
│  ✓ check_watchlists()             → OFAC, internal fraud DB                  │
│  ✓ calculate_risk_score()         → Initial risk assessment (0-1)           │
│  ✓ get_similar_alerts()           → Historical similar alerts               │
│                                                                              │
│  State Updates:                                                              │
│  • customer_profile, transaction_history                                     │
│  • watchlist_hits, similar_alerts                                            │
│  • risk_score, confidence                                                    │
│  • tools_used list                                                           │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  3. ANALYZE_RISK (LLM)                                                       │
│  ────────────────────────────                                                │
│  • Format gathered context for LLM                                           │
│  • Invoke LLM with comprehensive prompt                                      │
│  • Parse LLM response for decision                                           │
│  • Extract: risk_level, risk_factors, decision, recommendation              │
│                                                                              │
│  LLM Prompt Includes:                                                        │
│  • Alert details (type, amount, customer)                                    │
│  • Customer profile summary                                                  │
│  • Transaction history summary                                               │
│  • Risk assessment score                                                    │
│  • Watchlist hits                                                           │
│  • Similar alerts                                                          │
│  • Account age, verification status                                         │
│                                                                              │
│  State Updates:                                                              │
│  • risk_level, risk_factors                                                  │
│  • decision, recommendation                                                  │
│  • requires_human_review                                                    │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │   route_decision()               │
                    │   ─────────────────              │
                    │   if risk_score > 0.8:           │
                    │       → escalate                 │
                    │   elif risk_score > 0.4:         │
                    │       → recommend                │
                    │   else:                          │
                    │       → auto_close               │
                    └─────────────────┬─────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
   risk > 0.8                  0.4 < risk ≤ 0.8              risk ≤ 0.4
   (CRITICAL)                      (HIGH/MEDIUM)                (LOW)
         │                            │                            │
         ▼                            ▼                            ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│ 4a. ESCALATE_ALERT  │   │ 4b. RECOMMEND_ACTION │   │ 4c. AUTO_CLOSE_ALERT │
│ ────────────────────│   │ ────────────────────│   │ ────────────────────│
│ • Set decision =     │   │ • Set decision =     │   │ • Set decision =     │
│   ESCALATE           │   │   REVIEW_REQUIRED   │   │   AUTO_CLOSE         │
│ • requires_review=   │   │ • requires_review=   │   │ • requires_review=   │
│   True               │   │   True               │   │   False              │
│ • Build escalation   │   │ • Build action       │   │ • Build closure      │
│   message            │   │   recommendation    │   │   message            │
│ • Calculate duration │   │ • Calculate duration │   │ • Calculate duration │
│                      │   │                      │   │                      │
│ 🚨 High Risk         │   │ 📋 Medium Risk       │   │ ✅ Low Risk          │
│ → Block transaction  │   │ → Monitor/Review     │   │ → Auto-close         │
│ → Escalate to team   │   │ → Additional checks  │   │ → Archive            │
└─────────────────────┘   └─────────────────────┘   └─────────────────────┘
         │                            │                            │
         └────────────────────────────┼────────────────────────────┘
                                      │
                                      ▼
                            ╔═════════════════╗
                            ║     END         ║
                            ║  Return State   ║
                            ╚═════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║                          Decision Routing Table                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Risk Score   │ Risk Level   │ Decision        │ Action                      ║
║──────────────║──────────────║─────────────────║─────────────────────────────║
║ 0.76 - 1.00  │ CRITICAL     │ ESCALATE        │ Block, Escalate            ║
║ 0.51 - 0.75  │ HIGH         │ ESCALATE        │ Escalate                   ║
║ 0.26 - 0.50  │ MEDIUM       │ REVIEW_REQUIRED │ Monitor, Review            ║
║ 0.00 - 0.25  │ LOW          │ AUTO_CLOSE      │ Close as False Positive    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def print_ascii() -> None:
    """Print the ASCII diagram to console."""
    print(get_ascii_diagram())


# =============================================================================
# Graphviz Export (if available)
# =============================================================================

def export_to_png(
    output_path: str = "fraud_triage_workflow.png",
    format_type: str = "png",
) -> bool:
    """
    Export the workflow graph to a PNG image using Graphviz.

    Args:
        output_path: Output file path for the image
        format_type: Image format (png, svg, pdf)

    Returns:
        True if export successful, False otherwise

    Example:
        >>> success = export_to_png("workflow.png")
        >>> if success:
        ...     print("Graph exported successfully")
    """
    try:
        import graphviz

        # Create directed graph
        dot = graphviz.Digraph(
            comment='FraudTriage-Agent Workflow',
            format=format_type,
        )

        # Set graph attributes
        dot.attr(
            rankdir='TB',
            fontname='Arial',
            fontsize='12',
            splines='ortho',
        )

        # Add nodes
        dot.node('START', 'Start\\nAlert Received', shape='ellipse', style='filled', fillcolor='lightgreen')
        dot.node('PARSE', 'parse_alert\\nValidate & Extract', shape='box', style='filled', fillcolor='lightblue')
        dot.node('CONTEXT', 'gather_context\\nCall Tools', shape='box', style='filled', fillcolor='lightblue')
        dot.node('ANALYZE', 'analyze_risk\\nLLM Analysis', shape='box', style='filled', fillcolor='lightyellow')
        dot.node('ESCALATE', 'escalate_alert\\nHigh Risk', shape='box', style='filled', fillcolor='lightcoral')
        dot.node('RECOMMEND', 'recommend_action\\nMedium Risk', shape='box', style='filled', fillcolor='lightyellow')
        dot.node('CLOSE', 'auto_close_alert\\nLow Risk', shape='box', style='filled', fillcolor='lightgreen')
        dot.node('END_ESC', 'Escalated', shape='ellipse', style='filled', fillcolor='lightcoral')
        dot.node('END_REC', 'Recommended', shape='ellipse', style='filled', fillcolor='lightyellow')
        dot.node('END_CLOSE', 'Auto-Closed', shape='ellipse', style='filled', fillcolor='lightgreen')

        # Add edges
        dot.edge('START', 'PARSE', 'Validate')
        dot.edge('PARSE', 'CONTEXT', 'Gather')
        dot.edge('CONTEXT', 'ANALYZE', 'Analyze')

        # Conditional edges
        dot.edge('ANALYZE', 'ESCALATE', 'risk > 0.8')
        dot.edge('ANALYZE', 'RECOMMEND', '0.4 < risk ≤ 0.8')
        dot.edge('ANALYZE', 'CLOSE', 'risk ≤ 0.4')

        # End edges
        dot.edge('ESCALATE', 'END_ESC')
        dot.edge('RECOMMEND', 'END_REC')
        dot.edge('CLOSE', 'END_CLOSE')

        # Render and save
        dot.render(output_path.replace(f'.{format_type}', ''), cleanup=True, format=format_type)

        logger.info(f"Graph exported to: {output_path}")
        return True

    except ImportError:
        logger.warning("graphviz not installed. Install with: pip install graphviz")
        return False
    except Exception as e:
        logger.error(f"Error exporting graph: {e}")
        return False


# =============================================================================
# State Visualization
# =============================================================================

def visualize_state(state: dict[str, Any]) -> str:
    """
    Create a visual representation of the current workflow state.

    Args:
        state: Current fraud triage state dictionary

    Returns:
        Formatted string representation of the state

    Example:
        >>> from src.agents.fraud_triage_agent import FraudTriageAgent
        >>> agent = FraudTriageAgent()
        >>> result = await agent.arun(...)
        >>> print(visualize_state(result))
    """
    lines = [
        "╔══════════════════════════════════════════════════════════════╗",
        "║              FraudTriage-Agent State Visualization            ║",
        "╚══════════════════════════════════════════════════════════════╝",
        "",
    ]

    # Alert Information
    lines.append("┌─ ALERT INFORMATION ─────────────────────────────────────")
    lines.append(f"│ Alert ID:        {state.get('alert_id', 'N/A')}")
    lines.append(f"│ Alert Type:      {state.get('alert_type', 'N/A')}")
    lines.append(f"│ Customer ID:     {state.get('customer_id', 'N/A')}")
    lines.append(f"│ Transaction:     ${state.get('transaction_amount', 0):.2f}")
    lines.append("└─────────────────────────────────────────────────────────")
    lines.append("")

    # Processing Status
    status = state.get('processing_started')
    completed = state.get('processing_completed')
    duration = state.get('processing_duration_ms')

    lines.append("┌─ PROCESSING STATUS ──────────────────────────────────────")
    lines.append(f"│ Status:          {state.get('status', 'Unknown')}")
    lines.append(f"│ Iterations:      {state.get('iteration_count', 0)}")
    lines.append(f"│ Started:         {status.strftime('%Y-%m-%d %H:%M:%S') if status else 'N/A'}")
    lines.append(f"│ Completed:       {completed.strftime('%Y-%m-%d %H:%M:%S') if completed else 'In Progress'}")
    lines.append(f"│ Duration:        {duration}ms" if duration else "│ Duration:        N/A")
    lines.append("└─────────────────────────────────────────────────────────")
    lines.append("")

    # Risk Assessment
    risk_score = state.get('risk_score')
    risk_level = state.get('risk_level')
    decision = state.get('decision')

    lines.append("┌─ RISK ASSESSMENT ──────────────────────────────────────────")

    # Risk score visualization
    if risk_score is not None:
        score_bar = "█" * int(risk_score * 40)
        lines.append(f"│ Risk Score:      {risk_score:.3f} / 1.0")
        lines.append(f"│                  [{score_bar:<40}]")
        lines.append(f"│ Risk Level:      {risk_level.value.upper() if risk_level else 'N/A'}")
    else:
        lines.append(f"│ Risk Score:      PENDING")
        lines.append(f"│ Risk Level:      PENDING")

    lines.append(f"│ Decision:        {decision.value.upper() if decision else 'PENDING'}")
    lines.append(f"│ Human Review:    {'YES ⚠️' if state.get('requires_human_review') else 'NO ✓'}")
    lines.append("└─────────────────────────────────────────────────────────")
    lines.append("")

    # Risk Factors
    factors = state.get('risk_factors', [])
    if factors:
        lines.append("┌─ RISK FACTORS ───────────────────────────────────────────")
        for i, factor in enumerate(factors[:10], 1):
            lines.append(f"│ {i}. {factor}")
        if len(factors) > 10:
            lines.append(f"│ ... and {len(factors) - 10} more")
        lines.append("└─────────────────────────────────────────────────────────")
        lines.append("")

    # Tools Used
    tools = state.get('tools_used', [])
    if tools:
        lines.append("┌─ TOOLS INVOKED ───────────────────────────────────────────")
        for tool in tools:
            lines.append(f"│ ✓ {tool}")
        lines.append("└─────────────────────────────────────────────────────────")
        lines.append("")

    # Recommendation
    recommendation = state.get('recommendation')
    if recommendation:
        lines.append("┌─ RECOMMENDATION ─────────────────────────────────────────")
        # Wrap long recommendations
        words = recommendation.split()
        line = "│ "
        for word in words:
            if len(line) + len(word) + 1 > 65:
                lines.append(line)
                line = "│ " + word + " "
            else:
                line += word + " "
        if line.strip():
            lines.append(line)
        lines.append("└─────────────────────────────────────────────────────────")
        lines.append("")

    # Error (if any)
    error = state.get('error_message')
    if error:
        lines.append("┌─ ERROR ───────────────────────────────────────────────────")
        lines.append(f"│ {error}")
        lines.append("└─────────────────────────────────────────────────────────")
        lines.append("")

    lines.append("╔══════════════════════════════════════════════════════════════╗")

    return "\n".join(lines)


def print_state(state: dict[str, Any]) -> None:
    """Print the visual state representation to console."""
    print(visualize_state(state))


# =============================================================================
# Summary Statistics
# =============================================================================

def get_workflow_summary() -> str:
    """
    Generate a summary of the workflow configuration and statistics.

    Returns:
        Formatted summary string
    """
    from src.agents.fraud_triage_agent import AgentConfig

    return f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    FraudTriage-Agent Configuration                         ║
╚═══════════════════════════════════════════════════════════════════════════╝

┌───────────────────────────────────────────────────────────────────────────┐
│  Workflow Configuration                                                      │
├───────────────────────────────────────────────────────────────────────────┤
│  Total Nodes:              6                                                │
│  Decision Nodes:          3 (escalate, recommend, auto_close)              │
│  Tool Functions:          5 (customer, transactions, watchlists, etc.)    │
│  Conditional Routes:      1 (based on risk_score)                          │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│  Risk Thresholds                                                              │
├───────────────────────────────────────────────────────────────────────────┤
│  ESCALATE_THRESHOLD:     {AgentConfig.ESCALATE_THRESHOLD} (> 0.8)             │
│  RECOMMEND_THRESHOLD:    {AgentConfig.RECOMMEND_THRESHOLD} (> 0.4)            │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│  Decision Matrix                                                              │
├───────────────────────────────────────────────────────────────────────────┤
│  Risk Score Range    │ Decision          │ Human Review │ Action            │
│  ────────────────────┼───────────────────┼──────────────┼──────────────────│
│  0.76 - 1.00        │ ESCALATE          │ YES          │ Block/Escalate   │
│  0.51 - 0.75        │ ESCALATE          │ YES          │ Escalate         │
│  0.26 - 0.50        │ REVIEW_REQUIRED   │ YES          │ Monitor/Review    │
│  0.00 - 0.25        │ AUTO_CLOSE        │ NO           │ Close/Archive    │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│  Nodes & Functions                                                           │
├───────────────────────────────────────────────────────────────────────────┤
│  1. parse_alert          → Validate and extract alert data                  │
│  2. gather_context       → Call 5 fraud tools to gather context              │
│  3. analyze_risk         → LLM-powered risk assessment and decision          │
│  4. escalate_alert       → Handle high-risk alerts (score > 0.8)            │
│  5. recommend_action     → Handle medium-risk alerts (0.4 < score ≤ 0.8)    │
│  6. auto_close_alert     → Handle low-risk alerts (score ≤ 0.4)             │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│  Tool Functions                                                               │
├───────────────────────────────────────────────────────────────────────────┤
│  • get_customer_profile     → Customer account and risk profile              │
│  • get_transaction_history  → Historical transaction data                    │
│  • check_watchlists         → OFAC, sanctions, internal fraud DB             │
│  • calculate_risk_score     → Rule-based risk calculation (0-1)            │
│  • get_similar_alerts       → Historical similar alerts with outcomes        │
└───────────────────────────────────────────────────────────────────────────┘
"""


def print_summary() -> None:
    """Print the workflow summary to console."""
    print(get_workflow_summary())


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """
    Main entry point for visualization utilities.

    Usage:
        python -m src.utils.visualize
    """
    print("\n" + "=" * 80)
    print("  FraudTriage-Agent Visualization")
    print("=" * 80 + "\n")

    # Print workflow summary
    print_summary()

    print("\n" + "=" * 80)
    print("  ASCII Workflow Diagram")
    print("=" * 80 + "\n")
    print_ascii()

    print("\n" + "=" * 80)
    print("  Mermaid Diagram (for Markdown/HTML)")
    print("=" * 80 + "\n")
    print_mermaid()

    # Try to export PNG
    print("\n" + "=" * 80)
    print("  Exporting to PNG...")
    print("=" * 80 + "\n")
    success = export_to_png()
    if success:
        print("✅ PNG exported successfully: fraud_triage_workflow.png")
    else:
        print("❌ PNG export failed. Install graphviz: pip install graphviz")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
