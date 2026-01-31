"""
LangGraph workflow definition for fraud triage.

This module defines the complete workflow graph including nodes,
edges, and conditional routing.
"""

from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .nodes import (
    assess_risk_node,
    gather_context_node,
    human_review_node,
    parse_alert_node,
    route_alert,
)
from .state import AgentState


def create_fraud_triage_graph() -> "FraudTriageGraph":
    """
    Create the fraud triage LangGraph workflow.

    Returns:
        Compiled LangGraph workflow
    """
    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("parse_alert", parse_alert_node)
    workflow.add_node("gather_context", gather_context_node)
    workflow.add_node("assess_risk", assess_risk_node)
    workflow.add_node("human_review", human_review_node)

    # Add edges
    workflow.set_entry_point("parse_alert")

    # Linear flow: parse -> gather -> assess -> route
    workflow.add_edge("parse_alert", "gather_context")
    workflow.add_edge("gather_context", "assess_risk")

    # Conditional routing based on risk assessment
    workflow.add_conditional_edges(
        "assess_risk",
        route_alert,
        {
            "auto_close": END,
            "escalate": END,
            "human_review": "human_review",
        },
    )

    # Human review can lead to different outcomes
    workflow.add_conditional_edges(
        "human_review",
        route_alert,
        {
            "auto_close": END,
            "escalate": END,
            "human_review": "human_review",  # Still waiting
        },
    )

    # Compile with checkpoint saver for persistence
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return FraudTriageGraph(app)


class FraudTriageGraph:
    """
    Wrapper class for the fraud triage LangGraph workflow.

    Provides a clean interface for running the workflow.
    """

    def __init__(self, graph):
        """Initialize with compiled LangGraph."""
        self.graph = graph

    async def arun(
        self,
        alert_data: dict,
        config: dict | None = None,
    ) -> dict:
        """
        Run the fraud triage workflow asynchronously.

        Args:
            alert_data: Fraud alert data
            config: Optional configuration for the run

        Returns:
            Final state after workflow completion
        """
        from .state import create_initial_state

        # Create initial state
        initial_state = create_initial_state(alert_data)

        # Configure run
        run_config = {
            "configurable": {"thread_id": alert_data.get("alert_id", "unknown")},
        }
        if config:
            run_config.update(config)

        # Run workflow
        result = await self.graph.ainvoke(initial_state, config=run_config)

        return result

    def run(
        self,
        alert_data: dict,
        config: dict | None = None,
    ) -> dict:
        """
        Run the fraud triage workflow synchronously.

        Args:
            alert_data: Fraud alert data
            config: Optional configuration for the run

        Returns:
            Final state after workflow completion
        """
        from .state import create_initial_state

        # Create initial state
        initial_state = create_initial_state(alert_data)

        # Configure run
        run_config = {
            "configurable": {"thread_id": alert_data.get("alert_id", "unknown")},
        }
        if config:
            run_config.update(config)

        # Run workflow
        result = self.graph.invoke(initial_state, config=run_config)

        return result

    def get_graph(self):
        """Get the underlying LangGraph instance."""
        return self.graph

    def print_graph(self):
        """Print the graph structure for debugging."""
        try:
            from IPython.display import Image, display

            display(Image(self.graph.get_graph().draw_mermaid_png()))
        except Exception:
            print(self.graph.get_graph().print_ascii())


# Create singleton instance
_fraud_triage_graph = None


def get_fraud_triage_graph() -> FraudTriageGraph:
    """Get or create the fraud triage graph instance."""
    global _fraud_triage_graph
    if _fraud_triage_graph is None:
        _fraud_triage_graph = create_fraud_triage_graph()
    return _fraud_triage_graph
