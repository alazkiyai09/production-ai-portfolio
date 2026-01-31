"""
LangGraph workflow definition for fraud alert triage.

This module defines the complete workflow graph including nodes,
edges, conditional routing, and state management.
"""

import logging
from typing import Any, Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.models.state import FraudTriageState
from src.agents.triage_nodes import (
    assess_risk_node,
    finalize_node,
    gather_context_node,
    human_review_node,
    parse_alert_node,
    route_after_assessment,
    route_after_human_review,
)

logger = logging.getLogger(__name__)


def create_fraud_triage_workflow() -> StateGraph:
    """
    Create the LangGraph workflow for fraud alert triage.

    The workflow follows this path:
        1. parse_alert - Validate and extract alert information
        2. gather_context - Collect data from multiple sources
        3. assess_risk - LLM-powered risk assessment
        4a. finalize - If no human review needed
        4b. human_review - Wait for human analyst decision
        5. finalize - Complete workflow and compute metrics

    Args:
        None

    Returns:
        Compiled StateGraph ready for execution
    """
    logger.info("Creating fraud triage workflow")

    # Create the graph with our state schema
    workflow = StateGraph(FraudTriageState)

    # -------------------------------------------------------------------------
    # Add Nodes
    # -------------------------------------------------------------------------
    workflow.add_node("parse_alert", parse_alert_node)
    workflow.add_node("gather_context", gather_context_node)
    workflow.add_node("assess_risk", assess_risk_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("finalize", finalize_node)

    logger.info("Added nodes: parse_alert, gather_context, assess_risk, human_review, finalize")

    # -------------------------------------------------------------------------
    # Define Entry Point
    # -------------------------------------------------------------------------
    workflow.set_entry_point("parse_alert")

    # -------------------------------------------------------------------------
    # Define Edges (Linear Flow)
    # -------------------------------------------------------------------------
    # Parse → Gather
    workflow.add_edge("parse_alert", "gather_context")

    # Gather → Assess
    workflow.add_edge("gather_context", "assess_risk")

    # -------------------------------------------------------------------------
    # Define Conditional Routing
    # -------------------------------------------------------------------------
    # After risk assessment, either route to human review or finalize
    workflow.add_conditional_edges(
        source="assess_risk",
        path=route_after_assessment,
        path_map={
            "human_review": "human_review",
            "finalize": "finalize",
        },
    )

    # After human review, either wait for more input or finalize
    workflow.add_conditional_edges(
        source="human_review",
        path=route_after_human_review,
        path_map={
            "human_review": "human_review",  # Still waiting
            "finalize": "finalize",
        },
    )

    # -------------------------------------------------------------------------
    # Define End Points
    # -------------------------------------------------------------------------
    workflow.add_edge("finalize", END)

    # -------------------------------------------------------------------------
    # Compile with Checkpointer
    # -------------------------------------------------------------------------
    # MemorySaver enables state persistence for human-in-the-loop
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    logger.info("Fraud triage workflow compiled successfully")

    return app


class FraudTriageWorkflow:
    """
    Wrapper class for the fraud triage LangGraph workflow.

    Provides a clean interface for running and managing the workflow.
    """

    def __init__(self, graph: StateGraph):
        """
        Initialize workflow wrapper.

        Args:
            graph: Compiled LangGraph StateGraph
        """
        self.graph = graph
        logger.info("FraudTriageWorkflow initialized")

    async def arun(
        self,
        alert_id: str,
        alert_type,  # Will be converted to AlertType enum
        transaction_amount: float,
        customer_id: str,
        **kwargs: Any,
    ) -> FraudTriageState:
        """
        Run the fraud triage workflow asynchronously.

        Args:
            alert_id: Unique alert identifier
            alert_type: Type of fraud alert (AlertType enum or string)
            transaction_amount: Amount of the flagged transaction
            customer_id: Customer identifier
            **kwargs: Additional optional state fields

        Returns:
            Final workflow state after completion

        Example:
            >>> result = await workflow.arun(
            ...     alert_id="ALERT-001",
            ...     alert_type=AlertType.LOCATION_MISMATCH,
            ...     transaction_amount=4250.00,
            ...     customer_id="CUST-12345",
            ...     account_id="ACC-67890",
            ...     transaction_country="NG"
            ... )
        """
        from src.models.state import AlertType, create_initial_state

        # Convert alert_type to enum if it's a string
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

        logger.info(f"Starting workflow for alert {alert_id}")

        # Configure run with thread_id for checkpointing
        config = {
            "configurable": {
                "thread_id": alert_id,
            }
        }

        # Run workflow
        try:
            result = await self.graph.ainvoke(initial_state, config=config)
            logger.info(f"Workflow completed for alert {alert_id}")
            return result
        except Exception as e:
            logger.error(f"Workflow failed for alert {alert_id}: {e}")
            # Return state with error information
            initial_state["error_message"] = str(e)
            initial_state["processing_completed"] = None
            return initial_state

    def run(
        self,
        alert_id: str,
        alert_type,
        transaction_amount: float,
        customer_id: str,
        **kwargs: Any,
    ) -> FraudTriageState:
        """
        Run the fraud triage workflow synchronously.

        Args:
            alert_id: Unique alert identifier
            alert_type: Type of fraud alert (AlertType enum or string)
            transaction_amount: Amount of the flagged transaction
            customer_id: Customer identifier
            **kwargs: Additional optional state fields

        Returns:
            Final workflow state after completion
        """
        from src.models.state import AlertType, create_initial_state

        # Convert alert_type to enum if it's a string
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

        logger.info(f"Starting workflow (sync) for alert {alert_id}")

        # Configure run
        config = {
            "configurable": {
                "thread_id": alert_id,
            }
        }

        # Run workflow
        try:
            result = self.graph.invoke(initial_state, config=config)
            logger.info(f"Workflow completed for alert {alert_id}")
            return result
        except Exception as e:
            logger.error(f"Workflow failed for alert {alert_id}: {e}")
            initial_state["error_message"] = str(e)
            return initial_state

    async def astream(
        self,
        alert_id: str,
        alert_type,
        transaction_amount: float,
        customer_id: str,
        **kwargs: Any,
    ):
        """
        Stream workflow execution asynchronously.

        Yields state updates as the workflow progresses through nodes.

        Args:
            alert_id: Unique alert identifier
            alert_type: Type of fraud alert
            transaction_amount: Amount of the flagged transaction
            customer_id: Customer identifier
            **kwargs: Additional optional state fields

        Yields:
            Tuples of (node_name, state_update) as workflow progresses
        """
        from src.models.state import AlertType, create_initial_state

        # Convert alert_type to enum if it's a string
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

        logger.info(f"Streaming workflow for alert {alert_id}")

        # Configure run
        config = {
            "configurable": {
                "thread_id": alert_id,
            }
        }

        # Stream workflow
        try:
            async for event in self.graph.astream(initial_state, config=config):
                yield event
        except Exception as e:
            logger.error(f"Workflow stream failed for alert {alert_id}: {e}")
            yield ("error", {"error_message": str(e)})

    def get_state(
        self,
        alert_id: str,
    ) -> FraudTriageState | None:
        """
        Get current state for an alert (for human-in-the-loop resumption).

        Args:
            alert_id: Alert identifier

        Returns:
            Current state if exists, None otherwise
        """
        config = {
            "configurable": {
                "thread_id": alert_id,
            }
        }

        try:
            state = self.graph.get_state(config)
            return state.values if state else None
        except Exception as e:
            logger.error(f"Error getting state for alert {alert_id}: {e}")
            return None

    def update_state(
        self,
        alert_id: str,
        updates: dict[str, Any],
    ) -> FraudTriageState | None:
        """
        Update state for an alert (for submitting human review).

        Args:
            alert_id: Alert identifier
            updates: State fields to update

        Returns:
            Updated state if successful, None otherwise
        """
        config = {
            "configurable": {
                "thread_id": alert_id,
            }
        }

        try:
            # Update state and resume workflow
            self.graph.update_state(config, updates)
            logger.info(f"State updated for alert {alert_id}: {list(updates.keys())}")

            # Get updated state
            return self.get_state(alert_id)
        except Exception as e:
            logger.error(f"Error updating state for alert {alert_id}: {e}")
            return None

    def get_graph(self) -> StateGraph:
        """Get the underlying LangGraph instance."""
        return self.graph

    def print_graph(self) -> str:
        """
        Print the graph structure as ASCII art.

        Returns:
            ASCII representation of the graph
        """
        try:
            return self.graph.get_graph().print_ascii()
        except Exception as e:
            return f"Unable to render graph: {e}"

    def get_mermaid(self) -> str:
        """
        Get Mermaid diagram representation of the graph.

        Returns:
            Mermaid diagram string
        """
        try:
            return str(self.graph.get_graph().print_mermaid())
        except Exception as e:
            return f"Unable to generate Mermaid diagram: {e}"


# =============================================================================
# Singleton Instance
# =============================================================================

_workflow_instance: FraudTriageWorkflow | None = None


def get_workflow() -> FraudTriageWorkflow:
    """
    Get or create the fraud triage workflow singleton instance.

    Returns:
        FraudTriageWorkflow instance
    """
    global _workflow_instance

    if _workflow_instance is None:
        logger.info("Creating workflow singleton instance")
        graph = create_fraud_triage_workflow()
        _workflow_instance = FraudTriageWorkflow(graph)

    return _workflow_instance


def reset_workflow():
    """
    Reset the workflow singleton instance.

    Useful for testing or when configuration changes.
    """
    global _workflow_instance
    _workflow_instance = None
    logger.info("Workflow singleton reset")


# =============================================================================
# Convenience Functions
# =============================================================================

async def triage_alert(
    alert_id: str,
    alert_type,
    transaction_amount: float,
    customer_id: str,
    **kwargs: Any,
) -> FraudTriageState:
    """
    Convenience function to triage a single alert.

    Args:
        alert_id: Unique alert identifier
        alert_type: Type of fraud alert
        transaction_amount: Amount of the flagged transaction
        customer_id: Customer identifier
        **kwargs: Additional optional fields

    Returns:
        Final workflow state
    """
    workflow = get_workflow()
    return await workflow.arun(
        alert_id=alert_id,
        alert_type=alert_type,
        transaction_amount=transaction_amount,
        customer_id=customer_id,
        **kwargs,
    )


def triage_alert_sync(
    alert_id: str,
    alert_type,
    transaction_amount: float,
    customer_id: str,
    **kwargs: Any,
) -> FraudTriageState:
    """
    Convenience function to triage a single alert (synchronous).

    Args:
        alert_id: Unique alert identifier
        alert_type: Type of fraud alert
        transaction_amount: Amount of the flagged transaction
        customer_id: Customer identifier
        **kwargs: Additional optional fields

    Returns:
        Final workflow state
    """
    workflow = get_workflow()
    return workflow.run(
        alert_id=alert_id,
        alert_type=alert_type,
        transaction_amount=transaction_amount,
        customer_id=customer_id,
        **kwargs,
    )
