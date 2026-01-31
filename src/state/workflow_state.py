"""
State definitions for AgenticFlow workflow.

This module defines the TypedDict states used throughout the multi-agent
workflow system, including the main WorkflowState and helper classes.
"""

from __future__ import annotations

import operator
from datetime import datetime
from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    Optional,
)

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


# =============================================================================
# Type Definitions
# =============================================================================

TaskType = Literal["research", "analysis", "content_creation", "general"]
WorkflowStatus = Literal[
    "planning",
    "researching",
    "analyzing",
    "writing",
    "reviewing",
    "complete",
    "error",
]
ApprovalStatus = Literal["pending", "approved", "needs_revision", "rejected"]


# =============================================================================
# Agent Output
# =============================================================================

class AgentOutput(TypedDict):
    """
    Output from a single agent step.

    This structure captures the result of each agent's execution,
    including what tools were used and when it occurred.

    Attributes:
        agent_name: Name of the agent that produced this output
        output: The main output content from the agent
        tools_used: List of tool names invoked during execution
        timestamp: ISO format timestamp of when the output was generated
        duration_seconds: Time taken for agent execution (optional)
        metadata: Additional agent-specific metadata (optional)
        success: Whether the agent completed successfully
    """
    agent_name: str
    output: str
    tools_used: list[str]
    timestamp: str
    duration_seconds: Optional[float]
    metadata: Optional[dict[str, Any]]
    success: bool


# =============================================================================
# Research Result
# =============================================================================

class ResearchResult(TypedDict):
    """
    Structured data for research results.

    Attributes:
        query: The search query that generated this result
        source: Source of the information (e.g., "tavily", "duckduckgo")
        title: Title of the result
        url: URL of the source
        content: Main content/snippet
        relevance_score: Score indicating relevance to the query (0-1)
        timestamp: When the research was conducted
    """
    query: str
    source: str
    title: str
    url: str
    content: str
    relevance_score: float
    timestamp: str


# =============================================================================
# Main Workflow State
# =============================================================================

class WorkflowState(TypedDict):
    """
    Main state for the multi-agent workflow.

    This TypedDict defines the complete state structure that flows through
    the LangGraph state machine. All agents read from and write to this state.

    The state is organized into logical sections:
    - Input: Initial task and type
    - Planning: Task decomposition and step tracking
    - Research: Search queries and results
    - Analysis: Processed findings and insights
    - Writing: Draft content and revisions
    - Review: Feedback and approval status
    - Output: Final results
    - Metadata: Status tracking and error handling
    - Messages: LangGraph message history (with accumulation)
    - Agent Outputs: Execution tracking for observability

    Notes:
        - The `messages` field uses Annotated with operator.add to enable
          message accumulation across workflow steps (LangGraph pattern)
        - All agents should use the helper functions to modify state
        - State modifications should be atomic per agent step
    """

    # =========================================================================
    # Input Section
    # =========================================================================

    task: str
    """The primary task description provided by the user."""

    task_type: TaskType
    """Category of task for routing and specialized processing."""

    task_context: Optional[str]
    """Additional context or constraints for the task."""

    # =========================================================================
    # Planning Section
    # =========================================================================

    plan: list[str]
    """List of steps decomposed by the Planner agent."""

    current_step: int
    """Index of the current step being executed (0-based)."""

    # =========================================================================
    # Research Section
    # =========================================================================

    research_queries: list[str]
    """Search queries generated for information gathering."""

    research_results: list[ResearchResult]
    """Collected research data with sources and relevance."""

    # =========================================================================
    # Analysis Section
    # =========================================================================

    analysis: str
    """Detailed analysis produced by the Analyzer agent."""

    key_findings: list[str]
    """Key insights extracted during analysis."""

    data_patterns: list[str]
    """Patterns identified in the research data."""

    # =========================================================================
    # Writing Section
    # =========================================================================

    draft: str
    """Draft content produced by the Writer agent."""

    revision_count: int
    """Number of revisions made to the draft."""

    draft_history: list[str]
    """Historical versions of the draft for tracking changes."""

    # =========================================================================
    # Review Section
    # =========================================================================

    feedback: list[str]
    """Feedback items from the Reviewer agent."""

    approval_status: ApprovalStatus
    """Current approval status of the output."""

    review_criteria: Optional[list[str]]
    """Criteria used for evaluation (optional)."""

    # =========================================================================
    # Output Section
    # =========================================================================

    final_output: str
    """Final approved output from the workflow."""

    output_format: Optional[str]
    """Desired output format (e.g., "markdown", "json", "html")."""

    # =========================================================================
    # Metadata Section
    # =========================================================================

    status: WorkflowStatus
    """Current workflow status for progress tracking."""

    error: Optional[str]
    """Error message if workflow encountered an error."""

    start_time: str
    """ISO format timestamp of workflow start."""

    end_time: Optional[str]
    """ISO format timestamp of workflow completion."""

    iteration_count: int
    """Number of iterations/steps executed so far."""

    # =========================================================================
    # Message History (LangGraph)
    # =========================================================================

    messages: Annotated[list[BaseMessage], operator.add]
    """
    Accumulated message history for LangGraph.

    Uses operator.add annotation to enable message accumulation
    across workflow steps. Each agent can append messages to this list.
    """

    # =========================================================================
    # Agent Outputs (Tracking)
    # =========================================================================

    agent_outputs: list[AgentOutput]
    """List of all agent outputs for observability and debugging."""

    # =========================================================================
    # Additional Metadata
    # =========================================================================

    metadata: dict[str, Any]
    """Additional metadata for custom use cases."""


# =============================================================================
# Helper Functions
# =============================================================================

def create_initial_state(
    task: str,
    task_type: TaskType = "general",
    task_context: Optional[str] = None,
    output_format: Optional[str] = None,
) -> WorkflowState:
    """
    Create an initial WorkflowState for a new task.

    This function initializes the workflow state with default values and
    the provided task information. All collections are empty and timestamps
    are set to the current time.

    Args:
        task: The primary task description
        task_type: Category of task (research, analysis, content_creation, general)
        task_context: Additional context or constraints (optional)
        output_format: Desired output format (optional)

    Returns:
        WorkflowState: Initialized workflow state ready for processing

    Example:
        >>> state = create_initial_state(
        ...     task="Research the latest AI trends",
        ...     task_type="research"
        ... )
        >>> print(state["status"])
        planning
    """
    current_time = datetime.utcnow().isoformat()

    return WorkflowState(
        # Input
        task=task,
        task_type=task_type,
        task_context=task_context,

        # Planning
        plan=[],
        current_step=0,

        # Research
        research_queries=[],
        research_results=[],

        # Analysis
        analysis="",
        key_findings=[],
        data_patterns=[],

        # Writing
        draft="",
        revision_count=0,
        draft_history=[],

        # Review
        feedback=[],
        approval_status="pending",
        review_criteria=None,

        # Output
        final_output="",
        output_format=output_format,

        # Metadata
        status="planning",
        error=None,
        start_time=current_time,
        end_time=None,
        iteration_count=0,

        # Messages
        messages=[],

        # Agent outputs
        agent_outputs=[],

        # Additional metadata
        metadata={},
    )


def update_state(
    state: WorkflowState,
    updates: dict[str, Any],
    agent_name: Optional[str] = None,
) -> WorkflowState:
    """
    Update a WorkflowState with new values.

    This function creates an updated copy of the state with the specified
    fields modified. It handles special cases like message accumulation
    and agent output tracking.

    Args:
        state: Current workflow state
        updates: Dictionary of fields to update with new values
        agent_name: Optional agent name for tracking agent outputs

    Returns:
        WorkflowState: Updated workflow state

    Notes:
        - This function does not mutate the original state
        - Messages are accumulated using operator.add (handled by LangGraph)
        - Agent outputs are automatically tracked if agent_name is provided

    Example:
        >>> state = create_initial_state("My task")
        >>> updated = update_state(
        ...     state,
        ...     {"status": "researching", "plan": ["Step 1", "Step 2"]}
        ... )
    """
    # Create a shallow copy of the state
    updated_state = WorkflowState({**state, **updates})

    # Track agent output if agent_name provided and output exists
    if agent_name and "output" in updates:
        agent_output = AgentOutput(
            agent_name=agent_name,
            output=updates["output"],
            tools_used=updates.get("tools_used", []),
            timestamp=datetime.utcnow().isoformat(),
            duration_seconds=updates.get("duration_seconds"),
            metadata=updates.get("metadata"),
            success=updates.get("success", True),
        )
        updated_state["agent_outputs"] = list(state["agent_outputs"]) + [agent_output]

    # Increment iteration count if not explicitly set
    if "iteration_count" not in updates:
        updated_state["iteration_count"] = state["iteration_count"] + 1

    return updated_state


def is_complete(state: WorkflowState) -> bool:
    """
    Check if the workflow has completed successfully.

    A workflow is considered complete if:
    - Status is "complete"
    - Final output exists
    - Approval status is "approved"

    Args:
        state: Current workflow state

    Returns:
        bool: True if workflow is complete, False otherwise

    Example:
        >>> state = create_initial_state("My task")
        >>> state["status"] = "complete"
        >>> state["final_output"] = "Done"
        >>> state["approval_status"] = "approved"
        >>> is_complete(state)
        True
    """
    return (
        state["status"] == "complete"
        and bool(state["final_output"])
        and state["approval_status"] == "approved"
    )


def is_error(state: WorkflowState) -> bool:
    """
    Check if the workflow has encountered an error.

    Args:
        state: Current workflow state

    Returns:
        bool: True if workflow is in error state, False otherwise
    """
    return state["status"] == "error" and state["error"] is not None


def needs_revision(state: WorkflowState) -> bool:
    """
    Check if the workflow needs revision based on reviewer feedback.

    Args:
        state: Current workflow state

    Returns:
        bool: True if revision is needed, False otherwise
    """
    return state["approval_status"] == "needs_revision"


def get_progress(state: WorkflowState) -> dict[str, Any]:
    """
    Get progress information for the workflow.

    This function calculates and returns various progress metrics
    useful for monitoring and displaying workflow status.

    Args:
        state: Current workflow state

    Returns:
        Dictionary containing:
        - status: Current workflow status
        - progress_percent: Estimated completion percentage (0-100)
        - steps_completed: Number of plan steps completed
        - total_steps: Total number of plan steps
        - iteration_count: Number of iterations executed
        - agents_executed: Number of unique agents that have run
        - has_research: Whether research has been conducted
        - has_analysis: Whether analysis has been performed
        - has_draft: Whether a draft has been created
        - is_reviewed: Whether the output has been reviewed
        - elapsed_time_seconds: Time since workflow start

    Example:
        >>> state = create_initial_state("My task")
        >>> state["plan"] = ["Step 1", "Step 2", "Step 3"]
        >>> state["current_step"] = 1
        >>> progress = get_progress(state)
        >>> print(progress["progress_percent"])
        33.33
    """
    # Calculate elapsed time
    start_time = datetime.fromisoformat(state["start_time"])
    end_time = (
        datetime.fromisoformat(state["end_time"])
        if state["end_time"]
        else datetime.utcnow()
    )
    elapsed_time = (end_time - start_time).total_seconds()

    # Calculate plan progress
    total_steps = len(state["plan"])
    steps_completed = state["current_step"] if total_steps > 0 else 0
    plan_progress = (steps_completed / total_steps * 100) if total_steps > 0 else 0

    # Calculate overall progress based on status
    status_progress = {
        "planning": 5,
        "researching": 25,
        "analyzing": 50,
        "writing": 75,
        "reviewing": 90,
        "complete": 100,
        "error": 0,
    }
    base_progress = status_progress.get(state["status"], 0)

    # Use the higher of plan-based or status-based progress
    progress_percent = max(base_progress, plan_progress)

    # Count unique agents executed
    agents_executed = len(set(output["agent_name"] for output in state["agent_outputs"]))

    return {
        "status": state["status"],
        "progress_percent": round(progress_percent, 2),
        "steps_completed": steps_completed,
        "total_steps": total_steps,
        "iteration_count": state["iteration_count"],
        "agents_executed": agents_executed,
        "has_research": len(state["research_results"]) > 0,
        "has_analysis": bool(state["analysis"]),
        "has_draft": bool(state["draft"]),
        "is_reviewed": state["approval_status"] != "pending",
        "elapsed_time_seconds": round(elapsed_time, 2),
    }


def get_status_message(state: WorkflowState) -> str:
    """
    Get a human-readable status message for the workflow.

    Args:
        state: Current workflow state

    Returns:
        A descriptive message about the current workflow state
    """
    status_messages = {
        "planning": "Creating execution plan...",
        "researching": f"Gathering information ({len(state['research_results'])} results)",
        "analyzing": "Analyzing data and identifying patterns...",
        "writing": f"Creating content (revision {state['revision_count']})",
        "reviewing": "Reviewing and evaluating output...",
        "complete": "Workflow completed successfully",
        "error": f"Error: {state['error']}",
    }

    return status_messages.get(state["status"], "Unknown status")


def get_next_step(state: WorkflowState) -> Optional[str]:
    """
    Get the next step in the plan if available.

    Args:
        state: Current workflow state

    Returns:
        The next step description or None if no steps remain
    """
    if state["current_step"] < len(state["plan"]):
        return state["plan"][state["current_step"]]
    return None


def advance_step(state: WorkflowState) -> WorkflowState:
    """
    Advance to the next step in the plan.

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state with current_step incremented
    """
    return update_state(state, {"current_step": state["current_step"] + 1})


def mark_complete(state: WorkflowState, final_output: str) -> WorkflowState:
    """
    Mark the workflow as complete with the final output.

    Args:
        state: Current workflow state
        final_output: The final output content

    Returns:
        Updated workflow state marked as complete
    """
    return update_state(
        state,
        {
            "final_output": final_output,
            "status": "complete",
            "approval_status": "approved",
            "end_time": datetime.utcnow().isoformat(),
        },
    )


def mark_error(state: WorkflowState, error_message: str) -> WorkflowState:
    """
    Mark the workflow as having encountered an error.

    Args:
        state: Current workflow state
        error_message: Description of the error

    Returns:
        Updated workflow state marked as error
    """
    return update_state(
        state,
        {
            "status": "error",
            "error": error_message,
            "end_time": datetime.utcnow().isoformat(),
        },
    )


# =============================================================================
# Type Aliases for Common Use
# =============================================================================

StateUpdater = Callable[[WorkflowState, dict[str, Any]], WorkflowState]
"""Type alias for state update functions."""

StatePredicate = Callable[[WorkflowState], bool]
"""Type alias for state predicate/check functions."""
