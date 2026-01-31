"""
State definitions for AgenticFlow.

This module contains TypedDict state definitions for:
- Agent states
- Workflow states
- Shared state between agents
"""

# Type aliases
from src.state.workflow_state import (
    TaskType,
    WorkflowStatus,
    ApprovalStatus,
    StateUpdater,
    StatePredicate,
)

# TypedDict classes
from src.state.workflow_state import (
    AgentOutput,
    ResearchResult,
    WorkflowState,
)

# Helper functions
from src.state.workflow_state import (
    create_initial_state,
    update_state,
    is_complete,
    is_error,
    needs_revision,
    get_progress,
    get_status_message,
    get_next_step,
    advance_step,
    mark_complete,
    mark_error,
)

__all__ = [
    # Type aliases
    "TaskType",
    "WorkflowStatus",
    "ApprovalStatus",
    "StateUpdater",
    "StatePredicate",

    # TypedDict classes
    "AgentOutput",
    "ResearchResult",
    "WorkflowState",

    # Helper functions
    "create_initial_state",
    "update_state",
    "is_complete",
    "is_error",
    "needs_revision",
    "get_progress",
    "get_status_message",
    "get_next_step",
    "advance_step",
    "mark_complete",
    "mark_error",
]
