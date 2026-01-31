"""
Experiment history tracking and management.

This module provides persistent storage and retrieval of
experiment results with versioning and lineage tracking.
"""

from src.prompt_optimizer.history.history import (
    # Enums
    HistorySortOrder,
    # Data classes
    ExperimentHistoryEntry,
    ExperimentLineage,
    # Main class
    HistoryManager,
    create_history_manager,
)

__all__ = [
    "HistorySortOrder",
    "ExperimentHistoryEntry",
    "ExperimentLineage",
    "HistoryManager",
    "create_history_manager",
]
