"""
Experiment history tracking and management.

This module provides persistent storage and retrieval of
experiment results with versioning and lineage tracking.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime
from pathlib import Path
import json
import logging
import uuid

from src.prompt_optimizer.experiments.framework import ExperimentResult

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class HistorySortOrder(Enum):
    """Sort order for history queries."""

    NEWEST_FIRST = "newest_first"
    OLDEST_FIRST = "oldest_first"
    NAME_ASC = "name_asc"
    NAME_DESC = "name_desc"
    BEST_SCORE = "best_score"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ExperimentHistoryEntry:
    """
    Entry in experiment history.

    Attributes:
        experiment_id: Unique experiment identifier
        name: Experiment name
        description: Experiment description
        status: Experiment status
        timestamp: When experiment was created
        result_path: Path to full result file
        best_variant: ID of best variant
        best_score: Score of best variant
        metrics_tracked: Metrics that were tracked
        tags: Tags for filtering
        metadata: Additional metadata
    """

    experiment_id: str
    name: str
    description: str
    status: str
    timestamp: str
    result_path: str
    best_variant: Optional[str]
    best_score: float
    metrics_tracked: List[str]
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "timestamp": self.timestamp,
            "result_path": self.result_path,
            "best_variant": self.best_variant,
            "best_score": self.best_score,
            "metrics_tracked": self.metrics_tracked,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentHistoryEntry":
        """Create from dictionary."""
        return cls(
            experiment_id=data["experiment_id"],
            name=data["name"],
            description=data["description"],
            status=data["status"],
            timestamp=data["timestamp"],
            result_path=data["result_path"],
            best_variant=data.get("best_variant"),
            best_score=data["best_score"],
            metrics_tracked=data["metrics_tracked"],
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExperimentLineage:
    """
    Lineage tracking for prompt variants.

    Attributes:
        variant_id: Variant identifier
        parent_variants: Parent variant IDs
        experiments: Experiments that included this variant
        creation_timestamp: When variant was created
        strategy: Strategy used to create variant
    """

    variant_id: str
    parent_variants: List[str]
    experiments: List[str]
    creation_timestamp: str
    strategy: str


# ============================================================================
# Main History Manager
# ============================================================================

class HistoryManager:
    """
    Manage experiment history with persistent storage.

    Provides storage, retrieval, and querying of experiment results.
    """

    def __init__(self, storage_dir: str = "data/experiments/history"):
        """
        Initialize history manager.

        Args:
            storage_dir: Directory for storing history
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.history_file = self.storage_dir / "history.json"
        self.lineage_file = self.storage_dir / "lineage.json"

        # Load existing history
        self._load_history()

    def add_experiment(
        self,
        result: ExperimentResult,
        result_path: str,
        tags: Optional[List[str]] = None,
    ) -> ExperimentHistoryEntry:
        """
        Add experiment to history.

        Args:
            result: Experiment result
            result_path: Path to saved result file
            tags: Optional tags for filtering

        Returns:
            Created history entry
        """
        # Get best score
        best_score = 0.0
        if result.best_variant and result.best_variant in result.variant_results:
            best_result = result.variant_results[result.best_variant]
            if best_result.mean_scores:
                best_score = max(best_result.mean_scores.values())

        entry = ExperimentHistoryEntry(
            experiment_id=result.experiment_id,
            name=result.config.name,
            description=result.config.description,
            status=result.status.value,
            timestamp=result.started_at,
            result_path=result_path,
            best_variant=result.best_variant,
            best_score=best_score,
            metrics_tracked=result.config.metrics,
            tags=tags or [],
            metadata={
                "total_variants": len(result.variant_results),
                "total_time": result.total_time,
            },
        )

        self.history.append(entry)
        self._save_history()

        # Update lineage
        self._update_lineage(result)

        return entry

    def get_experiment(
        self,
        experiment_id: str,
    ) -> Optional[ExperimentResult]:
        """
        Load full experiment result by ID.

        Args:
            experiment_id: Experiment identifier

        Returns:
            ExperimentResult if found
        """
        # Find in history
        entry = next(
            (e for e in self.history if e.experiment_id == experiment_id),
            None
        )

        if not entry:
            return None

        # Load from file
        result_path = Path(entry.result_path)
        if not result_path.exists():
            logger.warning(f"Result file not found: {result_path}")
            return None

        return ExperimentResult.load(str(result_path))

    def list_experiments(
        self,
        limit: int = 100,
        sort_by: HistorySortOrder = HistorySortOrder.NEWEST_FIRST,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[ExperimentHistoryEntry]:
        """
        List experiments with optional filtering and sorting.

        Args:
            limit: Maximum number of experiments to return
            sort_by: Sort order
            filters: Optional filters (status, tags, etc.)

        Returns:
            List of history entries
        """
        results = self.history.copy()

        # Apply filters
        if filters:
            if "status" in filters:
                results = [e for e in results if e.status == filters["status"]]

            if "tags" in filters:
                filter_tags = set(filters["tags"])
                results = [
                    e for e in results
                    if filter_tags.intersection(e.tags)
                ]

            if "name_contains" in filters:
                substring = filters["name_contains"].lower()
                results = [e for e in results if substring in e.name.lower()]

        # Sort
        if sort_by == HistorySortOrder.NEWEST_FIRST:
            results.sort(key=lambda e: e.timestamp, reverse=True)
        elif sort_by == HistorySortOrder.OLDEST_FIRST:
            results.sort(key=lambda e: e.timestamp)
        elif sort_by == HistorySortOrder.NAME_ASC:
            results.sort(key=lambda e: e.name.lower())
        elif sort_by == HistorySortOrder.NAME_DESC:
            results.sort(key=lambda e: e.name.lower(), reverse=True)
        elif sort_by == HistorySortOrder.BEST_SCORE:
            results.sort(key=lambda e: e.best_score, reverse=True)

        # Limit
        return results[:limit]

    def get_lineage(
        self,
        variant_id: str,
    ) -> Optional[ExperimentLineage]:
        """
        Get lineage information for a variant.

        Args:
            variant_id: Variant identifier

        Returns:
            ExperimentLineage if found
        """
        return self.lineage.get(variant_id)

    def get_variant_history(
        self,
        variant_id: str,
    ) -> List[ExperimentHistoryEntry]:
        """
        Get all experiments that included a variant.

        Args:
            variant_id: Variant identifier

        Returns:
            List of experiments
        """
        return [
            e for e in self.history
            if variant_id in self._get_variants_in_experiment(e)
        ]

    def compare_experiments(
        self,
        experiment_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Compare multiple experiments.

        Args:
            experiment_ids: List of experiment IDs to compare

        Returns:
            Comparison dictionary
        """
        experiments = []
        for exp_id in experiment_ids:
            result = self.get_experiment(exp_id)
            if result:
                experiments.append(result)

        if not experiments:
            return {"error": "No valid experiments found"}

        # Build comparison
        comparison = {
            "num_experiments": len(experiments),
            "experiments": [],
            "best_scores": {},
            "avg_scores": {},
        }

        for exp in experiments:
            entry = next(
                (e for e in self.history if e.experiment_id == exp.experiment_id),
                None
            )

            comparison["experiments"].append({
                "id": exp.experiment_id,
                "name": exp.config.name,
                "best_variant": exp.best_variant,
                "num_variants": len(exp.variant_results),
                "timestamp": exp.started_at,
            })

            if entry:
                comparison["best_scores"][exp.experiment_id] = entry.best_score

        return comparison

    def delete_experiment(
        self,
        experiment_id: str,
        delete_files: bool = False,
    ) -> bool:
        """
        Delete experiment from history.

        Args:
            experiment_id: Experiment to delete
            delete_files: Whether to delete result files

        Returns:
            True if deleted
        """
        # Find entry
        entry = next(
            (e for e in self.history if e.experiment_id == experiment_id),
            None
        )

        if not entry:
            return False

        # Remove from history
        self.history = [e for e in self.history if e.experiment_id != experiment_id]
        self._save_history()

        # Delete files if requested
        if delete_files:
            result_path = Path(entry.result_path)
            if result_path.exists():
                result_path.unlink()

        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics."""
        return {
            "total_experiments": len(self.history),
            "completed": sum(1 for e in self.history if e.status == "completed"),
            "failed": sum(1 for e in self.history if e.status == "failed"),
            "total_variants": sum(e.metadata.get("total_variants", 0) for e in self.history),
            "unique_variants": len(self.lineage),
        }

    def _load_history(self) -> None:
        """Load history from disk."""
        self.history: List[ExperimentHistoryEntry] = []

        if self.history_file.exists():
            try:
                with open(self.history_file, "r") as f:
                    data = json.load(f)
                    self.history = [
                        ExperimentHistoryEntry.from_dict(e)
                        for e in data
                    ]
            except Exception as e:
                logger.error(f"Error loading history: {e}")

        # Load lineage
        self.lineage: Dict[str, ExperimentLineage] = {}

        if self.lineage_file.exists():
            try:
                with open(self.lineage_file, "r") as f:
                    data = json.load(f)
                    for var_id, lineage_data in data.items():
                        self.lineage[var_id] = ExperimentLineage(
                            variant_id=var_id,
                            parent_variants=lineage_data.get("parent_variants", []),
                            experiments=lineage_data.get("experiments", []),
                            creation_timestamp=lineage_data.get("creation_timestamp", ""),
                            strategy=lineage_data.get("strategy", ""),
                        )
            except Exception as e:
                logger.error(f"Error loading lineage: {e}")

    def _save_history(self) -> None:
        """Save history to disk."""
        try:
            with open(self.history_file, "w") as f:
                json.dump([e.to_dict() for e in self.history], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving history: {e}")

    def _save_lineage(self) -> None:
        """Save lineage to disk."""
        try:
            data = {
                var_id: {
                    "parent_variants": l.parent_variants,
                    "experiments": l.experiments,
                    "creation_timestamp": l.creation_timestamp,
                    "strategy": l.strategy,
                }
                for var_id, l in self.lineage.items()
            }
            with open(self.lineage_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving lineage: {e}")

    def _update_lineage(self, result: ExperimentResult) -> None:
        """Update lineage from experiment result."""
        for var_id, variant_result in result.variant_results.items():
            if var_id not in self.lineage:
                self.lineage[var_id] = ExperimentLineage(
                    variant_id=var_id,
                    parent_variants=[],
                    experiments=[],
                    creation_timestamp=result.started_at,
                    strategy=variant_result.strategy.value,
                )

            # Add experiment if not present
            if result.experiment_id not in self.lineage[var_id].experiments:
                self.lineage[var_id].experiments.append(result.experiment_id)

        self._save_lineage()

    def _get_variants_in_experiment(
        self,
        entry: ExperimentHistoryEntry,
    ) -> List[str]:
        """Get list of variant IDs from experiment entry."""
        # This is a simplified version - in practice you'd load
        # the full result or store variant IDs in the entry
        return []


# ============================================================================
# Convenience Functions
# ============================================================================

def create_history_manager(
    storage_dir: str = "data/experiments/history",
) -> HistoryManager:
    """
    Factory function to create a HistoryManager.

    Args:
        storage_dir: Directory for storing history

    Returns:
        Configured HistoryManager
    """
    return HistoryManager(storage_dir=storage_dir)


# Export
__all__ = [
    # Enums
    "HistorySortOrder",
    # Data classes
    "ExperimentHistoryEntry",
    "ExperimentLineage",
    # Main class
    "HistoryManager",
    "create_history_manager",
]
