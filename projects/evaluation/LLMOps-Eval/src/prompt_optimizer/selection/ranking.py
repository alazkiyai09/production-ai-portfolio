"""
Prompt ranking methods for comparing variants.

This module provides various ranking approaches including
statistical ranking, Pareto fronts, and custom scoring.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Set
from enum import Enum
import numpy as np
from datetime import datetime
import logging

from src.prompt_optimizer.experiments.framework import VariantResult

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class RankingMethod(Enum):
    """Methods for ranking prompt variants."""

    MEAN_SCORE = "mean_score"  # Rank by mean score
    MEDIAN_SCORE = "median_score"  # Rank by median score
    BORDA_COUNT = "borda_count"  # Borda count voting method
    COPELAND = "copeland"  # Copeland pairwise comparison
    RANK_AGGREGATION = "rank_aggregation"  # Aggregate ranks across metrics
    PARETO = "pareto"  # Pareto frontier (non-dominated)
    TOPSIS = "topsis"  # Technique for Order Preference by Similarity to Ideal Solution
    CONDORCET = "condorcet"  # Condorcet winner (beats all others)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RankingEntry:
    """
    Single entry in a ranking.

    Attributes:
        variant_id: Variant identifier
        rank: Position in ranking (1 = best)
        score: Raw score used for ranking
        metrics: Individual metric scores
        metadata: Additional metadata
    """

    variant_id: str
    rank: int
    score: float
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParetoFront:
    """
    Pareto frontier of non-dominated variants.

    Attributes:
        frontier_variants: Variants on the Pareto frontier
        dominated_variants: Dominated variants
        fronts: List of Pareto fronts (ranked)
        criteria: Criteria used for Pareto analysis
    """

    frontier_variants: List[str]
    dominated_variants: List[str]
    fronts: List[List[str]]  # First front is optimal
    criteria: List[str]


@dataclass
class RankingResult:
    """
    Complete ranking of prompt variants.

    Attributes:
        method: Ranking method used
        rankings: Full ranking by metric
        overall_ranking: Overall ranking
        pareto_front: Pareto frontier (if applicable)
        ties: Groups of tied variants
        metadata: Additional metadata
    """

    method: RankingMethod
    rankings: Dict[str, List[RankingEntry]]  # metric -> ranking
    overall_ranking: List[RankingEntry]
    pareto_front: Optional[ParetoFront]
    ties: List[Set[str]]  # Groups of tied variants
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_top_n(self, n: int) -> List[str]:
        """Get top N variant IDs."""
        return [entry.variant_id for entry in self.overall_ranking[:n]]

    def get_variant_rank(self, variant_id: str) -> Optional[int]:
        """Get rank of a specific variant."""
        for entry in self.overall_ranking:
            if entry.variant_id == variant_id:
                return entry.rank
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method.value,
            "rankings": {
                metric: [
                    {
                        "variant_id": e.variant_id,
                        "rank": e.rank,
                        "score": e.score,
                        "metrics": e.metrics,
                        "metadata": e.metadata,
                    }
                    for e in ranking
                ]
                for metric, ranking in self.rankings.items()
            },
            "overall_ranking": [
                {
                    "variant_id": e.variant_id,
                    "rank": e.rank,
                    "score": e.score,
                    "metrics": e.metrics,
                    "metadata": e.metadata,
                }
                for e in self.overall_ranking
            ],
            "pareto_front": {
                "frontier_variants": self.pareto_front.frontier_variants,
                "dominated_variants": self.pareto_front.dominated_variants,
                "fronts": self.pareto_front.fronts,
                "criteria": self.pareto_front.criteria,
            } if self.pareto_front else None,
            "ties": [list(tie) for tie in self.ties],
            "metadata": self.metadata,
        }


# ============================================================================
# Main Ranking Class
# ============================================================================

class PromptRanker:
    """
    Rank prompt variants using various methods.

    Provides multiple ranking approaches to handle different
    optimization scenarios and trade-offs.
    """

    def __init__(self):
        """Initialize the prompt ranker."""
        pass

    def rank(
        self,
        variant_results: Dict[str, VariantResult],
        method: RankingMethod = RankingMethod.MEAN_SCORE,
        metrics: Optional[List[str]] = None,
        higher_is_better: bool = True,
    ) -> RankingResult:
        """
        Rank prompt variants using specified method.

        Args:
            variant_results: Results for each variant
            method: Ranking method to use
            metrics: Metrics to consider (None = all)
            higher_is_better: Whether higher scores are better

        Returns:
            RankingResult with full rankings
        """
        if metrics is None:
            # Get all available metrics
            metrics = set()
            for result in variant_results.values():
                metrics.update(result.mean_scores.keys())
            metrics = list(metrics)

        # Dispatch to appropriate method
        if method == RankingMethod.MEAN_SCORE:
            return self._rank_by_mean_score(
                variant_results, metrics, higher_is_better
            )
        elif method == RankingMethod.MEDIAN_SCORE:
            return self._rank_by_median_score(
                variant_results, metrics, higher_is_better
            )
        elif method == RankingMethod.BORDA_COUNT:
            return self._rank_by_borda_count(
                variant_results, metrics, higher_is_better
            )
        elif method == RankingMethod.COPELAND:
            return self._rank_by_copeland(
                variant_results, metrics, higher_is_better
            )
        elif method == RankingMethod.RANK_AGGREGATION:
            return self._rank_by_aggregation(
                variant_results, metrics, higher_is_better
            )
        elif method == RankingMethod.PARETO:
            return self._rank_by_pareto(
                variant_results, metrics
            )
        elif method == RankingMethod.TOPSIS:
            return self._rank_by_topsis(
                variant_results, metrics, higher_is_better
            )
        elif method == RankingMethod.CONDORCET:
            return self._rank_by_condorcet(
                variant_results, metrics, higher_is_better
            )
        else:
            raise ValueError(f"Unknown ranking method: {method}")

    def _rank_by_mean_score(
        self,
        variant_results: Dict[str, VariantResult],
        metrics: List[str],
        higher_is_better: bool,
    ) -> RankingResult:
        """Rank by mean score across metrics."""
        rankings = {}
        overall_scores = {}

        for metric in metrics:
            entries = []
            for var_id, result in variant_results.items():
                score = result.mean_scores.get(metric, 0.0)
                entries.append(RankingEntry(
                    variant_id=var_id,
                    rank=0,
                    score=score,
                    metrics={metric: score},
                ))

            # Sort by score
            reverse = higher_is_better
            entries.sort(key=lambda e: e.score, reverse=reverse)

            # Assign ranks (handle ties)
            self._assign_ranks(entries)

            rankings[metric] = entries

            # Track overall scores
            for entry in entries:
                if entry.variant_id not in overall_scores:
                    overall_scores[entry.variant_id] = 0.0
                overall_scores[entry.variant_id] += entry.score

        # Overall ranking
        overall_entries = []
        for var_id, total_score in overall_scores.items():
            avg_score = total_score / len(metrics)
            overall_entries.append(RankingEntry(
                variant_id=var_id,
                rank=0,
                score=avg_score,
                metrics={m: variant_results[var_id].mean_scores.get(m, 0.0) for m in metrics},
            ))

        reverse = higher_is_better
        overall_entries.sort(key=lambda e: e.score, reverse=reverse)
        self._assign_ranks(overall_entries)

        ties = self._find_ties(overall_entries)

        return RankingResult(
            method=RankingMethod.MEAN_SCORE,
            rankings=rankings,
            overall_ranking=overall_entries,
            pareto_front=None,
            ties=ties,
            metadata={"higher_is_better": higher_is_better},
        )

    def _rank_by_median_score(
        self,
        variant_results: Dict[str, VariantResult],
        metrics: List[str],
        higher_is_better: bool,
    ) -> RankingResult:
        """Rank by median score across metrics."""
        rankings = {}
        overall_scores = {}

        for metric in metrics:
            entries = []
            for var_id, result in variant_results.items():
                score_list = result.scores.get(metric, [])
                median = np.median(score_list) if score_list else 0.0
                entries.append(RankingEntry(
                    variant_id=var_id,
                    rank=0,
                    score=median,
                    metrics={metric: median},
                ))

            reverse = higher_is_better
            entries.sort(key=lambda e: e.score, reverse=reverse)
            self._assign_ranks(entries)

            rankings[metric] = entries

            for entry in entries:
                if entry.variant_id not in overall_scores:
                    overall_scores[entry.variant_id] = 0.0
                overall_scores[entry.variant_id] += entry.score

        # Overall ranking
        overall_entries = []
        for var_id, total_score in overall_scores.items():
            avg_score = total_score / len(metrics)
            overall_entries.append(RankingEntry(
                variant_id=var_id,
                rank=0,
                score=avg_score,
                metrics={m: np.median(variant_results[var_id].scores.get(m, [0.0])) for m in metrics},
            ))

        reverse = higher_is_better
        overall_entries.sort(key=lambda e: e.score, reverse=reverse)
        self._assign_ranks(overall_entries)

        ties = self._find_ties(overall_entries)

        return RankingResult(
            method=RankingMethod.MEDIAN_SCORE,
            rankings=rankings,
            overall_ranking=overall_entries,
            pareto_front=None,
            ties=ties,
            metadata={"higher_is_better": higher_is_better},
        )

    def _rank_by_borda_count(
        self,
        variant_results: Dict[str, VariantResult],
        metrics: List[str],
        higher_is_better: bool,
    ) -> RankingResult:
        """Rank using Borda count (voting method)."""
        rankings = {}
        borda_scores = {var_id: 0 for var_id in variant_results}

        for metric in metrics:
            entries = []
            for var_id, result in variant_results.items():
                score = result.mean_scores.get(metric, 0.0)
                entries.append((var_id, score))

            # Sort by score
            reverse = higher_is_better
            entries.sort(key=lambda x: x[1], reverse=reverse)

            # Award Borda points (n-1 for first, n-2 for second, etc.)
            n = len(entries)
            for rank, (var_id, _) in enumerate(entries):
                points = n - rank - 1
                borda_scores[var_id] += points

        # Overall ranking by Borda score
        overall_entries = []
        for var_id, total_score in borda_scores.items():
            overall_entries.append(RankingEntry(
                variant_id=var_id,
                rank=0,
                score=total_score,
                metrics={},
            ))

        overall_entries.sort(key=lambda e: e.score, reverse=True)
        self._assign_ranks(overall_entries)

        ties = self._find_ties(overall_entries)

        return RankingResult(
            method=RankingMethod.BORDA_COUNT,
            rankings={},  # Borda doesn't have per-metric rankings
            overall_ranking=overall_entries,
            pareto_front=None,
            ties=ties,
            metadata={"higher_is_better": higher_is_better},
        )

    def _rank_by_copeland(
        self,
        variant_results: Dict[str, VariantResult],
        metrics: List[str],
        higher_is_better: bool,
    ) -> RankingResult:
        """Rank using Copeland method (pairwise comparison)."""
        variant_ids = list(variant_results.keys())
        n = len(variant_ids)

        # Initialize pairwise wins
        wins = {var_id: 0 for var_id in variant_ids}

        # Compare each pair
        for i, var1 in enumerate(variant_ids):
            for var2 in variant_ids[i + 1:]:
                # Count metrics where var1 beats var2
                var1_wins = 0
                var2_wins = 0

                for metric in metrics:
                    score1 = variant_results[var1].mean_scores.get(metric, 0.0)
                    score2 = variant_results[var2].mean_scores.get(metric, 0.0)

                    if higher_is_better:
                        if score1 > score2:
                            var1_wins += 1
                        elif score2 > score1:
                            var2_wins += 1
                    else:
                        if score1 < score2:
                            var1_wins += 1
                        elif score2 < score1:
                            var2_wins += 1

                # Award Copeland points
                if var1_wins > var2_wins:
                    wins[var1] += 1
                elif var2_wins > var1_wins:
                    wins[var2] += 1
                # Tie = no points

        # Overall ranking by Copeland score
        overall_entries = []
        for var_id, score in wins.items():
            overall_entries.append(RankingEntry(
                variant_id=var_id,
                rank=0,
                score=score,
                metrics={},
            ))

        overall_entries.sort(key=lambda e: e.score, reverse=True)
        self._assign_ranks(overall_entries)

        ties = self._find_ties(overall_entries)

        return RankingResult(
            method=RankingMethod.COPELAND,
            rankings={},
            overall_ranking=overall_entries,
            pareto_front=None,
            ties=ties,
            metadata={"higher_is_better": higher_is_better},
        )

    def _rank_by_aggregation(
        self,
        variant_results: Dict[str, VariantResult],
        metrics: List[str],
        higher_is_better: bool,
    ) -> RankingResult:
        """Rank by aggregating ranks across metrics."""
        rankings = {}
        rank_sums = {var_id: 0 for var_id in variant_results}

        for metric in metrics:
            entries = []
            for var_id, result in variant_results.items():
                score = result.mean_scores.get(metric, 0.0)
                entries.append(RankingEntry(
                    variant_id=var_id,
                    rank=0,
                    score=score,
                    metrics={metric: score},
                ))

            reverse = higher_is_better
            entries.sort(key=lambda e: e.score, reverse=reverse)
            self._assign_ranks(entries)

            rankings[metric] = entries

            # Sum ranks
            for entry in entries:
                rank_sums[entry.variant_id] += entry.rank

        # Overall ranking by average rank
        overall_entries = []
        for var_id, rank_sum in rank_sums.items():
            avg_rank = rank_sum / len(metrics)
            overall_entries.append(RankingEntry(
                variant_id=var_id,
                rank=0,
                score=-avg_rank,  # Negative because lower rank sum is better
                metrics={},
            ))

        overall_entries.sort(key=lambda e: e.score, reverse=True)
        self._assign_ranks(overall_entries)

        ties = self._find_ties(overall_entries)

        return RankingResult(
            method=RankingMethod.RANK_AGGREGATION,
            rankings=rankings,
            overall_ranking=overall_entries,
            pareto_front=None,
            ties=ties,
            metadata={"higher_is_better": higher_is_better},
        )

    def _rank_by_pareto(
        self,
        variant_results: Dict[str, VariantResult],
        metrics: List[str],
    ) -> RankingResult:
        """Rank using Pareto dominance."""
        # Find Pareto fronts
        variant_ids = list(variant_results.keys())
        remaining = set(variant_ids)
        fronts = []

        while remaining:
            # Find non-dominated variants in remaining set
            current_front = []
            for var1 in list(remaining):
                dominated = False
                for var2 in remaining:
                    if var1 == var2:
                        continue

                    # Check if var2 dominates var1
                    var2_dominates = True
                    for metric in metrics:
                        score1 = variant_results[var1].mean_scores.get(metric, 0.0)
                        score2 = variant_results[var2].mean_scores.get(metric, 0.0)

                        if score2 < score1:
                            var2_dominates = False
                            break

                    if var2_dominates:
                        dominated = True
                        break

                if not dominated:
                    current_front.append(var1)

            fronts.append(current_front)
            remaining -= set(current_front)

        # Create ranking entries (frontier gets rank 1)
        overall_entries = []
        for front_rank, front in enumerate(fronts):
            for var_id in front:
                overall_entries.append(RankingEntry(
                    variant_id=var_id,
                    rank=front_rank + 1,
                    score=-(front_rank + 1),  # Negative for sorting
                    metrics={m: variant_results[var_id].mean_scores.get(m, 0.0) for m in metrics},
                ))

        # Sort by front
        overall_entries.sort(key=lambda e: e.rank)

        pareto_front = ParetoFront(
            frontier_variants=fronts[0] if fronts else [],
            dominated_variants=list(set(variant_ids) - set(fronts[0])) if fronts else [],
            fronts=fronts,
            criteria=metrics,
        )

        ties = [set(front) for front in fronts]

        return RankingResult(
            method=RankingMethod.PARETO,
            rankings={},
            overall_ranking=overall_entries,
            pareto_front=pareto_front,
            ties=ties,
            metadata={"criteria": metrics},
        )

    def _rank_by_topsis(
        self,
        variant_results: Dict[str, VariantResult],
        metrics: List[str],
        higher_is_better: bool,
    ) -> RankingResult:
        """Rank using TOPSIS method."""
        # Create decision matrix
        variant_ids = list(variant_results.keys())
        n_variants = len(variant_ids)
        n_metrics = len(metrics)

        matrix = np.zeros((n_variants, n_metrics))
        for i, var_id in enumerate(variant_ids):
            for j, metric in enumerate(metrics):
                matrix[i, j] = variant_results[var_id].mean_scores.get(metric, 0.0)

        # Normalize matrix
        norm_matrix = matrix.copy()
        for j in range(n_metrics):
            col_sum = np.sqrt(np.sum(matrix[:, j] ** 2))
            if col_sum > 0:
                norm_matrix[:, j] /= col_sum

        # Calculate ideal and negative ideal solutions
        weights = np.ones(n_metrics) / n_metrics  # Equal weights

        if higher_is_better:
            ideal_best = np.max(norm_matrix, axis=0)
            ideal_worst = np.min(norm_matrix, axis=0)
        else:
            ideal_best = np.min(norm_matrix, axis=0)
            ideal_worst = np.max(norm_matrix, axis=0)

        # Calculate distances
        dist_best = np.zeros(n_variants)
        dist_worst = np.zeros(n_variants)

        for i in range(n_variants):
            dist_best[i] = np.sqrt(np.sum(weights * (norm_matrix[i] - ideal_best) ** 2))
            dist_worst[i] = np.sqrt(np.sum(weights * (norm_matrix[i] - ideal_worst) ** 2))

        # Calculate relative closeness
        closeness = dist_worst / (dist_best + dist_worst)

        # Create ranking entries
        overall_entries = []
        for i, var_id in enumerate(variant_ids):
            overall_entries.append(RankingEntry(
                variant_id=var_id,
                rank=0,
                score=closeness[i],
                metrics={m: variant_results[var_id].mean_scores.get(m, 0.0) for m in metrics},
            ))

        overall_entries.sort(key=lambda e: e.score, reverse=True)
        self._assign_ranks(overall_entries)

        ties = self._find_ties(overall_entries)

        return RankingResult(
            method=RankingMethod.TOPSIS,
            rankings={},
            overall_ranking=overall_entries,
            pareto_front=None,
            ties=ties,
            metadata={"higher_is_better": higher_is_better},
        )

    def _rank_by_condorcet(
        self,
        variant_results: Dict[str, VariantResult],
        metrics: List[str],
        higher_is_better: bool,
    ) -> RankingResult:
        """Rank using Condorcet method."""
        # Same as Copeland for now
        return self._rank_by_copeland(variant_results, metrics, higher_is_better)

    def _assign_ranks(self, entries: List[RankingEntry]) -> None:
        """Assign ranks to entries, handling ties."""
        if not entries:
            return

        current_rank = 1
        i = 0

        while i < len(entries):
            j = i
            # Find all entries with same score
            while j < len(entries) and entries[j].score == entries[i].score:
                j += 1

            # Assign same rank to tied entries
            for k in range(i, j):
                entries[k].rank = current_rank

            # Move to next rank
            current_rank += (j - i)
            i = j

    def _find_ties(self, entries: List[RankingEntry]) -> List[Set[str]]:
        """Find groups of tied variants."""
        ties = []
        i = 0

        while i < len(entries):
            tie_group = {entries[i].variant_id}
            j = i + 1

            while j < len(entries) and entries[j].rank == entries[i].rank:
                if entries[j].score == entries[i].score:
                    tie_group.add(entries[j].variant_id)
                j += 1

            if len(tie_group) > 1:
                ties.append(tie_group)

            i = j

        return ties


# ============================================================================
# Convenience Functions
# ============================================================================

def create_prompt_ranker() -> PromptRanker:
    """Factory function to create a PromptRanker."""
    return PromptRanker()


# Export
__all__ = [
    # Enums
    "RankingMethod",
    # Data classes
    "RankingEntry",
    "ParetoFront",
    "RankingResult",
    # Main class
    "PromptRanker",
    "create_prompt_ranker",
]
