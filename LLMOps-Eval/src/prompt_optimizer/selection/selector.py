"""
Best prompt selection module.

This module provides intelligent selection of the best prompt variant
based on multi-metric criteria, weighted scoring, confidence assessment,
and human-readable explanations.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import numpy as np


@dataclass
class SelectionCriteria:
    """Criteria for prompt selection."""
    metric_weights: Dict[str, float]  # e.g., {"accuracy": 0.5, "latency": 0.3, "cost": 0.2}
    min_confidence: float = 0.95
    min_sample_size: int = 30
    prefer_simpler: bool = True  # Prefer simpler prompts when tied


@dataclass
class SelectionResult:
    """Result of prompt selection."""
    selected_variant_id: str
    selected_prompt: str
    confidence_score: float
    weighted_score: float
    metric_scores: Dict[str, float]
    comparison_to_baseline: Dict[str, float]
    explanation: str
    runner_up_variants: List[Dict[str, Any]]


class BestPromptSelector:
    """Select the best prompt based on experiment results."""

    def __init__(self, analyzer: 'StatisticalAnalyzer'):
        self.analyzer = analyzer

    def select_best(
        self,
        experiment: 'Experiment',
        criteria: SelectionCriteria
    ) -> SelectionResult:
        """
        Select the best prompt from experiment results.

        Args:
            experiment: Completed experiment
            criteria: Selection criteria

        Returns:
            Selection result with explanation
        """
        # Analyze each metric
        metric_analyses = {}
        for metric in experiment.metrics:
            if metric in criteria.metric_weights:
                analysis = self.analyzer.analyze_experiment(experiment, metric)
                metric_analyses[metric] = analysis

        # Calculate weighted scores for each variant
        variant_scores = self._calculate_weighted_scores(
            experiment, metric_analyses, criteria
        )

        # Sort by weighted score
        sorted_variants = sorted(
            variant_scores.items(),
            key=lambda x: x[1]["weighted_score"],
            reverse=True
        )

        # Get best variant
        best_id, best_data = sorted_variants[0]

        # Check confidence
        confidence = self._calculate_confidence(
            best_data, sorted_variants[1] if len(sorted_variants) > 1 else None
        )

        # Find the prompt content
        best_variant = None
        for variant in experiment.variants:
            if variant.id == best_id:
                best_variant = variant
                break

        # Generate explanation
        explanation = self._generate_explanation(
            best_id, best_data, sorted_variants, criteria
        )

        # Get runner-ups
        runner_ups = [
            {
                "variant_id": vid,
                "weighted_score": data["weighted_score"],
                "metric_scores": data["metric_scores"]
            }
            for vid, data in sorted_variants[1:4]  # Top 3 runner-ups
        ]

        return SelectionResult(
            selected_variant_id=best_id,
            selected_prompt=best_variant.prompt_variation.prompt_content if best_variant else "",
            confidence_score=confidence,
            weighted_score=best_data["weighted_score"],
            metric_scores=best_data["metric_scores"],
            comparison_to_baseline=best_data.get("vs_baseline", {}),
            explanation=explanation,
            runner_up_variants=runner_ups
        )

    def _calculate_weighted_scores(
        self,
        experiment: 'Experiment',
        metric_analyses: Dict[str, 'ExperimentAnalysis'],
        criteria: SelectionCriteria
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate weighted scores for all variants."""
        scores = {}

        # Get all variant IDs
        variant_ids = [v.id for v in experiment.variants]

        for variant_id in variant_ids:
            metric_scores = {}

            for metric, analysis in metric_analyses.items():
                # Get the score for this variant on this metric
                if variant_id == analysis.control_variant_id:
                    # Control variant - baseline score
                    control_values = self._get_variant_metric_values(
                        experiment.results, variant_id, metric
                    )
                    metric_scores[metric] = np.mean(control_values) if len(control_values) > 0 else 0
                elif variant_id in analysis.treatment_results:
                    # Treatment variant - get improvement
                    result = analysis.treatment_results[variant_id]
                    control_values = self._get_variant_metric_values(
                        experiment.results, analysis.control_variant_id, metric
                    )
                    treatment_values = self._get_variant_metric_values(
                        experiment.results, variant_id, metric
                    )
                    metric_scores[metric] = np.mean(treatment_values) if len(treatment_values) > 0 else 0

            # Normalize scores (0-1 scale)
            normalized = self._normalize_scores(metric_scores, experiment, criteria)

            # Calculate weighted score
            weighted = sum(
                normalized.get(m, 0) * w
                for m, w in criteria.metric_weights.items()
            )

            scores[variant_id] = {
                "metric_scores": metric_scores,
                "normalized_scores": normalized,
                "weighted_score": weighted
            }

        # Add baseline comparison
        control_id = next(
            (v.id for v in experiment.variants if v.is_control), None
        )
        if control_id and control_id in scores:
            baseline_score = scores[control_id]["weighted_score"]
            for variant_id in scores:
                scores[variant_id]["vs_baseline"] = {
                    "absolute": scores[variant_id]["weighted_score"] - baseline_score,
                    "relative": (
                        (scores[variant_id]["weighted_score"] - baseline_score) / baseline_score
                        if baseline_score != 0 else 0
                    )
                }

        return scores

    def _get_variant_metric_values(
        self,
        results: List['ExperimentResult'],
        variant_id: str,
        metric: str
    ) -> List[float]:
        """Get all metric values for a variant."""
        values = []
        for result in results:
            if result.variant_id == variant_id and metric in result.metrics:
                values.append(result.metrics[metric])
        return values

    def _normalize_scores(
        self,
        scores: Dict[str, float],
        experiment: 'Experiment',
        criteria: SelectionCriteria
    ) -> Dict[str, float]:
        """Normalize scores to 0-1 scale."""
        # Get min/max across all variants for each metric
        all_scores = {metric: [] for metric in scores}

        for result in experiment.results:
            for metric, value in result.metrics.items():
                if metric in all_scores:
                    all_scores[metric].append(value)

        normalized = {}
        for metric, value in scores.items():
            if metric in all_scores and all_scores[metric]:
                min_val = min(all_scores[metric])
                max_val = max(all_scores[metric])
                if max_val > min_val:
                    # Handle metrics where lower is better (latency, cost)
                    if metric in ["latency", "cost", "latency_ms", "cost_usd"]:
                        normalized[metric] = 1 - (value - min_val) / (max_val - min_val)
                    else:
                        normalized[metric] = (value - min_val) / (max_val - min_val)
                else:
                    normalized[metric] = 1.0
            else:
                normalized[metric] = 0.5

        return normalized

    def _calculate_confidence(
        self,
        best_data: Dict[str, Any],
        second_best: Optional[tuple]
    ) -> float:
        """Calculate confidence in the selection."""
        if not second_best:
            return 1.0

        second_best_id, second_data = second_best

        # Confidence based on score gap
        gap = best_data["weighted_score"] - second_data["weighted_score"]

        # Sigmoid-like transformation
        confidence = 1 / (1 + np.exp(-10 * gap))

        return float(confidence)

    def _generate_explanation(
        self,
        best_id: str,
        best_data: Dict[str, Any],
        all_variants: List[tuple],
        criteria: SelectionCriteria
    ) -> str:
        """Generate human-readable explanation for selection."""
        lines = [f"Selected variant: {best_id}"]
        lines.append(f"Weighted score: {best_data['weighted_score']:.3f}"]

        lines.append("\nMetric breakdown:")
        for metric, score in best_data["metric_scores"].items():
            weight = criteria.metric_weights.get(metric, 0)
            lines.append(f"  - {metric}: {score:.3f} (weight: {weight})")

        if "vs_baseline" in best_data:
            baseline = best_data["vs_baseline"]
            lines.append(f"\nVs baseline: {baseline['relative']*100:+.1f}%")

        if len(all_variants) > 1:
            second_id, second_data = all_variants[1]
            gap = best_data["weighted_score"] - second_data["weighted_score"]
            lines.append(f"\nMargin over runner-up ({second_id}): {gap:.3f}")

        return "\n".join(lines)


# Extended selector with additional methods

class AdvancedPromptSelector(BestPromptSelector):
    """Extended prompt selector with advanced selection strategies."""

    def __init__(self, analyzer: 'StatisticalAnalyzer'):
        super().__init__(analyzer)

    def select_with_risk_analysis(
        self,
        experiment: 'Experiment',
        criteria: SelectionCriteria,
        risk_tolerance: str = "medium"  # "low", "medium", "high"
    ) -> SelectionResult:
        """
        Select best prompt with risk consideration.

        Args:
            experiment: Completed experiment
            criteria: Selection criteria
            risk_tolerance: Risk tolerance level

        Returns:
            Selection result with risk-adjusted scoring
        """
        # Get base scores
        variant_scores = self._calculate_weighted_scores(
            experiment, {}, criteria  # Simplified for risk analysis
        )

        # Calculate risk scores (variance-based)
        risk_scores = {}
        for variant_id in variant_scores:
            metric_variance = {}
            for metric in criteria.metric_weights:
                values = self._get_variant_metric_values(
                    experiment.results, variant_id, metric
                )
                if values:
                    metric_variance[metric] = np.var(values)

            # Risk penalty based on variance
            risk_penalty = sum(
                metric_variance.get(m, 0) * w
                for m, w in criteria.metric_weights.items()
            )

            # Adjust scores based on risk tolerance
            if risk_tolerance == "low":
                risk_multiplier = 2.0
            elif risk_tolerance == "high":
                risk_multiplier = 0.5
            else:  # medium
                risk_multiplier = 1.0

            risk_scores[variant_id] = risk_penalty * risk_multiplier

        # Combine score and risk
        adjusted_scores = {}
        for variant_id in variant_scores:
            base_score = variant_scores[variant_id]["weighted_score"]
            risk_penalty = risk_scores.get(variant_id, 0)
            adjusted_scores[variant_id] = base_score - risk_penalty

        # Select best adjusted score
        best_id = max(adjusted_scores, key=adjusted_scores.get)

        # Create modified selection result
        result = self.select_best(experiment, criteria)

        # Add risk information to explanation
        result.explanation += f"\n\nRisk-adjusted selection (tolerance: {risk_tolerance})"
        result.explanation += f"\nRisk penalty applied: {risk_scores.get(best_id, 0):.3f}"

        return result

    def select_pareto_optimal(
        self,
        experiment: 'Experiment',
        criteria: SelectionCriteria
    ) -> List[SelectionResult]:
        """
        Find all Pareto-optimal prompts.

        Returns non-dominated variants where no other variant
        is better on all metrics.

        Args:
            experiment: Completed experiment
            criteria: Selection criteria

        Returns:
            List of Pareto-optimal selection results
        """
        # Get scores for all variants
        variant_scores = self._calculate_weighted_scores(
            experiment, {}, criteria
        )

        # Find Pareto frontier
        pareto_variants = []
        variant_ids = list(variant_scores.keys())

        for variant_id in variant_ids:
            is_dominated = False
            for other_id in variant_ids:
                if variant_id == other_id:
                    continue

                # Check if other_id dominates variant_id
                dominates = True
                for metric in criteria.metric_weights:
                    var_score = variant_scores[variant_id]["metric_scores"].get(metric, 0)
                    other_score = variant_scores[other_id]["metric_scores"].get(metric, 0)

                    # For metrics where lower is better
                    if metric in ["latency", "cost", "latency_ms", "cost_usd"]:
                        if other_score > var_score:
                            dominates = False
                            break
                    else:
                        if other_score < var_score:
                            dominates = False
                            break

                if dominates:
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_variants.append(variant_id)

        # Create selection results for all Pareto-optimal variants
        results = []
        for variant_id in pareto_variants:
            # Create a modified criteria that selects this variant
            modified_criteria = SelectionCriteria(
                metric_weights=criteria.metric_weights.copy(),
                min_confidence=criteria.min_confidence,
                min_sample_size=criteria.min_sample_size,
                prefer_simpler=criteria.prefer_simpler
            )

            result = self.select_best(experiment, modified_criteria)
            result.explanation = f"Pareto-optimal variant: {variant_id}\n" + result.explanation
            results.append(result)

        return results

    def compare_variants(
        self,
        experiment: 'Experiment',
        variant_ids: List[str],
        criteria: SelectionCriteria
    ) -> Dict[str, Any]:
        """
        Compare specific variants in detail.

        Args:
            experiment: Completed experiment
            variant_ids: Variants to compare
            criteria: Selection criteria

        Returns:
            Detailed comparison
        """
        comparison = {
            "variants": {},
            "metrics": {},
            "rankings": {},
        }

        # Get scores for each variant
        for variant_id in variant_ids:
            metric_scores = {}
            for metric in experiment.metrics:
                values = self._get_variant_metric_values(
                    experiment.results, variant_id, metric
                )
                if values:
                    metric_scores[metric] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "count": len(values),
                    }

            comparison["variants"][variant_id] = metric_scores

        # Compare pairwise
        for i, v1 in enumerate(variant_ids):
            for v2 in variant_ids[i + 1:]:
                pair_key = f"{v1}_vs_{v2}"
                comparison["metrics"][pair_key] = {}

                for metric in experiment.metrics:
                    if metric in comparison["variants"][v1]:
                        score1 = comparison["variants"][v1][metric]["mean"]
                        score2 = comparison["variants"][v2][metric]["mean"]
                        diff = score2 - score1

                        comparison["metrics"][pair_key][metric] = {
                            "difference": diff,
                            "percent_change": (diff / score1 * 100) if score1 != 0 else 0,
                            "direction": "better" if (
                                (metric not in ["latency", "cost", "latency_ms", "cost_usd"] and diff > 0) or
                                (metric in ["latency", "cost", "latency_ms", "cost_usd"] and diff < 0)
                            ) else "worse"
                        }

        # Rank variants
        variant_rankings = []
        for variant_id in variant_ids:
            weighted_score = sum(
                comparison["variants"][variant_id].get(m, {}).get("mean", 0) * w
                for m, w in criteria.metric_weights.items()
            )
            variant_rankings.append((variant_id, weighted_score))

        variant_rankings.sort(key=lambda x: x[1], reverse=True)
        comparison["rankings"] = {
            variant_id: rank + 1
            for rank, (variant_id, _) in enumerate(variant_rankings)
        }

        return comparison


# Export
__all__ = [
    "SelectionCriteria",
    "SelectionResult",
    "BestPromptSelector",
    "AdvancedPromptSelector",
]
