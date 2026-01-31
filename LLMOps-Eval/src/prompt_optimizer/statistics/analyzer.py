"""
Statistical analysis module for experiment results.

This module provides comprehensive statistical analysis for A/B test experiments,
including hypothesis testing, effect size calculation, confidence intervals,
power analysis, and sample size recommendations.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy import stats
from scipy.stats import norm
import warnings


@dataclass
class StatisticalTestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float
    effect_size_interpretation: str
    confidence_interval: Tuple[float, float]
    sample_size_control: int
    sample_size_treatment: int
    power: float
    recommendation: str


@dataclass
class ExperimentAnalysis:
    """Complete analysis of an experiment."""
    experiment_id: str
    control_variant_id: str
    treatment_results: Dict[str, StatisticalTestResult]
    best_variant_id: str
    best_variant_improvement: float
    confidence_level: float
    warnings: List[str]
    recommendations: List[str]


class StatisticalAnalyzer:
    """Statistical analysis for A/B test experiments."""

    def __init__(
        self,
        significance_level: float = 0.05,
        min_effect_size: float = 0.2,
        target_power: float = 0.8
    ):
        self.alpha = significance_level
        self.min_effect_size = min_effect_size
        self.target_power = target_power

    def analyze_experiment(
        self,
        experiment: 'Experiment',
        metric: str
    ) -> ExperimentAnalysis:
        """
        Analyze experiment results for a specific metric.

        Args:
            experiment: Completed experiment
            metric: Which metric to analyze

        Returns:
            Complete analysis with statistical tests
        """
        # Extract control and treatment data
        control_variant = None
        treatment_variants = []

        for variant in experiment.variants:
            if variant.is_control:
                control_variant = variant
            else:
                treatment_variants.append(variant)

        if not control_variant:
            raise ValueError("No control variant found")

        # Get metric values per variant
        control_values = self._get_metric_values(
            experiment.results, control_variant.id, metric
        )

        treatment_results = {}
        warnings_list = []

        # Analyze each treatment vs control
        for treatment in treatment_variants:
            treatment_values = self._get_metric_values(
                experiment.results, treatment.id, metric
            )

            if len(control_values) < 5 or len(treatment_values) < 5:
                warnings_list.append(
                    f"Low sample size for {treatment.id}: "
                    f"control={len(control_values)}, treatment={len(treatment_values)}"
                )

            test_result = self._compare_groups(
                control_values,
                treatment_values,
                treatment.id
            )
            treatment_results[treatment.id] = test_result

        # Apply multiple comparison correction if needed
        if len(treatment_variants) > 1:
            treatment_results = self._apply_bonferroni_correction(
                treatment_results, len(treatment_variants)
            )

        # Find best variant
        best_variant_id, best_improvement = self._find_best_variant(
            control_variant.id, treatment_results, control_values
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            treatment_results, warnings_list
        )

        return ExperimentAnalysis(
            experiment_id=experiment.id,
            control_variant_id=control_variant.id,
            treatment_results=treatment_results,
            best_variant_id=best_variant_id,
            best_variant_improvement=best_improvement,
            confidence_level=1 - self.alpha,
            warnings=warnings_list,
            recommendations=recommendations
        )

    def _get_metric_values(
        self,
        results: List['ExperimentResult'],
        variant_id: str,
        metric: str
    ) -> np.ndarray:
        """Extract metric values for a specific variant."""
        values = []
        for result in results:
            if result.variant_id == variant_id and metric in result.metrics:
                values.append(result.metrics[metric])
        return np.array(values)

    def _compare_groups(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        treatment_id: str
    ) -> StatisticalTestResult:
        """Compare control and treatment groups statistically."""

        # Check normality
        _, control_normal_p = stats.shapiro(control) if len(control) >= 3 else (0, 0)
        _, treatment_normal_p = stats.shapiro(treatment) if len(treatment) >= 3 else (0, 0)

        is_normal = control_normal_p > 0.05 and treatment_normal_p > 0.05

        # Choose appropriate test
        if is_normal:
            # Welch's t-test (doesn't assume equal variance)
            statistic, p_value = stats.ttest_ind(
                treatment, control, equal_var=False
            )
            test_name = "Welch's t-test"
        else:
            # Mann-Whitney U test (non-parametric)
            statistic, p_value = stats.mannwhitneyu(
                treatment, control, alternative='two-sided'
            )
            test_name = "Mann-Whitney U"

        # Effect size (Cohen's d)
        effect_size = self._cohens_d(control, treatment)
        effect_interpretation = self._interpret_effect_size(effect_size)

        # Confidence interval for difference in means
        ci = self._confidence_interval(control, treatment)

        # Statistical power
        power = self._calculate_power(
            len(control), len(treatment), effect_size
        )

        # Determine significance
        significant = p_value < self.alpha

        # Generate recommendation
        if significant and abs(effect_size) >= self.min_effect_size:
            if np.mean(treatment) > np.mean(control):
                recommendation = f"Treatment {treatment_id} significantly outperforms control"
            else:
                recommendation = f"Control outperforms treatment {treatment_id}"
        elif not significant:
            recommendation = "No significant difference detected. Consider increasing sample size."
        else:
            recommendation = "Significant but small effect size. Practical significance unclear."

        return StatisticalTestResult(
            test_name=test_name,
            statistic=float(statistic),
            p_value=float(p_value),
            significant=significant,
            effect_size=float(effect_size),
            effect_size_interpretation=effect_interpretation,
            confidence_interval=ci,
            sample_size_control=len(control),
            sample_size_treatment=len(treatment),
            power=power,
            recommendation=recommendation
        )

    def _cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (np.mean(group2) - np.mean(group1)) / pooled_std

    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        d_abs = abs(d)
        if d_abs < 0.2:
            return "negligible"
        elif d_abs < 0.5:
            return "small"
        elif d_abs < 0.8:
            return "medium"
        else:
            return "large"

    def _confidence_interval(
        self,
        group1: np.ndarray,
        group2: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate confidence interval for difference in means."""
        mean_diff = np.mean(group2) - np.mean(group1)

        # Standard error of the difference
        se1 = np.std(group1, ddof=1) / np.sqrt(len(group1))
        se2 = np.std(group2, ddof=1) / np.sqrt(len(group2))
        se_diff = np.sqrt(se1**2 + se2**2)

        # CI using normal approximation
        z = norm.ppf(1 - self.alpha / 2)
        ci_lower = mean_diff - z * se_diff
        ci_upper = mean_diff + z * se_diff

        return (float(ci_lower), float(ci_upper))

    def _calculate_power(
        self,
        n1: int,
        n2: int,
        effect_size: float
    ) -> float:
        """Calculate statistical power."""
        if effect_size == 0:
            return self.alpha  # Power equals alpha when no effect

        # Harmonic mean of sample sizes
        n_harmonic = 2 * n1 * n2 / (n1 + n2) if (n1 + n2) > 0 else 0

        # Non-centrality parameter
        ncp = abs(effect_size) * np.sqrt(n_harmonic / 2)

        # Critical value
        critical_value = norm.ppf(1 - self.alpha / 2)

        # Power calculation
        power = 1 - norm.cdf(critical_value - ncp) + norm.cdf(-critical_value - ncp)

        return float(min(power, 1.0))

    def calculate_required_sample_size(
        self,
        effect_size: float,
        power: float = 0.8,
        alpha: float = 0.05
    ) -> int:
        """
        Calculate required sample size per group.

        Args:
            effect_size: Expected Cohen's d
            power: Desired statistical power
            alpha: Significance level

        Returns:
            Required sample size per group
        """
        if effect_size == 0:
            return float('inf')

        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)

        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

        return int(np.ceil(n))

    def _apply_bonferroni_correction(
        self,
        results: Dict[str, StatisticalTestResult],
        num_comparisons: int
    ) -> Dict[str, StatisticalTestResult]:
        """Apply Bonferroni correction for multiple comparisons."""
        corrected_alpha = self.alpha / num_comparisons

        for variant_id, result in results.items():
            result.significant = result.p_value < corrected_alpha
            if result.significant:
                result.recommendation = (
                    f"{result.recommendation} (after Bonferroni correction)"
                )
            else:
                result.recommendation = (
                    f"Not significant after Bonferroni correction "
                    f"(corrected α = {corrected_alpha:.4f})"
                )

        return results

    def _find_best_variant(
        self,
        control_id: str,
        treatment_results: Dict[str, StatisticalTestResult],
        control_values: np.ndarray
    ) -> Tuple[str, float]:
        """Find the best performing variant."""
        control_mean = np.mean(control_values)

        best_id = control_id
        best_improvement = 0.0

        for variant_id, result in treatment_results.items():
            if result.significant and result.effect_size > 0:
                improvement = result.effect_size
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_id = variant_id

        return best_id, best_improvement

    def _generate_recommendations(
        self,
        results: Dict[str, StatisticalTestResult],
        warnings: List[str]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Check for low power
        low_power = [
            (vid, r) for vid, r in results.items()
            if r.power < self.target_power
        ]
        if low_power:
            min_power = min(r.power for _, r in low_power)
            recommended_n = self.calculate_required_sample_size(
                self.min_effect_size, self.target_power, self.alpha
            )
            recommendations.append(
                f"Low statistical power ({min_power:.2f}). "
                f"Recommend at least {recommended_n} samples per variant."
            )

        # Check for significant results
        significant = [
            (vid, r) for vid, r in results.items() if r.significant
        ]
        if significant:
            for vid, r in significant:
                if r.effect_size > 0:
                    recommendations.append(
                        f"Consider adopting {vid} - "
                        f"{r.effect_size_interpretation} positive effect detected."
                    )
        else:
            recommendations.append(
                "No significant improvements found. Consider testing different "
                "variation strategies or accepting the current prompt."
            )

        return recommendations


# Additional statistical methods for comprehensive analysis

class AdvancedStatisticalAnalyzer(StatisticalAnalyzer):
    """Extended statistical analyzer with additional methods."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def equivalence_test(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        margin: float = 0.2
    ) -> Dict[str, Any]:
        """
        Test for equivalence (two one-sided tests).

        Args:
            control: Control group values
            treatment: Treatment group values
            margin: Equivalence margin (effect size)

        Returns:
            Dictionary with test results
        """
        diff = np.mean(treatment) - np.mean(control)
        n1, n2 = len(control), len(treatment)

        # Standard error
        se = np.sqrt(
            np.var(control, ddof=1) / n1 +
            np.var(treatment, ddof=1) / n2
        )

        # TOST statistics
        t1 = (diff - margin) / se
        t2 = (diff + margin) / se

        # P-values (one-sided)
        df = n1 + n2 - 2
        p1 = 1 - stats.t.cdf(t1, df)
        p2 = stats.t.cdf(t2, df)

        # Overall p-value
        p_value = max(p1, p2)

        # Is equivalent?
        is_equivalent = p_value < self.alpha and abs(diff) < margin

        return {
            "is_equivalent": is_equivalent,
            "p_value": p_value,
            "difference": diff,
            "margin": margin,
            "tost_p1": p1,
            "tost_p2": p2,
        }

    def non_inferiority_test(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        margin: float = 0.2
    ) -> Dict[str, Any]:
        """
        Test for non-inferiority.

        Args:
            control: Control group values
            treatment: Treatment group values
            margin: Non-inferiority margin

        Returns:
            Dictionary with test results
        """
        diff = np.mean(treatment) - np.mean(control)
        n1, n2 = len(control), len(treatment)

        # Standard error
        se = np.sqrt(
            np.var(control, ddof=1) / n1 +
            np.var(treatment, ddof=1) / n2
        )

        # T-statistic
        t_stat = (diff + margin) / se

        # P-value (one-sided)
        df = n1 + n2 - 2
        p_value = 1 - stats.t.cdf(t_stat, df)

        # Is non-inferior?
        is_non_inferior = p_value < self.alpha and diff > -margin

        return {
            "is_non_inferior": is_non_inferior,
            "p_value": p_value,
            "difference": diff,
            "margin": margin,
            "lower_bound": diff - margin,
        }

    def bayesian_analysis(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        num_samples: int = 10000
    ) -> Dict[str, Any]:
        """
        Bayesian analysis using Monte Carlo sampling.

        Args:
            control: Control group values
            treatment: Treatment group values
            num_samples: Number of MC samples

        Returns:
            Dictionary with Bayesian results
        """
        # Estimate posterior parameters (conjugate priors)
        # Using normal-inverse-gamma for conjugate analysis

        n1, n2 = len(control), len(treatment)
        mean1, mean2 = np.mean(control), np.mean(treatment)
        var1, var2 = np.var(control, ddof=1), np.var(treatment, ddof=1)

        # Sample from posterior predictive distributions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            samples1 = np.random.normal(
                mean1, np.sqrt(var1), num_samples
            )
            samples2 = np.random.normal(
                mean2, np.sqrt(var2), num_samples
            )

        # Probability that treatment > control
        prob_treatment_better = np.mean(samples2 > samples1)

        # Credible interval for difference
        diff_samples = samples2 - samples1
        ci_lower = np.percentile(diff_samples, 2.5)
        ci_upper = np.percentile(diff_samples, 97.5)

        # Expected improvement
        expected_improvement = np.mean(diff_samples)

        return {
            "prob_treatment_better": prob_treatment_better,
            "expected_improvement": expected_improvement,
            "credible_interval": (ci_lower, ci_upper),
            "samples_control": samples1,
            "samples_treatment": samples2,
        }

    def anova_test(
        self,
        groups: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        One-way ANOVA for comparing multiple groups.

        Args:
            groups: Dictionary of group names to values

        Returns:
            Dictionary with ANOVA results
        """
        # Prepare data
        group_values = list(groups.values())
        group_names = list(groups.keys())

        # Perform ANOVA
        statistic, p_value = stats.f_oneway(*group_values)

        # Effect size (eta-squared)
        # Between-group variance
        all_values = np.concatenate(group_values)
        grand_mean = np.mean(all_values)

        ss_between = sum(
            len(g) * (np.mean(g) - grand_mean) ** 2
            for g in group_values
        )
        ss_total = sum((v - grand_mean) ** 2 for v in all_values)

        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        # Interpret effect size
        if eta_squared < 0.01:
            effect_interp = "small"
        elif eta_squared < 0.06:
            effect_interp = "medium"
        else:
            effect_interp = "large"

        return {
            "f_statistic": statistic,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "eta_squared": eta_squared,
            "effect_size_interpretation": effect_interp,
            "num_groups": len(groups),
        }

    def pairwise_comparisons(
        self,
        groups: Dict[str, np.ndarray],
        method: str = "tukey"
    ) -> List[Dict[str, Any]]:
        """
        Pairwise comparisons between groups.

        Args:
            groups: Dictionary of group names to values
            method: Correction method ("tukey", "bonferroni", "holm")

        Returns:
            List of comparison results
        """
        from itertools import combinations

        results = []
        group_names = list(groups.keys())

        # All pairwise combinations
        for (name1, name2) in combinations(group_names, 2):
            values1 = groups[name1]
            values2 = groups[name2]

            # T-test
            statistic, p_value = stats.ttest_ind(values1, values2, equal_var=False)

            # Effect size
            effect_size = self._cohens_d(values1, values2)
            effect_interp = self._interpret_effect_size(effect_size)

            results.append({
                "group1": name1,
                "group2": name2,
                "statistic": statistic,
                "p_value": p_value,
                "effect_size": effect_size,
                "effect_size_interpretation": effect_interp,
            })

        # Apply corrections
        num_comparisons = len(results)
        if method == "bonferroni":
            for result in results:
                result["p_value_corrected"] = min(result["p_value"] * num_comparisons, 1.0)
                result["significant"] = result["p_value_corrected"] < self.alpha

        elif method == "holm":
            # Sort by p-value
            sorted_results = sorted(results, key=lambda x: x["p_value"])
            for i, result in enumerate(sorted_results):
                result["p_value_corrected"] = min(
                    result["p_value"] * (num_comparisons - i),
                    1.0
                )
                result["significant"] = result["p_value_corrected"] < self.alpha

        # Tukey requires specialized implementation
        elif method == "tukey":
            # Simplified Tukey HSD
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            try:
                # Prepare data for statsmodels
                data = []
                group_labels = []
                for name, values in groups.items():
                    data.extend(values)
                    group_labels.extend([name] * len(values))

                tukey = pairwise_tukeyhsd(data, group_labels)
                # Map results back
                for i, result in enumerate(results):
                    result.update({
                        "p_value_corrected": tukey.pvalues[i],
                        "significant": tukey.reject[i],
                    })
            except ImportError:
                # Fallback to Bonferroni if statsmodels not available
                for result in results:
                    result["p_value_corrected"] = min(
                        result["p_value"] * num_comparisons, 1.0
                    )
                    result["significant"] = result["p_value_corrected"] < self.alpha

        return results


# Export
__all__ = [
    "StatisticalTestResult",
    "ExperimentAnalysis",
    "StatisticalAnalyzer",
    "AdvancedStatisticalAnalyzer",
]
