"""
FastAPI routes for PromptOptimizer.

This module provides REST API endpoints for prompt optimization,
including experiment management, variation generation, and result analysis.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

from src.prompt_optimizer.experiments.framework import (
    ExperimentFramework,
    ExperimentConfig,
    ExperimentResult,
)
from src.prompt_optimizer.variations.variation_generator import (
    VariationGenerator,
    VariationSet,
    VariationStrategy,
)
from src.prompt_optimizer.selection.selector import (
    PromptSelector,
    SelectionStrategy,
    SelectionResult,
)
from src.prompt_optimizer.selection.ranking import (
    PromptRanker,
    RankingMethod,
    RankingResult,
)
from src.prompt_optimizer.history.history import (
    HistoryManager,
    HistorySortOrder,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/optimizer", tags=["prompt-optimizer"])

# Global instances (in production, use dependency injection)
_experiment_framework: Optional[ExperimentFramework] = None
_variation_generator: Optional[VariationGenerator] = None
_selector: Optional[PromptSelector] = None
_ranker: Optional[PromptRanker] = None
_history: Optional[HistoryManager] = None


# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class ExperimentConfigRequest(BaseModel):
    """Request model for creating an experiment."""

    name: str = Field(..., description="Experiment name")
    description: str = Field(..., description="Experiment description")
    base_template_id: str = Field(..., description="Base template ID")
    dataset_name: str = Field(..., description="Dataset to use")
    strategies: List[str] = Field(..., description="Variation strategies")
    variations_per_strategy: int = Field(3, description="Variations per strategy")
    combine_strategies: bool = Field(False, description="Combine strategies")
    sample_size: Optional[int] = Field(None, description="Sample size")
    metrics: List[str] = Field(
        default=["semantic_similarity", "llm_judge"],
        description="Metrics to evaluate"
    )
    provider: str = Field("openai", description="LLM provider")
    model: str = Field("gpt-4o-mini", description="Model name")
    alpha: float = Field(0.05, description="Significance level")
    paired_design: bool = Field(True, description="Use paired tests")


class VariationRequest(BaseModel):
    """Request model for generating variations."""

    base_prompt: str = Field(..., description="Base prompt to vary")
    strategies: List[str] = Field(..., description="Strategies to use")
    variations_per_strategy: int = Field(3, description="Variations per strategy")
    combine_strategies: bool = Field(False, description="Combine strategies")


class SelectionRequest(BaseModel):
    """Request model for selecting best variant."""

    variant_results: Dict[str, Dict[str, Any]] = Field(..., description="Variant results")
    statistical_tests: Dict[str, Dict[str, Dict[str, Any]]] = Field(
        default={},
        description="Statistical test results"
    )
    strategy: str = Field("statistical_winner", description="Selection strategy")
    primary_metric: str = Field("semantic_similarity", description="Primary metric")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Constraints")


class RankingRequest(BaseModel):
    """Request model for ranking variants."""

    variant_results: Dict[str, Dict[str, Any]] = Field(..., description="Variant results")
    method: str = Field("mean_score", description="Ranking method")
    metrics: Optional[List[str]] = Field(None, description="Metrics to consider")


class ExperimentResponse(BaseModel):
    """Response model for experiment results."""

    experiment_id: str
    status: str
    message: str
    result_url: Optional[str] = None


# ============================================================================
# Dependency Injection
# ============================================================================

def get_experiment_framework() -> ExperimentFramework:
    """Get experiment framework instance."""
    global _experiment_framework
    if _experiment_framework is None:
        from src.prompt_optimizer.experiments.framework import create_experiment_framework
        _experiment_framework = create_experiment_framework()
    return _experiment_framework


def get_variation_generator() -> VariationGenerator:
    """Get variation generator instance."""
    global _variation_generator
    if _variation_generator is None:
        from src.prompt_optimizer.variations.variation_generator import create_variation_generator
        _variation_generator = create_variation_generator()
    return _variation_generator


def get_selector() -> PromptSelector:
    """Get prompt selector instance."""
    global _selector
    if _selector is None:
        from src.prompt_optimizer.selection.selector import create_prompt_selector
        _selector = create_prompt_selector()
    return _selector


def get_ranker() -> PromptRanker:
    """Get prompt ranker instance."""
    global _ranker
    if _ranker is None:
        from src.prompt_optimizer.selection.ranking import create_prompt_ranker
        _ranker = create_prompt_ranker()
    return _ranker


def get_history() -> HistoryManager:
    """Get history manager instance."""
    global _history
    if _history is None:
        from src.prompt_optimizer.history.history import create_history_manager
        _history = create_history_manager()
    return _history


# ============================================================================
# API Routes
# ============================================================================

@router.post("/variations/generate", response_model=Dict[str, Any])
async def generate_variations(
    request: VariationRequest,
    generator: VariationGenerator = Depends(get_variation_generator),
):
    """
    Generate prompt variations.

    Creates systematic variations of a base prompt using specified strategies.
    """
    try:
        # Convert strategy strings to enums
        strategies = []
        for s in request.strategies:
            try:
                strategies.append(VariationStrategy(s))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown strategy: {s}"
                )

        # Generate variations
        variation_set = generator.generate(
            base_prompt=request.base_prompt,
            strategies=strategies,
            variations_per_strategy=request.variations_per_strategy,
            combine_strategies=request.combine_strategies,
        )

        return {
            "variation_set_id": variation_set.id,
            "num_variations": len(variation_set.variations),
            "strategies_used": [s.value for s in variation_set.strategies_used],
            "variations": [
                {
                    "id": v.id,
                    "strategy": v.strategy.value,
                    "description": v.description,
                    "prompt_content": v.prompt_content[:200] + "...",  # Truncate
                }
                for v in variation_set.variations
            ],
        }

    except Exception as e:
        logger.error(f"Error generating variations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/create", response_model=Dict[str, Any])
async def create_experiment(
    request: ExperimentConfigRequest,
    background_tasks: BackgroundTasks,
    framework: ExperimentFramework = Depends(get_experiment_framework),
):
    """
    Create and run a new experiment.

    Runs prompt optimization experiment in the background.
    """
    try:
        # Convert strategies
        strategies = []
        for s in request.strategies:
            try:
                strategies.append(VariationStrategy(s))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown strategy: {s}"
                )

        # Create experiment config
        from src.prompt_optimizer.experiments.ab_test import (
            ABTestConfig,
            TestMethod,
            MultipleComparisonCorrection,
        )

        test_config = ABTestConfig(
            alpha=request.alpha,
            test_method=TestMethod.T_TEST,
            paired_design=request.paired_design,
        )

        config = ExperimentConfig(
            name=request.name,
            description=request.description,
            base_template_id=request.base_template_id,
            dataset_name=request.dataset_name,
            strategies=strategies,
            variations_per_strategy=request.variations_per_strategy,
            combine_strategies=request.combine_strategies,
            sample_size=request.sample_size,
            metrics=request.metrics,
            provider=request.provider,
            model=request.model,
            test_config=test_config,
        )

        # Generate experiment ID
        import uuid
        experiment_id = str(uuid.uuid4())[:8]

        # Start experiment in background
        # In production, you'd use a proper task queue
        background_tasks.add_task(
            _run_experiment_async,
            framework,
            config,
            experiment_id,
        )

        return {
            "experiment_id": experiment_id,
            "status": "pending",
            "message": "Experiment started",
        }

    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _run_experiment_async(
    framework: ExperimentFramework,
    config: ExperimentConfig,
    experiment_id: str,
):
    """Run experiment in background."""
    try:
        # Get base prompt from template
        # For now, use a placeholder
        base_prompt = "Summarize the following text:"

        result = await framework.run_experiment(
            config=config,
            base_prompt=base_prompt,
        )

        # Save to history
        history_mgr = get_history()
        result_path = result.save()
        history_mgr.add_experiment(result, result_path)

        logger.info(f"Experiment {experiment_id} completed")

    except Exception as e:
        logger.error(f"Experiment {experiment_id} failed: {e}")


@router.get("/experiments/{experiment_id}")
async def get_experiment(
    experiment_id: str,
    history: HistoryManager = Depends(get_history),
):
    """Get experiment result by ID."""
    result = history.get_experiment(experiment_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return result.to_dict()


@router.get("/experiments")
async def list_experiments(
    limit: int = 100,
    sort: str = "newest_first",
    status: Optional[str] = None,
    tags: Optional[str] = None,
    history: HistoryManager = Depends(get_history),
):
    """List experiments with optional filtering."""
    # Parse sort order
    try:
        sort_order = HistorySortOrder(sort)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown sort order: {sort}")

    # Build filters
    filters = {}
    if status:
        filters["status"] = status
    if tags:
        filters["tags"] = tags.split(",")

    # Get experiments
    experiments = history.list_experiments(
        limit=limit,
        sort_by=sort_order,
        filters=filters if filters else None,
    )

    return {
        "experiments": [e.to_dict() for e in experiments],
        "total": len(experiments),
    }


@router.post("/selection/select", response_model=Dict[str, Any])
async def select_variant(
    request: SelectionRequest,
    selector: PromptSelector = Depends(get_selector),
):
    """Select best variant using specified strategy."""
    try:
        # Convert strategy
        try:
            strategy = SelectionStrategy(request.strategy)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown selection strategy: {request.strategy}"
            )

        # Convert variant results to proper format
        from src.prompt_optimizer.experiments.framework import VariantResult

        variant_results = {}
        for var_id, data in request.variant_results.items():
            variant_results[var_id] = VariantResult(
                variant_id=var_id,
                variant_name=data.get("variant_name", var_id),
                strategy=VariationStrategy(data.get("strategy", "instruction_rephrase")),
                prompt_content=data.get("prompt_content", ""),
                system_prompt=data.get("system_prompt"),
                scores=data.get("scores", {}),
                mean_scores=data.get("mean_scores", {}),
                std_scores=data.get("std_scores", {}),
                evaluation_time=data.get("evaluation_time", 0.0),
                token_usage=data.get("token_usage", {}),
                cost=data.get("cost", 0.0),
                metadata=data.get("metadata", {}),
            )

        # Convert statistical tests
        from src.prompt_optimizer.experiments.ab_test import TestResult

        statistical_tests = {}
        for metric, variants in request.statistical_tests.items():
            statistical_tests[metric] = {}
            for var_id, test_data in variants.items():
                statistical_tests[metric][var_id] = TestResult(
                    test_name=test_data.get("test_name", ""),
                    statistic=test_data.get("statistic", 0.0),
                    p_value=test_data.get("p_value", 1.0),
                    is_significant=test_data.get("is_significant", False),
                    effect_size=test_data.get("effect_size", 0.0),
                    effect_size_magnitude=test_data.get("effect_size_magnitude", ""),
                    confidence_interval=test_data.get("confidence_interval", (0.0, 0.0)),
                    control_mean=test_data.get("control_mean", 0.0),
                    treatment_mean=test_data.get("treatment_mean", 0.0),
                    control_std=test_data.get("control_std", 0.0),
                    treatment_std=test_data.get("treatment_std", 0.0),
                    sample_size=test_data.get("sample_size", 0),
                )

        # Run selection
        result = selector.select(
            variant_results=variant_results,
            statistical_tests=statistical_tests,
            strategy=strategy,
            primary_metric=request.primary_metric,
            constraints=request.constraints,
        )

        return result.to_dict()

    except Exception as e:
        logger.error(f"Error selecting variant: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ranking/rank", response_model=Dict[str, Any])
async def rank_variants(
    request: RankingRequest,
    ranker: PromptRanker = Depends(get_ranker),
):
    """Rank variants using specified method."""
    try:
        # Convert method
        try:
            method = RankingMethod(request.method)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown ranking method: {request.method}"
            )

        # Convert variant results
        from src.prompt_optimizer.experiments.framework import VariantResult

        variant_results = {}
        for var_id, data in request.variant_results.items():
            variant_results[var_id] = VariantResult(
                variant_id=var_id,
                variant_name=data.get("variant_name", var_id),
                strategy=VariationStrategy(data.get("strategy", "instruction_rephrase")),
                prompt_content=data.get("prompt_content", ""),
                system_prompt=data.get("system_prompt"),
                scores=data.get("scores", {}),
                mean_scores=data.get("mean_scores", {}),
                std_scores=data.get("std_scores", {}),
                evaluation_time=data.get("evaluation_time", 0.0),
                token_usage=data.get("token_usage", {}),
                cost=data.get("cost", 0.0),
                metadata=data.get("metadata", {}),
            )

        # Run ranking
        result = ranker.rank(
            variant_results=variant_results,
            method=method,
            metrics=request.metrics,
        )

        return result.to_dict()

    except Exception as e:
        logger.error(f"Error ranking variants: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/statistics")
async def get_history_statistics(
    history: HistoryManager = Depends(get_history),
):
    """Get overall experiment statistics."""
    return history.get_statistics()


@router.get("/lineage/{variant_id}")
async def get_variant_lineage(
    variant_id: str,
    history: HistoryManager = Depends(get_history),
):
    """Get lineage information for a variant."""
    lineage = history.get_lineage(variant_id)

    if lineage is None:
        raise HTTPException(status_code=404, detail="Variant not found")

    return {
        "variant_id": lineage.variant_id,
        "parent_variants": lineage.parent_variants,
        "experiments": lineage.experiments,
        "creation_timestamp": lineage.creation_timestamp,
        "strategy": lineage.strategy,
    }


# Export router
__all__ = ["router"]
