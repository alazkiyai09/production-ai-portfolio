"""
FastAPI endpoints for PromptOptimizer.

This module provides REST API endpoints for prompt template management,
variation generation, experiment execution, and statistical analysis.
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
import uuid

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/optimizer", tags=["prompt-optimizer"])

# ============================================================================
# Pydantic Models
# ============================================================================

# --- Template Models ---

class TemplateCreateRequest(BaseModel):
    """Request to create a new prompt template."""
    name: str = Field(..., description="Template name", min_length=1, max_length=100)
    description: str = Field("", description="Template description")
    template_string: str = Field(..., description="Jinja2 template string")
    default_variables: Dict[str, Any] = Field(default_factory=dict)
    category: str = Field("general", description="Template category")
    tags: List[str] = Field(default_factory=list)
    version: str = Field("1.0", description="Template version")


class TemplateRenderRequest(BaseModel):
    """Request to render a template."""
    variables: Dict[str, Any] = Field(..., description="Variables for template rendering")
    version: Optional[str] = Field(None, description="Specific version to render")


class TemplateResponse(BaseModel):
    """Template response."""
    id: str
    name: str
    description: str
    template_string: str
    default_variables: Dict[str, Any]
    category: str
    tags: List[str]
    version: str
    created_at: str
    variables_extracted: List[str]


class TemplateRenderResponse(BaseModel):
    """Rendered template response."""
    rendered_prompt: str
    template_name: str
    version: str
    variables_used: Dict[str, Any]
    estimated_tokens: int


# --- Variation Models ---

class VariationGenerateRequest(BaseModel):
    """Request to generate prompt variations."""
    base_prompt: str = Field(..., description="Base prompt to vary")
    strategies: List[str] = Field(
        ...,
        description="Variation strategies (instruction_rephrase, few_shot_selection, etc.)"
    )
    variations_per_strategy: int = Field(3, ge=1, le=20, description="Variations per strategy")
    combine_strategies: bool = Field(False, description="Combine multiple strategies")
    max_total_variations: int = Field(50, ge=1, le=200, description="Maximum total variations")

    @validator("strategies")
    def validate_strategies(cls, v):
        valid_strategies = {
            "instruction_rephrase", "few_shot_selection", "few_shot_order",
            "output_format", "cot_style", "system_prompt", "emphasis",
            "verbosity", "temperature", "top_p", "context_position"
        }
        invalid = set(v) - valid_strategies
        if invalid:
            raise ValueError(f"Invalid strategies: {invalid}")
        return v


class VariationResponse(BaseModel):
    """Single variation response."""
    id: str
    strategy: str
    description: str
    prompt_content: str
    system_prompt: Optional[str]
    variation_params: Dict[str, Any]


class VariationSetResponse(BaseModel):
    """Variation set response."""
    id: str
    name: str
    base_prompt: str
    total_variations: int
    strategies_used: List[str]
    variations: List[VariationResponse]
    created_at: str


# --- Experiment Models ---

class ExperimentCreateRequest(BaseModel):
    """Request to create an experiment."""
    name: str = Field(..., description="Experiment name", min_length=1, max_length=200)
    description: str = Field(..., description="What we're testing")
    base_template_id: str = Field(..., description="Base template ID")
    dataset_name: str = Field(..., description="Dataset to use")
    strategies: List[str] = Field(..., description="Variation strategies to test")
    variations_per_strategy: int = Field(3, ge=1, le=10)
    combine_strategies: bool = Field(False)
    sample_size: Optional[int] = Field(None, ge=10, le=10000)
    metrics: List[str] = Field(
        default=["semantic_similarity", "llm_judge"],
        description="Metrics to evaluate"
    )
    provider: str = Field("openai", description="LLM provider")
    model: str = Field("gpt-4o-mini", description="Model name")
    min_sample_size: int = Field(30, ge=5, description="Min samples per variant")
    significance_level: float = Field(0.05, ge=0.01, le=0.2, description="Alpha level")
    paired_design: bool = Field(True, description="Use paired tests")


class ExperimentResponse(BaseModel):
    """Experiment response."""
    experiment_id: str
    name: str
    description: str
    status: str
    base_template_id: str
    dataset_name: str
    metrics: List[str]
    provider: str
    model: str
    num_variants: int
    sample_size_per_variant: int
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]


class ExperimentResultsResponse(BaseModel):
    """Experiment results response."""
    experiment_id: str
    total_results: int
    variant_results: Dict[str, List[Dict[str, Any]]]
    summary: Dict[str, Any]


class StatisticalAnalysisResponse(BaseModel):
    """Statistical analysis response."""
    experiment_id: str
    control_variant_id: str
    treatment_analyses: Dict[str, Dict[str, Any]]
    best_variant_id: str
    best_variant_improvement: float
    confidence_level: float
    warnings: List[str]
    recommendations: List[str]


class BestPromptSelectionResponse(BaseModel):
    """Best prompt selection response."""
    selected_variant_id: str
    selected_prompt: str
    confidence_score: float
    weighted_score: float
    metric_scores: Dict[str, float]
    comparison_to_baseline: Dict[str, float]
    explanation: str
    runner_ups: List[Dict[str, Any]]


# --- Sample Size Calculator Models ---

class SampleSizeRequest(BaseModel):
    """Request to calculate sample size."""
    effect_size: float = Field(..., gt=0, description="Expected Cohen's d", ge=0.1, le=3.0)
    power: float = Field(0.80, ge=0.5, le=0.99, description="Desired statistical power")
    alpha: float = Field(0.05, ge=0.01, le=0.2, description="Significance level")
    ratio: float = Field(1.0, gt=0, description="Sample size ratio (n2/n1)")


class SampleSizeResponse(BaseModel):
    """Sample size calculation response."""
    required_sample_size: int
    effect_size: float
    power: float
    alpha: float
    achieved_power: float
    minimum_detectable_effect: float
    recommendations: List[str]


# ============================================================================
# Template Endpoints
# ============================================================================

@router.post("/templates", response_model=TemplateResponse, status_code=201)
async def create_template(request: TemplateCreateRequest):
    """Create a new prompt template."""
    try:
        from src.prompt_optimizer.templates import TemplateManager

        manager = TemplateManager()

        # Create template
        template = manager.create_template(
            name=request.name,
            template_string=request.template_string,
            description=request.description,
            default_variables=request.default_variables,
            category=request.category,
            tags=request.tags,
            version=request.version,
        )

        return TemplateResponse(
            id=template.id,
            name=template.name,
            description=template.description,
            template_string=template.template_string,
            default_variables=template.default_variables,
            category=template.category,
            tags=template.tags,
            version=template.version,
            created_at=template.created_at,
            variables_extracted=template.variables,
        )

    except Exception as e:
        logger.error(f"Error creating template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates", response_model=List[TemplateResponse])
async def list_templates(
    category: Optional[str] = Query(None, description="Filter by category"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    limit: int = Query(100, ge=1, le=500),
):
    """List all prompt templates."""
    try:
        from src.prompt_optimizer.templates import TemplateManager

        manager = TemplateManager()
        templates = manager.list_templates()

        # Apply filters
        if category:
            templates = [t for t in templates if t.category == category]
        if tag:
            templates = [t for t in templates if tag in t.tags]

        # Limit
        templates = templates[:limit]

        return [
            TemplateResponse(
                id=t.id,
                name=t.name,
                description=t.description,
                template_string=t.template_string,
                default_variables=t.default_variables,
                category=t.category,
                tags=t.tags,
                version=t.version,
                created_at=t.created_at,
                variables_extracted=t.variables,
            )
            for t in templates
        ]

    except Exception as e:
        logger.error(f"Error listing templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates/{name}", response_model=TemplateResponse)
async def get_template(
    name: str,
    version: Optional[str] = Query(None, description="Specific version")
):
    """Get a template by name (and optionally version)."""
    try:
        from src.prompt_optimizer.templates import TemplateManager

        manager = TemplateManager()

        if version:
            template = manager.get_version(name, version)
        else:
            template = manager.get_template(name)

        if not template:
            raise HTTPException(status_code=404, detail=f"Template not found: {name}")

        return TemplateResponse(
            id=template.id,
            name=template.name,
            description=template.description,
            template_string=template.template_string,
            default_variables=template.default_variables,
            category=template.category,
            tags=template.tags,
            version=template.version,
            created_at=template.created_at,
            variables_extracted=template.variables,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/templates/{name}/render", response_model=TemplateRenderResponse)
async def render_template(
    name: str,
    request: TemplateRenderRequest,
):
    """Render a template with variables."""
    try:
        from src.prompt_optimizer.templates import TemplateManager

        manager = TemplateManager()

        # Get template
        template = manager.get_template(name)
        if not template:
            raise HTTPException(status_code=404, detail=f"Template not found: {name}")

        # Merge variables
        variables = {**template.default_variables, **request.variables}

        # Render
        rendered = manager.render_template(name, variables)

        # Estimate tokens
        estimated_tokens = len(rendered.rendered_content) // 4

        return TemplateRenderResponse(
            rendered_prompt=rendered.rendered_content,
            template_name=template.name,
            version=template.version,
            variables_used=variables,
            estimated_tokens=estimated_tokens,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rendering template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Variation Endpoints
# ============================================================================

@router.post("/variations/generate", response_model=VariationSetResponse, status_code=201)
async def generate_variations(request: VariationGenerateRequest):
    """Generate prompt variations using specified strategies."""
    try:
        from src.prompt_optimizer.variations import create_variation_generator, VariationStrategy

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
        generator = create_variation_generator()
        variation_set = generator.generate(
            base_prompt=request.base_prompt,
            strategies=strategies,
            variations_per_strategy=request.variations_per_strategy,
            combine_strategies=request.combine_strategies,
            max_total_variations=request.max_total_variations,
        )

        return VariationSetResponse(
            id=variation_set.id,
            name=variation_set.name,
            base_prompt=request.base_prompt,
            total_variations=len(variation_set.variations),
            strategies_used=[s.value for s in variation_set.strategies_used],
            variations=[
                VariationResponse(
                    id=v.id,
                    strategy=v.strategy.value,
                    description=v.description,
                    prompt_content=v.prompt_content,
                    system_prompt=v.system_prompt,
                    variation_params=v.variation_params,
                )
                for v in variation_set.variations
            ],
            created_at=variation_set.created_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating variations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/variations/{set_id}", response_model=VariationSetResponse)
async def get_variation_set(set_id: str):
    """Get a variation set by ID."""
    try:
        # In a real implementation, this would load from storage
        raise HTTPException(
            status_code=501,
            detail="Variation set retrieval not yet implemented"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting variation set: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Experiment Endpoints
# ============================================================================

@router.post("/experiments", response_model=ExperimentResponse, status_code=201)
async def create_experiment(
    request: ExperimentCreateRequest,
    background_tasks: BackgroundTasks,
):
    """Create a new A/B test experiment."""
    try:
        from src.prompt_optimizer.experiments import (
            ExperimentConfig,
            create_experiment_framework,
        )
        from src.prompt_optimizer.variations import VariationStrategy

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
            alpha=request.significance_level,
            min_sample_size=request.min_sample_size,
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
        experiment_id = str(uuid.uuid4())[:8]

        # Store config (in real implementation, use database)
        # For now, return a placeholder response
        return ExperimentResponse(
            experiment_id=experiment_id,
            name=config.name,
            description=config.description,
            status="pending",
            base_template_id=config.base_template_id,
            dataset_name=config.dataset_name,
            metrics=config.metrics,
            provider=config.provider,
            model=config.model,
            num_variants=0,
            sample_size_per_variant=0,
            created_at=datetime.utcnow().isoformat(),
            started_at=None,
            completed_at=None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments", response_model=List[ExperimentResponse])
async def list_experiments(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=500),
):
    """List all experiments."""
    try:
        from src.prompt_optimizer.history import HistoryManager, HistorySortOrder

        history = HistoryManager()

        # Parse status filter
        from src.prompt_optimizer.experiments.ab_testing import ExperimentStatus
        filter_status = ExperimentStatus(status) if status else None

        # Get experiments
        experiments = history.list_experiments(
            limit=limit,
            sort_by=HistorySortOrder.NEWEST_FIRST,
            filters={"status": status} if status else None,
        )

        return [
            ExperimentResponse(
                experiment_id=exp.experiment_id,
                name=exp.name,
                description=exp.description,
                status=exp.status,
                base_template_id="",  # Would be in full data
                dataset_name="",  # Would be in full data
                metrics=exp.metrics_tracked,
                provider="",  # Would be in full data
                model="",  # Would be in full data
                num_variants=exp.metadata.get("total_variants", 0),
                sample_size_per_variant=0,  # Would be in full data
                created_at=exp.timestamp,
                started_at=None,
                completed_at=None,
            )
            for exp in experiments
        ]

    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{id}", response_model=ExperimentResponse)
async def get_experiment(id: str):
    """Get experiment details."""
    try:
        from src.prompt_optimizer.history import HistoryManager

        history = HistoryManager()
        result = history.get_experiment(id)

        if not result:
            raise HTTPException(status_code=404, detail=f"Experiment not found: {id}")

        return ExperimentResponse(
            experiment_id=result.experiment_id,
            name=result.config.name,
            description=result.config.description,
            status=result.status.value,
            base_template_id=result.config.base_template_id,
            dataset_name=result.config.dataset_name,
            metrics=result.config.metrics,
            provider=result.config.provider,
            model=result.config.model,
            num_variants=len(result.variant_results),
            sample_size_per_variant=0,  # Calculate from results
            created_at=result.started_at,
            started_at=result.started_at,
            completed_at=result.completed_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/{id}/start", response_model=ExperimentResponse)
async def start_experiment(id: str):
    """Start a pending experiment."""
    try:
        from src.prompt_optimizer.experiments.ab_testing import ABTestingFramework, ExperimentStatus

        framework = ABTestingFramework()
        experiment = framework.start_experiment(id)

        return ExperimentResponse(
            experiment_id=experiment.id,
            name=experiment.name,
            description=experiment.description,
            status=experiment.status.value,
            base_template_id="",
            dataset_name=experiment.test_dataset_id,
            metrics=experiment.metrics,
            provider="",
            model="",
            num_variants=len(experiment.variants),
            sample_size_per_variant=experiment.sample_size_per_variant,
            created_at=experiment.created_at,
            started_at=experiment.started_at,
            completed_at=experiment.completed_at,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/{id}/run", response_model=ExperimentResponse)
async def run_experiment(
    id: str,
    background_tasks: BackgroundTasks,
):
    """Run an experiment to completion (async)."""
    try:
        # Start experiment in background
        background_tasks.add_task(_run_experiment_task, id)

        return JSONResponse(
            status_code=202,
            content={
                "message": "Experiment started",
                "experiment_id": id,
                "status": "running",
            }
        )

    except Exception as e:
        logger.error(f"Error running experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _run_experiment_task(experiment_id: str):
    """Background task to run experiment."""
    try:
        from src.prompt_optimizer.experiments.ab_testing import ABTestingFramework
        from src.datasets import DatasetManager
        from src.models import create_provider

        framework = ABTestingFramework()
        experiment = framework.get_experiment(experiment_id)

        if not experiment:
            logger.error(f"Experiment not found: {experiment_id}")
            return

        # Load dataset
        dataset_mgr = DatasetManager()
        dataset = dataset_mgr.load_dataset(experiment.test_dataset_id)

        # Create LLM provider
        provider = create_provider(experiment.config.get("provider", "openai"))

        # Run experiment
        await framework.run_experiment(
            experiment_id,
            dataset.test_cases,
            provider
        )

        logger.info(f"Experiment {experiment_id} completed")

    except Exception as e:
        logger.error(f"Error running experiment {experiment_id}: {e}")


@router.get("/experiments/{id}/results", response_model=ExperimentResultsResponse)
async def get_experiment_results(id: str):
    """Get raw experiment results."""
    try:
        from src.prompt_optimizer.history import HistoryManager

        history = HistoryManager()
        result = history.get_experiment(id)

        if not result:
            raise HTTPException(status_code=404, detail=f"Experiment not found: {id}")

        # Group results by variant
        variant_results = {}
        for var_id, var_result in result.variant_results.items():
            variant_results[var_id] = [
                {"metric": metric, "value": value}
                for metric, values in var_result.scores.items()
                for value in values
            ]

        return ExperimentResultsResponse(
            experiment_id=id,
            total_results=sum(len(rs) for rs in variant_results.values()),
            variant_results=variant_results,
            summary={
                "num_variants": len(result.variant_results),
                "best_variant": result.best_variant,
                "total_time": result.total_time,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{id}/analysis", response_model=StatisticalAnalysisResponse)
async def get_experiment_analysis(
    id: str,
    metric: str = Query(..., description="Metric to analyze"),
):
    """Get statistical analysis of experiment results."""
    try:
        from src.prompt_optimizer.history import HistoryManager
        from src.prompt_optimizer.statistics.analyzer import StatisticalAnalyzer

        history = HistoryManager()
        result = history.get_experiment(id)

        if not result:
            raise HTTPException(status_code=404, detail=f"Experiment not found: {id}")

        # Run statistical analysis
        analyzer = StatisticalAnalyzer()

        # Convert to Experiment format expected by analyzer
        # This is a simplified version - real implementation would convert properly
        analysis = analyzer.analyze_experiment(result, metric)

        return StatisticalAnalysisResponse(
            experiment_id=id,
            control_variant_id=analysis.control_variant_id,
            treatment_analyses={
                vid: {
                    "test_name": r.test_name,
                    "p_value": r.p_value,
                    "significant": r.significant,
                    "effect_size": r.effect_size,
                    "effect_size_interpretation": r.effect_size_interpretation,
                    "confidence_interval": r.confidence_interval,
                    "power": r.power,
                    "recommendation": r.recommendation,
                }
                for vid, r in analysis.treatment_results.items()
            },
            best_variant_id=analysis.best_variant_id,
            best_variant_improvement=analysis.best_variant_improvement,
            confidence_level=analysis.confidence_level,
            warnings=analysis.warnings,
            recommendations=analysis.recommendations,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/{id}/select-best", response_model=BestPromptSelectionResponse)
async def select_best_prompt(
    id: str,
    metric_weights: Dict[str, float] = Body(..., description="Metric weights"),
):
    """Select the best prompt from experiment results."""
    try:
        from src.prompt_optimizer.history import HistoryManager
        from src.prompt_optimizer.selection.selector import (
            SelectionCriteria,
            BestPromptSelector,
        )
        from src.prompt_optimizer.statistics.analyzer import StatisticalAnalyzer

        history = HistoryManager()
        result = history.get_experiment(id)

        if not result:
            raise HTTPException(status_code=404, detail=f"Experiment not found: {id}")

        # Create selection criteria
        criteria = SelectionCriteria(
            metric_weights=metric_weights,
            min_confidence=0.95,
            min_sample_size=30,
        )

        # Select best
        analyzer = StatisticalAnalyzer()
        selector = BestPromptSelector(analyzer)

        selection = selector.select_best(result, criteria)

        return BestPromptSelectionResponse(
            selected_variant_id=selection.selected_variant_id,
            selected_prompt=selection.selected_prompt,
            confidence_score=selection.confidence_score,
            weighted_score=selection.weighted_score,
            metric_scores=selection.metric_scores,
            comparison_to_baseline=selection.comparison_to_baseline,
            explanation=selection.explanation,
            runner_ups=selection.runner_up_variants,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error selecting best prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Sample Size Calculator
# ============================================================================

@router.get("/sample-size-calculator", response_model=SampleSizeResponse)
async def calculate_sample_size(
    effect_size: float = Query(..., gt=0, ge=0.1, le=3.0),
    power: float = Query(0.80, ge=0.5, le=0.99),
    alpha: float = Query(0.05, ge=0.01, le=0.2),
    ratio: float = Query(1.0, gt=0),
):
    """Calculate required sample size for A/B test."""
    try:
        from src.prompt_optimizer.statistics.analyzer import StatisticalAnalyzer

        analyzer = StatisticalAnalyzer()

        required_n = analyzer.calculate_required_sample_size(
            effect_size=effect_size,
            power=power,
            alpha=alpha,
        )

        # Calculate achieved power with this sample size
        achieved_power = analyzer._calculate_power(required_n, int(required_n * ratio), effect_size)

        # Minimum detectable effect
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)
        mde = (z_alpha + z_beta) * np.sqrt((1 + ratio) / (required_n * ratio))

        # Generate recommendations
        recommendations = []
        if required_n > 1000:
            recommendations.append(
                "Large sample size required. Consider increasing effect size threshold "
                "or reducing power requirements."
            )
        if effect_size < 0.3:
            recommendations.append(
                "Small effect size may not be practically significant. "
                "Consider if the detected difference would be meaningful in practice."
            )
        if power < 0.8:
            recommendations.append(
                f"Statistical power ({power:.2f}) is below recommended 0.80. "
                "Consider increasing sample size."
            )
        if not recommendations:
            recommendations.append("Sample size is adequate for the specified parameters.")

        return SampleSizeResponse(
            required_sample_size=required_n,
            effect_size=effect_size,
            power=power,
            alpha=alpha,
            achieved_power=achieved_power,
            minimum_detectable_effect=mde,
            recommendations=recommendations,
        )

    except Exception as e:
        logger.error(f"Error calculating sample size: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "prompt-optimizer",
        "version": "0.1.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


# Export router
__all__ = ["router"]
