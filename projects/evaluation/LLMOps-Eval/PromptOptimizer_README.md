# PromptOptimizer: Automated Prompt Testing & Optimization

A production-ready system for systematic prompt engineering with A/B testing, statistical analysis, and automated prompt optimization.

## 🎯 Overview

PromptOptimizer extends LLMOps-Eval with rigorous prompt experimentation capabilities. Instead of random prompt tweaks, it provides **systematic variation strategies**, **statistical A/B testing**, and **data-driven prompt selection**.

### Key Features

- 🔄 **11 Systematic Variation Strategies** - Instruction rephrasing, few-shot selection, CoT styles, etc.
- 🧪 **A/B/n Testing Framework** - Proper experimental design with random assignment
- 📊 **Statistical Analysis** - T-tests, Mann-Whitney, effect sizes, power analysis
- 🏆 **Intelligent Selection** - Multi-criteria ranking with confidence scoring
- 📈 **Interactive Dashboard** - Streamlit UI for experiment management
- 🔁 **Reproducibility** - Seeded randomness for consistent results

## 📚 Key Concepts

### Statistical Significance

Before adopting a prompt change, ensure it's **statistically significant**:

- **P-value < α (typically 0.05)**: The improvement isn't due to chance
- **Effect Size (Cohen's d)**: Magnitude of improvement
  - 0.2 = small, 0.5 = medium, 0.8 = large
- **Statistical Power**: Probability of detecting a real effect (aim for ≥0.80)

### Multiple Comparison Correction

When testing multiple variants, correct for false positives:

- **Bonferroni**: Conservative, divides α by number of comparisons
- **Benjamini-Hochberg (FDR)**: Less conservative, controls false discovery rate
- **Holm-Bonferroni**: Step-down procedure, good balance

### Sample Size Planning

Calculate required sample size **before** experimenting:

```python
from src.prompt_optimizer.statistics.analyzer import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()
n = analyzer.calculate_required_sample_size(
    effect_size=0.5,  # Medium effect
    power=0.80,
    alpha=0.05
)
print(f"Need {n} samples per variant")
```

## 🚀 Quick Start

### 1. Create a Prompt Template

```python
from src.prompt_optimizer.templates import create_template_manager

manager = create_template_manager()

template = manager.create_template(
    name="summarization",
    template_string="You are a {{role}}. Summarize: {{text}}",
    default_variables={"role": "helpful assistant"},
    category="nlp",
    tags=["summarization", "text-processing"]
)
```

### 2. Generate Prompt Variations

```python
from src.prompt_optimizer.variations import (
    create_variation_generator,
    VariationStrategy,
)

generator = create_variation_generator(seed=42)

variations = generator.generate(
    base_prompt="Summarize the following text.",
    strategies=[
        VariationStrategy.INSTRUCTION_REPHRASE,
        VariationChainOfThoughtStrategy,
        VariationStrategy.OUTPUT_FORMAT,
    ],
    variations_per_strategy=3,
)

print(f"Generated {len(variations.variations)} variations")
```

### 3. Run an A/B Test Experiment

```python
from src.prompt_optimizer.experiments import (
    create_experiment_framework,
    ExperimentConfig,
)
from src.prompt_optimizer.experiments.ab_test import ABTestConfig

# Create framework
framework = create_experiment_framework()

# Configure experiment
config = ExperimentConfig(
    name="Instruction Phrasing A/B Test",
    description="Test different instruction phrasings for summarization",
    base_template_id="summarization",
    dataset_name="summarization_test",
    strategies=[VariationStrategy.INSTRUCTION_REPHRASE],
    variations_per_strategy=5,
    metrics=["semantic_similarity", "llm_judge"],
    provider="openai",
    model="gpt-4o-mini",
    test_config=ABTestConfig(
        alpha=0.05,
        min_sample_size=30,
        paired_design=True,
    ),
)

# Run experiment
result = await framework.run_experiment(
    config=config,
    base_prompt="Summarize the text.",
)

# Save results
result.save("results/experiment_1.json")
```

### 4. Analyze Results

```python
from src.prompt_optimizer.statistics.analyzer import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()

analysis = analyzer.analyze_experiment(
    experiment=result,
    metric="semantic_similarity"
)

print(f"Best variant: {analysis.best_variant_id}")
print(f"Improvement: {analysis.best_variant_improvement:.3f}")

for variant_id, test_result in analysis.treatment_results.items():
    print(f"{variant_id}:")
    print(f"  P-value: {test_result.p_value:.4f}")
    print(f"  Effect size: {test_result.effect_size:.3f} ({test_result.effect_size_interpretation})")
    print(f"  Significant: {test_result.significant}")
```

### 5. Select Best Prompt

```python
from src.prompt_optimizer.selection.selector import (
    BestPromptSelector,
    SelectionCriteria,
)

# Configure selection criteria
criteria = SelectionCriteria(
    metric_weights={
        "semantic_similarity": 0.6,
        "llm_judge": 0.3,
        "latency": 0.1,
    },
    min_confidence=0.95,
    min_sample_size=30,
)

# Select best
selector = BestPromptSelector(analyzer)
selection = selector.select_best(result, criteria)

print(f"Selected: {selection.selected_variant_id}")
print(f"Confidence: {selection.confidence_score:.1%}")
print(f"Explanation:\n{selection.explanation}")
```

## 📝 Template System Usage

### Jinja2 Templates

Templates use Jinja2 syntax with custom filters:

```python
template_string = """
You are {{ role }}.

{% if constraints %}
Constraints:
{% for constraint in constraints %}
- {{ constraint }}
{% endfor %}
{% endif %}

{{ instruction }}

{% if examples %}
Examples:
{{ examples|examples }}
{% endif %}

Output: {{ output|json }}
"""
```

### Custom Filters

- `{{ value|datetime }}` - Format datetime
- `{{ value|json }}` - Format as JSON
- `{{ value|truncate:20 }}` - Truncate to N words
- `{{ examples|examples }}` - Format few-shot examples
- `{{ value|token_estimate }}` - Estimate token count

### Variable Extraction

```python
from src.prompt_optimizer.templates import create_template_manager

manager = create_template_manager()

template = manager.create_template(
    name="test",
    template_string="{{var1}} and {{var2}}",
)

print(template.variables)  # {'var1', 'var2'}
```

### Template Validation

```python
validation = template.validate()
print(validation.is_valid)
print(validation.errors)
print(validation.missing_variables)
```

## 🧪 Running Experiments

### Experiment Lifecycle

```
DRAFT → RUNNING → PAUSED/COMPLETED/CANCELLED
```

### Creating Experiments

```python
from src.prompt_optimizer.experiments.ab_testing import ABTestingFramework

framework = ABTestingFramework(storage_path="./experiments")

experiment = framework.create_experiment(
    name="My A/B Test",
    description="Testing instruction variations",
    control_prompt=control_variation,
    treatment_prompts=[treatment1, treatment2],
    test_dataset_id="my_dataset",
    metrics=["accuracy", "latency"],
    config={
        "min_sample_size": 50,
        "significance_level": 0.05,
        "random_seed": 42,
    },
)

experiment = framework.start_experiment(experiment.id)
```

### Monitoring Progress

```python
experiment = framework.get_experiment(experiment_id)

print(f"Status: {experiment.status}")
print(f"Samples per variant: {experiment.sample_size_per_variant}")
print(f"Total results: {len(experiment.results)}")

# Check if complete
if experiment.status == ExperimentStatus.COMPLETED:
    print("Experiment complete!")
```

## 📊 Interpreting Results

### Statistical Test Results

Each variant comparison provides:

```python
{
    "test_name": "Welch's t-test",
    "statistic": 2.45,
    "p_value": 0.015,
    "significant": True,
    "effect_size": 0.65,
    "effect_size_interpretation": "medium",
    "confidence_interval": (0.12, 1.18),
    "power": 0.78,
}
```

### Effect Size Interpretation

| Cohen's d | Interpretation |
|-----------|---------------|
| < 0.2 | Negligible |
| 0.2 - 0.5 | Small |
| 0.5 - 0.8 | Medium |
| > 0.8 | Large |

### Confidence Intervals

- **95% CI**: Range containing true effect with 95% confidence
- **Non-overlapping CIs**: Strong evidence of difference
- **CI excludes 0**: Statistically significant

### Power Analysis

Low power (< 0.80) means:
- May miss real effects (false negatives)
- Increase sample size or effect size threshold

## 🏆 Best Practices for Prompt Optimization

### 1. Start with Hypotheses

Don't random-test. Have clear hypotheses:

```
Hypothesis: "Using step-by-step reasoning will improve accuracy"
Strategy: Chain-of-Thought variations
Expected effect size: 0.3 (small-medium)
Required n: ~175 per variant (for 80% power, α=0.05)
```

### 2. Use Paired Design

When possible, use **paired tests** (same test cases for all variants):

```python
ABTestConfig(
    paired_design=True,  # More statistical power
)
```

### 3. Correct for Multiple Comparisons

With 5+ variants, ALWAYS apply correction:

```python
ABTestConfig(
    correction_method=MultipleComparisonCorrection.BENJAMINI_HOCHBERG,
)
```

### 4. Focus on Effect Size, Not Just P-Values

A statistically significant result with tiny effect size may not be practically significant:

```python
if test_result.significant and test_result.effect_size > 0.5:
    # Meaningful improvement - adopt it
    pass
```

### 5. Consider Trade-offs

Multi-objective selection balances metrics:

```python
criteria = SelectionCriteria(
    metric_weights={
        "accuracy": 0.5,      # Primary
        "latency": 0.3,       # Secondary
        "cost": 0.2,          # Tertiary
    },
)
```

### 6. Document Everything

```python
experiment.metadata = {
    "hypothesis": "...",
    "expected_effect": "medium",
    "rationale": "...",
}
```

### 7. Iterate Based on Findings

Use experiment results to inform next variations:

```python
if analysis.best_variant_id.endswith("_rephrase"):
    # Rephrasing worked, try more rephrasing
    next_strategies = [VariationStrategy.INSTRUCTION_REPHRASE]
elif analysis.best_variant_id.endswith("_cot"):
    # CoT worked, try different CoT styles
    next_strategies = [VariationStrategy.COT_STYLE]
```

## 📖 API Documentation

### Templates Module

```python
from src.prompt_optimizer.templates import (
    PromptTemplate,
    TemplateManager,
    create_template_manager,
)

# Create manager
manager = create_template_manager()

# Create template
template = manager.create_template(
    name="my_template",
    template_string="...",
    description="...",
    category="general",
    tags=["tag1", "tag2"],
    version="1.0",
)

# Render
rendered = manager.render_template(
    "my_template",
    variables={"key": "value"},
)

# List templates
templates = manager.list_templates()
templates = manager.list_templates(category="nlp")

# Get specific template
template = manager.get_template("my_template")
version = manager.get_version("my_template", "2.0")
```

### Variations Module

```python
from src.prompt_optimizer.variations import (
    VariationGenerator,
    PromptVariation,
    VariationSet,
    VariationStrategy,
    create_variation_generator,
)

# Create generator
generator = create_variation_generator(
    seed=42,
    max_variations=10,
)

# Generate variations
variation_set = generator.generate(
    base_prompt="...",
    strategies=[
        VariationStrategy.INSTRUCTION_REPHRASE,
        VariationStrategy.COT_STYLE,
        VariationStrategy.OUTPUT_FORMAT,
    ],
    variations_per_strategy=3,
    combine_strategies=True,
)

# Access variations
for variation in variation_set.variations:
    print(f"{variation.id}: {variation.description}")
    print(f"{variation.prompt_content}")
```

### Experiments Module

```python
from src.prompt_optimizer.experiments import (
    ExperimentFramework,
    ExperimentConfig,
    ExperimentResult,
    create_experiment_framework,
)
from src.prompt_optimizer.experiments.ab_test import (
    ABTestConfig,
    ABTester,
    TestMethod,
    MultipleComparisonCorrection,
)

# Create framework
framework = create_experiment_framework()

# Create experiment config
config = ExperimentConfig(
    name="Test",
    description="...",
    base_template_id="...",
    dataset_name="...",
    strategies=[...],
    metrics=["semantic_similarity"],
    test_config=ABTestConfig(
        alpha=0.05,
        power=0.80,
        effect_size_threshold=0.5,
        correction_method=MultipleComparisonCorrection.BENJAMINI_HOCHBERG,
        test_method=TestMethod.T_TEST,
        paired_design=True,
    ),
)

# Run experiment
result = await framework.run_experiment(
    config=config,
    base_prompt="...",
    progress_callback=lambda msg, p: print(f"{msg}: {p:.0%}"),
)

# Save/Load
result.save("path/to/result.json")
loaded = ExperimentResult.load("path/to/result.json")
```

### Statistics Module

```python
from src.prompt_optimizer.statistics import (
    StatisticalAnalyzer,
    MultipleComparisonCorrection,
)

# Analyze experiment
analyzer = StatisticalAnalyzer(alpha=0.05)

analysis = analyzer.analyze_experiment(
    experiment=result,
    metric="semantic_similarity",
)

print(f"Best variant: {analysis.best_variant_id}")
print(f"Recommendations: {analysis.recommendations}")

# Sample size calculation
n = analyzer.calculate_required_sample_size(
    effect_size=0.5,
    power=0.80,
    alpha=0.05,
)

# Multiple comparison correction
from src.prompt_optimizer.statistics import MultipleComparisonCorrection

corrector = MultipleComparisonCorrection()

corrected = corrector.apply_correction(
    p_values=[0.01, 0.03, 0.05, 0.10],
    method=CorrectionMethod.BONFERRONI,
    alpha=0.05,
)
```

### Selection Module

```python
from src.prompt_optimizer.selection import (
    BestPromptSelector,
    PromptRanker,
    SelectionStrategy,
    RankingMethod,
    SelectionCriteria,
)

# Select best prompt
selector = BestPromptSelector(analyzer)

criteria = SelectionCriteria(
    metric_weights={"accuracy": 0.7, "latency": 0.3},
    min_confidence=0.95,
)

selection = selector.select(
    variant_results=...,
    statistical_tests=...,
    strategy=SelectionStrategy.STATISTICAL_WINNER,
    primary_metric="accuracy",
)

# Rank variants
ranker = PromptRanker()

ranking = ranker.rank(
    variant_results=...,
    method=RankingMethod.BORDA_COUNT,
    metrics=["accuracy", "latency"],
)
```

### History Module

```python
from src.prompt_optimizer.history import (
    HistoryManager,
    HistorySortOrder,
)

manager = HistoryManager(storage_path="./experiments/history")

# Add experiment to history
entry = manager.add_experiment(
    result=result,
    result_path="results/exp1.json",
    tags=["summarization", "cot"],
)

# List experiments
experiments = manager.list_experiments(
    limit=50,
    sort_by=HistorySortOrder.NEWEST_FIRST,
    filters={"status": "completed", "tags": ["cot"]},
)

# Get statistics
stats = manager.get_statistics()
print(f"Total experiments: {stats['total_experiments']}")

# Compare experiments
comparison = manager.compare_experiments(["exp1", "exp2"])
```

### API Endpoints

```python
# Start FastAPI server
uvicorn src.api.main:app --reload

# Endpoints:
POST /api/v1/optimizer/templates
GET  /api/v1/optimizer/templates
POST /api/v1/optimizer/templates/{name}/render

POST /api/v1/optimizer/variations/generate
GET  /api/v1/optimizer/variations/{set_id}

POST /api/v1/optimizer/experiments
GET  /api/v1/optimizer/experiments
POST /api/v1/optimizer/experiments/{id}/run
GET  /api/v1/optimizer/experiments/{id}/analysis
POST /api/v1/optimizer/experiments/{id}/select-best

GET  /api/v1/optimizer/sample-size-calculator
```

### Dashboard

```bash
# Run Streamlit dashboard
streamlit run src/prompt_optimizer/dashboard/app.py
```

Features:
- 📝 Template management (create, browse, render)
- 🔄 Variation generator with 11 strategies
- 🧪 Experiment creation wizard
- 📊 Live experiment monitoring
- 📈 Statistical analysis visualizations
- 🏆 Best prompt selection with explanations

## 🔧 Configuration

### Environment Variables

```bash
# LLM Provider Configuration
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...

# Optimizer Settings
OPTIMIZER_DEFAULT_ALPHA=0.05
OPTIMIZER_DEFAULT_POWER=0.80
OPTIMIZER_EFFECT_SIZE_THRESHOLD=0.5

# Storage
OPTIMIZER_STORAGE_PATH=./data/experiments
OPTIMIZER_TEMPLATE_PATH=./data/prompt_templates
```

### Config File

```python
# src/prompt_optimizer/config.py
from pydantic_settings import BaseSettings

class OptimizerSettings(BaseSettings):
    # Statistical parameters
    confidence_level: float = 0.95
    significance_level: float = 0.05
    effect_size_threshold: float = 0.5

    # Sample size
    min_sample_size: int = 30
    required_sample_size: int = 100

    # Correction method
    correction_method: str = "benjamini_hochberg"

    # Variation limits
    max_variations_per_strategy: int = 10
    max_total_variations: int = 50

    class Config:
        env_prefix = "OPTIMIZER"
```

## 📦 Project Structure

```
src/prompt_optimizer/
├── __init__.py
├── config.py                      # Configuration settings
├── api/
│   ├── __init__.py
│   ├── routes.py                  # FastAPI routes
│   └── endpoints.py               # API endpoints
├── templates/
│   ├── __init__.py
│   ├── template_manager.py        # Template CRUD
│   └── jinja_env.py               # Jinja2 environment
├── variations/
│   ├── __init__.py
│   └── variation_generator.py     # 11 variation strategies
├── experiments/
│   ├── __init__.py
│   ├── ab_test.py                 # Statistical tests
│   ├── ab_testing.py              # A/B testing framework
│   └── framework.py               # Experiment orchestration
├── statistics/
│   ├── __init__.py
│   ├── analyzer.py                # Statistical analysis
│   ├── tests.py                   # Statistical tests
│   └── corrections.py             # Multiple comparison corrections
├── selection/
│   ├── __init__.py
│   ├── selector.py                # Best prompt selection
│   └── ranking.py                 # Variant ranking methods
├── history/
│   ├── __init__.py
│   └── history.py                 # Experiment history tracking
└── dashboard/
    └── app.py                     # Streamlit dashboard

tests/
└── test_prompt_optimizer.py       # Comprehensive tests

data/
├── experiments/                   # Experiment results
│   ├── history/                   # History storage
│   └── prompt_templates/          # Template definitions
└── datasets/                      # Test datasets
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/test_prompt_optimizer.py -v

# Run specific test class
pytest tests/test_prompt_optimizer.py::TestVariationGeneration -v

# Run with coverage
pytest tests/test_prompt_optimizer.py --cov=src/prompt_optimizer --cov-report=html
```

## 🚢 Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/

EXPOSE 8501 8000

# Streamlit dashboard
CMD ["streamlit", "run", "src/prompt_optimizer/dashboard/app.py", "--server.port=8501"]
```

### Docker Compose

```yaml
services:
  optimizer-api:
    build: .
    command: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}

  optimizer-dashboard:
    build: .
    command: streamlit run src/prompt_optimizer/dashboard/app.py
    ports:
      - "8501:8501"
```

## 📈 Monitoring & Metrics

Prometheus metrics exposed at `/metrics`:

- `promptoptimizer_experiments_total`
- `promptoptimizer_variations_generated`
- `promptoptimizer_statistical_tests_duration_seconds`
- `promptoptimizer_selection_confidence_score`

## 🤝 Contributing

1. Add tests for new features
2. Ensure statistical rigor
3. Document assumptions
4. Use reproducible seeds

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

Built on top of [LLMOps-Eval](../) with statistical methods from:
- SciPy (statistical tests)
- Statsmodels (ANOVA, post-hoc tests)
- Research on prompt engineering
