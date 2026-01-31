# GLM-4.7 Implementation Guide: Project 3B - PromptOptimizer
## Automated Prompt Testing & Optimization Toolkit

---

# PROJECT 3B: PromptOptimizer
## Complete Implementation Guide

### Project Overview

**What You'll Build**: A comprehensive prompt engineering toolkit with:
- Automated prompt variation generation
- A/B testing framework for prompts
- Statistical significance testing
- Performance tracking across versions
- Best prompt selection with confidence scores
- Prompt versioning and history

**Why This Matters for Jobs**:
- Shows systematic prompt engineering (not just trial-and-error)
- Demonstrates data-driven optimization mindset
- Useful internal tool for any AI team
- Differentiates you from "vibe-based" prompt engineers

**Prerequisites**: Project 3A (LLMOps-Eval) should be completed first, as this extends that infrastructure.

**Time Estimate**: 5-7 days

---

## SESSION SETUP PROMPT

Copy and paste this to start your GLM-4.7 session:

```
You are an expert Python developer helping me build a prompt optimization toolkit.

PROJECT: PromptOptimizer
PURPOSE: An automated prompt testing and optimization system demonstrating:
- Prompt variation generation (systematic, not random)
- A/B testing framework with proper experimental design
- Statistical significance testing (t-tests, confidence intervals)
- Performance tracking across prompt versions
- Best prompt selection with statistical confidence
- Integration with LLMOps-Eval evaluation framework

TECH STACK:
- Python 3.11+
- Extends LLMOps-Eval (Project 3A) infrastructure
- scipy for statistical analysis
- Jinja2 for prompt templates
- pandas for data analysis
- plotly for visualization
- FastAPI for API
- SQLite for prompt history
- Streamlit for dashboard

PROMPT OPTIMIZATION TECHNIQUES:
1. Template variations (instruction phrasing, ordering)
2. Few-shot example selection and ordering
3. Output format variations (JSON, markdown, structured)
4. Chain-of-thought vs direct prompting
5. System prompt variations
6. Temperature/parameter sweeps

STATISTICAL RIGOR:
- Proper sample sizes for significance
- Multiple comparison correction (Bonferroni)
- Effect size calculation (Cohen's d)
- Confidence intervals
- Power analysis

QUALITY REQUIREMENTS:
- Type hints on all functions
- Comprehensive docstrings
- Reproducible experiments
- Audit trail for all tests
- Production-ready code

USER CONTEXT:
- This builds on my LLMOps-Eval project
- Targeting AI Engineer roles that value systematic approaches
- Shows I understand prompt engineering as engineering, not art

RULES:
1. Generate complete, runnable code
2. Include all imports
3. Add comments explaining statistical methods
4. Follow experimental design best practices

Please confirm you understand, then we'll build this file by file.
```

---

## PROMPT 3B.1: Project Structure & Configuration

```
Create the complete project structure for PromptOptimizer.

This extends the LLMOps-Eval project, so it should integrate with that codebase.

Generate these files:

1. Directory structure (show as tree)
2. requirements.txt (additional dependencies)
3. src/prompt_optimizer/__init__.py
4. src/prompt_optimizer/config.py

For additional requirements.txt dependencies:
- scipy>=1.11.0
- jinja2>=3.1.0
- statsmodels>=0.14.0
- pandas>=2.0.0
- plotly>=5.18.0
- sqlalchemy>=2.0.0

Structure (within the LLMOps-Eval project):
src/prompt_optimizer/
├── templates/           # Prompt template management
├── variations/          # Variation generation
├── experiments/         # A/B testing framework
├── statistics/          # Statistical analysis
├── selection/           # Best prompt selection
├── history/             # Version tracking
└── api/                 # Additional API endpoints

For config.py, define:
- Default significance level (alpha = 0.05)
- Minimum sample size for tests
- Maximum concurrent experiments
- Template directory path
- Database path for history

Output all files completely.
```

---

## PROMPT 3B.2: Prompt Template System

```
Create the prompt template management system.

File: src/prompt_optimizer/templates/template_manager.py

Requirements:
1. Jinja2-based template system
2. Template versioning
3. Variable extraction from templates
4. Template validation
5. Template storage and retrieval

Implementation:

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
from jinja2 import Environment, BaseLoader, meta
import hashlib
import json
from datetime import datetime

@dataclass
class PromptTemplate:
    """A versioned prompt template."""
    id: str
    name: str
    description: str
    template_string: str
    variables: List[str]
    version: int
    created_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def hash(self) -> str:
        """Generate hash for template content."""
        return hashlib.sha256(self.template_string.encode()).hexdigest()[:12]

@dataclass
class RenderedPrompt:
    """A rendered prompt ready for execution."""
    template_id: str
    template_version: int
    content: str
    variables_used: Dict[str, Any]
    rendered_at: str

class TemplateManager:
    """Manage prompt templates with versioning."""
    
    def __init__(self, storage_path: str = "./templates"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.env = Environment(loader=BaseLoader())
        self._templates: Dict[str, List[PromptTemplate]] = {}
        self._load_templates()
    
    def create_template(
        self,
        name: str,
        template_string: str,
        description: str = "",
        metadata: Dict[str, Any] = None
    ) -> PromptTemplate:
        """
        Create a new prompt template.
        
        Args:
            name: Template name (unique identifier)
            template_string: Jinja2 template string
            description: Human-readable description
            metadata: Additional metadata (task type, model, etc.)
        
        Returns:
            Created PromptTemplate
        """
        # Extract variables from template
        variables = self._extract_variables(template_string)
        
        # Validate template syntax
        self._validate_template(template_string)
        
        # Determine version
        existing = self._templates.get(name, [])
        version = len(existing) + 1
        
        template = PromptTemplate(
            id=f"{name}_v{version}",
            name=name,
            description=description,
            template_string=template_string,
            variables=variables,
            version=version,
            created_at=datetime.utcnow().isoformat(),
            metadata=metadata or {}
        )
        
        # Store
        if name not in self._templates:
            self._templates[name] = []
        self._templates[name].append(template)
        self._save_template(template)
        
        return template
    
    def get_template(
        self,
        name: str,
        version: Optional[int] = None
    ) -> Optional[PromptTemplate]:
        """
        Get a template by name and optional version.
        
        Args:
            name: Template name
            version: Specific version (None = latest)
        
        Returns:
            PromptTemplate or None
        """
        templates = self._templates.get(name, [])
        if not templates:
            return None
        
        if version is None:
            return templates[-1]  # Latest
        
        for t in templates:
            if t.version == version:
                return t
        return None
    
    def render(
        self,
        name: str,
        variables: Dict[str, Any],
        version: Optional[int] = None
    ) -> RenderedPrompt:
        """
        Render a template with variables.
        
        Args:
            name: Template name
            variables: Variables to fill in template
            version: Specific version (None = latest)
        
        Returns:
            RenderedPrompt with filled content
        """
        template = self.get_template(name, version)
        if not template:
            raise ValueError(f"Template not found: {name}")
        
        # Check required variables
        missing = set(template.variables) - set(variables.keys())
        if missing:
            raise ValueError(f"Missing variables: {missing}")
        
        # Render
        jinja_template = self.env.from_string(template.template_string)
        content = jinja_template.render(**variables)
        
        return RenderedPrompt(
            template_id=template.id,
            template_version=template.version,
            content=content,
            variables_used=variables,
            rendered_at=datetime.utcnow().isoformat()
        )
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List all templates with their versions."""
        result = []
        for name, versions in self._templates.items():
            result.append({
                "name": name,
                "versions": len(versions),
                "latest_version": versions[-1].version if versions else 0,
                "description": versions[-1].description if versions else "",
                "variables": versions[-1].variables if versions else []
            })
        return result
    
    def get_version_history(self, name: str) -> List[PromptTemplate]:
        """Get all versions of a template."""
        return self._templates.get(name, [])
    
    def compare_versions(
        self,
        name: str,
        version1: int,
        version2: int
    ) -> Dict[str, Any]:
        """Compare two versions of a template."""
        t1 = self.get_template(name, version1)
        t2 = self.get_template(name, version2)
        
        if not t1 or not t2:
            raise ValueError("One or both versions not found")
        
        # Simple diff
        lines1 = t1.template_string.split('\n')
        lines2 = t2.template_string.split('\n')
        
        return {
            "version1": version1,
            "version2": version2,
            "variables_added": list(set(t2.variables) - set(t1.variables)),
            "variables_removed": list(set(t1.variables) - set(t2.variables)),
            "line_count_change": len(lines2) - len(lines1),
            "hash_v1": t1.hash,
            "hash_v2": t2.hash
        }
    
    def _extract_variables(self, template_string: str) -> List[str]:
        """Extract variable names from Jinja2 template."""
        ast = self.env.parse(template_string)
        return list(meta.find_undeclared_variables(ast))
    
    def _validate_template(self, template_string: str) -> bool:
        """Validate Jinja2 template syntax."""
        try:
            self.env.from_string(template_string)
            return True
        except Exception as e:
            raise ValueError(f"Invalid template syntax: {e}")
    
    def _save_template(self, template: PromptTemplate):
        """Save template to disk."""
        template_dir = self.storage_path / template.name
        template_dir.mkdir(exist_ok=True)
        
        file_path = template_dir / f"v{template.version}.json"
        with open(file_path, 'w') as f:
            json.dump({
                "id": template.id,
                "name": template.name,
                "description": template.description,
                "template_string": template.template_string,
                "variables": template.variables,
                "version": template.version,
                "created_at": template.created_at,
                "metadata": template.metadata
            }, f, indent=2)
    
    def _load_templates(self):
        """Load all templates from disk."""
        if not self.storage_path.exists():
            return
        
        for template_dir in self.storage_path.iterdir():
            if template_dir.is_dir():
                for version_file in sorted(template_dir.glob("v*.json")):
                    with open(version_file) as f:
                        data = json.load(f)
                        template = PromptTemplate(**data)
                        
                        if template.name not in self._templates:
                            self._templates[template.name] = []
                        self._templates[template.name].append(template)


# Predefined template patterns for common optimizations
TEMPLATE_PATTERNS = {
    "instruction_first": """{{ instruction }}

{{ context }}

{{ output_format }}""",
    
    "context_first": """{{ context }}

Based on the above, {{ instruction }}

{{ output_format }}""",
    
    "cot_explicit": """{{ instruction }}

{{ context }}

Let's think step by step:
1. First, I'll analyze the key information
2. Then, I'll apply the relevant criteria
3. Finally, I'll formulate my response

{{ output_format }}""",
    
    "cot_implicit": """{{ instruction }}

{{ context }}

Think through this carefully before responding.

{{ output_format }}""",
    
    "few_shot": """{{ instruction }}

Here are some examples:
{% for example in examples %}
Input: {{ example.input }}
Output: {{ example.output }}
{% endfor %}

Now, given:
{{ context }}

{{ output_format }}"""
}

Output the complete file.
```

---

## PROMPT 3B.3: Prompt Variation Generator

```
Create the prompt variation generator module.

File: src/prompt_optimizer/variations/variation_generator.py

Requirements:
1. Generate systematic prompt variations
2. Support multiple variation strategies
3. Track variation lineage
4. Ensure reproducibility

Variation strategies to implement:
1. Instruction rephrasing (synonyms, structure)
2. Few-shot example selection and ordering
3. Output format variations
4. Chain-of-thought variations
5. System prompt variations
6. Emphasis and tone variations

Implementation:

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from enum import Enum
import itertools
import random
from abc import ABC, abstractmethod

class VariationStrategy(Enum):
    INSTRUCTION_REPHRASE = "instruction_rephrase"
    FEW_SHOT_SELECTION = "few_shot_selection"
    FEW_SHOT_ORDER = "few_shot_order"
    OUTPUT_FORMAT = "output_format"
    COT_STYLE = "cot_style"
    SYSTEM_PROMPT = "system_prompt"
    EMPHASIS = "emphasis"
    VERBOSITY = "verbosity"

@dataclass
class PromptVariation:
    """A single prompt variation."""
    id: str
    base_template_id: str
    strategy: VariationStrategy
    variation_params: Dict[str, Any]
    prompt_content: str
    system_prompt: Optional[str]
    description: str

@dataclass
class VariationSet:
    """A set of variations for A/B testing."""
    id: str
    name: str
    base_template_id: str
    variations: List[PromptVariation]
    strategies_used: List[VariationStrategy]
    created_at: str

class BaseVariationStrategy(ABC):
    """Base class for variation strategies."""
    
    @property
    @abstractmethod
    def strategy_type(self) -> VariationStrategy:
        pass
    
    @abstractmethod
    def generate_variations(
        self,
        base_prompt: str,
        num_variations: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate variations with parameters."""
        pass

class InstructionRephraseStrategy(BaseVariationStrategy):
    """Generate instruction rephrasing variations."""
    
    @property
    def strategy_type(self) -> VariationStrategy:
        return VariationStrategy.INSTRUCTION_REPHRASE
    
    # Rephrase patterns
    REPHRASE_PATTERNS = {
        "direct": "{{ instruction }}",
        "polite": "Please {{ instruction|lower }}",
        "imperative": "You must {{ instruction|lower }}",
        "question": "Can you {{ instruction|lower }}?",
        "task_framing": "Your task is to {{ instruction|lower }}",
        "role_based": "As an expert, {{ instruction|lower }}",
        "outcome_focused": "Provide a response that {{ instruction|lower }}"
    }
    
    def generate_variations(
        self,
        base_prompt: str,
        num_variations: int,
        instruction: str = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate instruction rephrase variations.
        
        Args:
            base_prompt: Original prompt
            num_variations: Number of variations to generate
            instruction: The instruction part to rephrase
        
        Returns:
            List of variation parameters
        """
        variations = []
        patterns = list(self.REPHRASE_PATTERNS.items())
        
        for i, (style, pattern) in enumerate(patterns[:num_variations]):
            variations.append({
                "style": style,
                "pattern": pattern,
                "description": f"Instruction style: {style}"
            })
        
        return variations

class FewShotSelectionStrategy(BaseVariationStrategy):
    """Generate few-shot example selection variations."""
    
    @property
    def strategy_type(self) -> VariationStrategy:
        return VariationStrategy.FEW_SHOT_SELECTION
    
    def generate_variations(
        self,
        base_prompt: str,
        num_variations: int,
        examples: List[Dict[str, Any]] = None,
        examples_per_prompt: int = 3,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate variations with different example selections.
        
        Args:
            base_prompt: Original prompt
            num_variations: Number of variations
            examples: Pool of available examples
            examples_per_prompt: How many examples per variation
        """
        if not examples or len(examples) < examples_per_prompt:
            return []
        
        variations = []
        
        # Generate combinations
        combinations = list(itertools.combinations(
            range(len(examples)), 
            examples_per_prompt
        ))
        
        # Select diverse combinations
        selected = random.sample(
            combinations, 
            min(num_variations, len(combinations))
        )
        
        for i, combo in enumerate(selected):
            variations.append({
                "example_indices": list(combo),
                "examples": [examples[j] for j in combo],
                "description": f"Example set {i+1}: indices {combo}"
            })
        
        return variations

class FewShotOrderStrategy(BaseVariationStrategy):
    """Generate few-shot example ordering variations."""
    
    @property
    def strategy_type(self) -> VariationStrategy:
        return VariationStrategy.FEW_SHOT_ORDER
    
    ORDER_STRATEGIES = {
        "original": lambda examples: examples,
        "reversed": lambda examples: list(reversed(examples)),
        "by_length_asc": lambda examples: sorted(examples, key=lambda x: len(str(x))),
        "by_length_desc": lambda examples: sorted(examples, key=lambda x: len(str(x)), reverse=True),
        "random": lambda examples: random.sample(examples, len(examples))
    }
    
    def generate_variations(
        self,
        base_prompt: str,
        num_variations: int,
        examples: List[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate variations with different example orderings."""
        if not examples:
            return []
        
        variations = []
        strategies = list(self.ORDER_STRATEGIES.items())
        
        for strategy_name, order_fn in strategies[:num_variations]:
            ordered = order_fn(examples.copy())
            variations.append({
                "order_strategy": strategy_name,
                "ordered_examples": ordered,
                "description": f"Example order: {strategy_name}"
            })
        
        return variations

class OutputFormatStrategy(BaseVariationStrategy):
    """Generate output format variations."""
    
    @property
    def strategy_type(self) -> VariationStrategy:
        return VariationStrategy.OUTPUT_FORMAT
    
    FORMAT_TEMPLATES = {
        "json": "Respond with a JSON object containing: {{ fields }}",
        "markdown": "Format your response using markdown with headers and bullet points.",
        "plain": "Provide a plain text response.",
        "structured": """Structure your response as:
1. Summary: [brief summary]
2. Details: [detailed explanation]
3. Conclusion: [final thoughts]""",
        "xml": "Respond with XML tags: <response><summary>...</summary><details>...</details></response>",
        "table": "Present your findings in a table format where applicable."
    }
    
    def generate_variations(
        self,
        base_prompt: str,
        num_variations: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate output format variations."""
        variations = []
        formats = list(self.FORMAT_TEMPLATES.items())
        
        for format_name, template in formats[:num_variations]:
            variations.append({
                "format_type": format_name,
                "format_instruction": template,
                "description": f"Output format: {format_name}"
            })
        
        return variations

class ChainOfThoughtStrategy(BaseVariationStrategy):
    """Generate chain-of-thought variations."""
    
    @property
    def strategy_type(self) -> VariationStrategy:
        return VariationStrategy.COT_STYLE
    
    COT_STYLES = {
        "none": "",
        "simple": "Think step by step.",
        "explicit_steps": """Let's approach this step by step:
1. First, identify the key elements
2. Analyze each element
3. Synthesize findings
4. Formulate response""",
        "reasoning_trace": "Show your reasoning process before giving the final answer.",
        "pros_cons": "Consider the pros and cons before reaching a conclusion.",
        "question_decomposition": "Break down this question into sub-questions and address each.",
        "analogical": "Think of analogous situations to help reason about this."
    }
    
    def generate_variations(
        self,
        base_prompt: str,
        num_variations: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate CoT style variations."""
        variations = []
        styles = list(self.COT_STYLES.items())
        
        for style_name, cot_text in styles[:num_variations]:
            variations.append({
                "cot_style": style_name,
                "cot_instruction": cot_text,
                "description": f"CoT style: {style_name}"
            })
        
        return variations

class SystemPromptStrategy(BaseVariationStrategy):
    """Generate system prompt variations."""
    
    @property
    def strategy_type(self) -> VariationStrategy:
        return VariationStrategy.SYSTEM_PROMPT
    
    SYSTEM_PROMPTS = {
        "minimal": "You are a helpful assistant.",
        "expert": "You are an expert assistant with deep knowledge in the relevant domain. Provide accurate, detailed responses.",
        "concise": "You are a concise assistant. Provide brief, to-the-point responses without unnecessary elaboration.",
        "thorough": "You are a thorough assistant. Consider all aspects of the question and provide comprehensive responses.",
        "careful": "You are a careful assistant. Double-check your reasoning and acknowledge uncertainty when present.",
        "creative": "You are a creative assistant. Think outside the box while remaining accurate.",
        "professional": "You are a professional assistant. Maintain a formal tone and structure in your responses."
    }
    
    def generate_variations(
        self,
        base_prompt: str,
        num_variations: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate system prompt variations."""
        variations = []
        prompts = list(self.SYSTEM_PROMPTS.items())
        
        for style, system_prompt in prompts[:num_variations]:
            variations.append({
                "system_style": style,
                "system_prompt": system_prompt,
                "description": f"System prompt: {style}"
            })
        
        return variations


class VariationGenerator:
    """Main class for generating prompt variations."""
    
    def __init__(self):
        self.strategies: Dict[VariationStrategy, BaseVariationStrategy] = {
            VariationStrategy.INSTRUCTION_REPHRASE: InstructionRephraseStrategy(),
            VariationStrategy.FEW_SHOT_SELECTION: FewShotSelectionStrategy(),
            VariationStrategy.FEW_SHOT_ORDER: FewShotOrderStrategy(),
            VariationStrategy.OUTPUT_FORMAT: OutputFormatStrategy(),
            VariationStrategy.COT_STYLE: ChainOfThoughtStrategy(),
            VariationStrategy.SYSTEM_PROMPT: SystemPromptStrategy(),
        }
    
    def generate(
        self,
        base_prompt: str,
        strategies: List[VariationStrategy],
        variations_per_strategy: int = 3,
        combine_strategies: bool = False,
        **kwargs
    ) -> VariationSet:
        """
        Generate prompt variations.
        
        Args:
            base_prompt: Original prompt to vary
            strategies: Which variation strategies to use
            variations_per_strategy: How many variations per strategy
            combine_strategies: Whether to create combined variations
            **kwargs: Strategy-specific parameters
        
        Returns:
            VariationSet with all generated variations
        """
        all_variations = []
        
        for strategy in strategies:
            if strategy not in self.strategies:
                continue
            
            strategy_impl = self.strategies[strategy]
            variations = strategy_impl.generate_variations(
                base_prompt,
                variations_per_strategy,
                **kwargs
            )
            
            for i, var_params in enumerate(variations):
                variation = PromptVariation(
                    id=f"{strategy.value}_{i}",
                    base_template_id="base",
                    strategy=strategy,
                    variation_params=var_params,
                    prompt_content=self._apply_variation(base_prompt, strategy, var_params),
                    system_prompt=var_params.get("system_prompt"),
                    description=var_params.get("description", "")
                )
                all_variations.append(variation)
        
        if combine_strategies and len(strategies) > 1:
            combined = self._generate_combined_variations(
                base_prompt, all_variations, variations_per_strategy
            )
            all_variations.extend(combined)
        
        return VariationSet(
            id=f"varset_{hash(base_prompt) % 10000}",
            name="Generated Variations",
            base_template_id="base",
            variations=all_variations,
            strategies_used=strategies,
            created_at=datetime.utcnow().isoformat()
        )
    
    def _apply_variation(
        self,
        base_prompt: str,
        strategy: VariationStrategy,
        params: Dict[str, Any]
    ) -> str:
        """Apply variation parameters to create new prompt."""
        if strategy == VariationStrategy.COT_STYLE:
            cot = params.get("cot_instruction", "")
            if cot:
                return f"{base_prompt}\n\n{cot}"
            return base_prompt
        
        if strategy == VariationStrategy.OUTPUT_FORMAT:
            format_inst = params.get("format_instruction", "")
            if format_inst:
                return f"{base_prompt}\n\n{format_inst}"
            return base_prompt
        
        # For other strategies, return base (actual application happens at render time)
        return base_prompt
    
    def _generate_combined_variations(
        self,
        base_prompt: str,
        single_variations: List[PromptVariation],
        max_combinations: int
    ) -> List[PromptVariation]:
        """Generate combinations of variations."""
        # Group by strategy
        by_strategy = {}
        for var in single_variations:
            if var.strategy not in by_strategy:
                by_strategy[var.strategy] = []
            by_strategy[var.strategy].append(var)
        
        if len(by_strategy) < 2:
            return []
        
        # Generate combinations
        strategy_lists = list(by_strategy.values())
        combinations = list(itertools.product(*strategy_lists[:3]))  # Limit to 3 strategies
        
        combined = []
        for combo in combinations[:max_combinations]:
            combined_params = {}
            descriptions = []
            for var in combo:
                combined_params.update(var.variation_params)
                descriptions.append(var.description)
            
            combined.append(PromptVariation(
                id=f"combined_{len(combined)}",
                base_template_id="base",
                strategy=VariationStrategy.INSTRUCTION_REPHRASE,  # Placeholder
                variation_params=combined_params,
                prompt_content=base_prompt,  # Would need full rendering
                system_prompt=combined_params.get("system_prompt"),
                description=" + ".join(descriptions)
            ))
        
        return combined

Output the complete file.
```

---

## PROMPT 3B.4: A/B Testing Framework

```
Create the A/B testing framework for prompt experiments.

File: src/prompt_optimizer/experiments/ab_testing.py

Requirements:
1. Proper experimental design
2. Random assignment
3. Sample size calculation
4. Multiple test management
5. Results collection

Implementation:

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from datetime import datetime
import random
import uuid
import json
from pathlib import Path

class ExperimentStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

@dataclass
class ExperimentVariant:
    """A variant in an A/B test."""
    id: str
    name: str
    prompt_variation: 'PromptVariation'
    is_control: bool = False
    
@dataclass
class ExperimentResult:
    """Result of a single test case in an experiment."""
    variant_id: str
    test_case_id: str
    response: str
    metrics: Dict[str, float]
    latency_ms: float
    cost_usd: float
    timestamp: str
    raw_response: Optional[Dict[str, Any]] = None

@dataclass
class Experiment:
    """An A/B test experiment."""
    id: str
    name: str
    description: str
    variants: List[ExperimentVariant]
    test_dataset_id: str
    metrics: List[str]
    status: ExperimentStatus
    config: Dict[str, Any]
    results: List[ExperimentResult] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    @property
    def sample_size_per_variant(self) -> int:
        """Get current sample size per variant."""
        counts = {}
        for result in self.results:
            counts[result.variant_id] = counts.get(result.variant_id, 0) + 1
        return min(counts.values()) if counts else 0

class ABTestingFramework:
    """Framework for running A/B tests on prompts."""
    
    def __init__(
        self,
        storage_path: str = "./experiments",
        evaluator: 'EvaluationRunner' = None
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.evaluator = evaluator
        self._experiments: Dict[str, Experiment] = {}
        self._load_experiments()
    
    def create_experiment(
        self,
        name: str,
        description: str,
        control_prompt: 'PromptVariation',
        treatment_prompts: List['PromptVariation'],
        test_dataset_id: str,
        metrics: List[str],
        config: Optional[Dict[str, Any]] = None
    ) -> Experiment:
        """
        Create a new A/B test experiment.
        
        Args:
            name: Experiment name
            description: What we're testing
            control_prompt: The baseline prompt (A)
            treatment_prompts: The variations to test (B, C, ...)
            test_dataset_id: Dataset to use for testing
            metrics: Which metrics to evaluate
            config: Additional configuration
        
        Returns:
            Created Experiment
        """
        experiment_id = str(uuid.uuid4())[:8]
        
        # Create variants
        variants = [
            ExperimentVariant(
                id=f"{experiment_id}_control",
                name="Control",
                prompt_variation=control_prompt,
                is_control=True
            )
        ]
        
        for i, treatment in enumerate(treatment_prompts):
            variants.append(ExperimentVariant(
                id=f"{experiment_id}_treatment_{i}",
                name=f"Treatment {i+1}",
                prompt_variation=treatment,
                is_control=False
            ))
        
        experiment = Experiment(
            id=experiment_id,
            name=name,
            description=description,
            variants=variants,
            test_dataset_id=test_dataset_id,
            metrics=metrics,
            status=ExperimentStatus.DRAFT,
            config=config or {
                "min_sample_size": 30,
                "significance_level": 0.05,
                "random_seed": 42
            }
        )
        
        self._experiments[experiment_id] = experiment
        self._save_experiment(experiment)
        
        return experiment
    
    def start_experiment(self, experiment_id: str) -> Experiment:
        """Start an experiment."""
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        if experiment.status != ExperimentStatus.DRAFT:
            raise ValueError(f"Cannot start experiment in status: {experiment.status}")
        
        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.utcnow().isoformat()
        self._save_experiment(experiment)
        
        return experiment
    
    async def run_experiment(
        self,
        experiment_id: str,
        test_cases: List['TestCase'],
        llm_provider: 'LLMProvider'
    ) -> Experiment:
        """
        Run an experiment to completion.
        
        Args:
            experiment_id: Experiment to run
            test_cases: Test cases from the dataset
            llm_provider: LLM provider to use
        
        Returns:
            Completed experiment with results
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        if experiment.status != ExperimentStatus.RUNNING:
            self.start_experiment(experiment_id)
            experiment = self._experiments[experiment_id]
        
        min_samples = experiment.config.get("min_sample_size", 30)
        random.seed(experiment.config.get("random_seed", 42))
        
        # Run until we have enough samples
        for test_case in test_cases:
            # Random assignment to variant
            variant = random.choice(experiment.variants)
            
            # Execute
            result = await self._execute_variant(
                variant,
                test_case,
                llm_provider,
                experiment.metrics
            )
            
            experiment.results.append(result)
            
            # Check if we have enough samples
            if experiment.sample_size_per_variant >= min_samples:
                break
        
        # Check completion
        if experiment.sample_size_per_variant >= min_samples:
            experiment.status = ExperimentStatus.COMPLETED
            experiment.completed_at = datetime.utcnow().isoformat()
        
        self._save_experiment(experiment)
        return experiment
    
    async def _execute_variant(
        self,
        variant: ExperimentVariant,
        test_case: 'TestCase',
        llm_provider: 'LLMProvider',
        metrics: List[str]
    ) -> ExperimentResult:
        """Execute a single variant on a test case."""
        import time
        
        prompt = variant.prompt_variation.prompt_content
        system_prompt = variant.prompt_variation.system_prompt
        
        start_time = time.perf_counter()
        response = await llm_provider.generate(
            prompt=prompt,
            system_prompt=system_prompt
        )
        latency = (time.perf_counter() - start_time) * 1000
        
        # Evaluate metrics
        metric_results = {}
        if self.evaluator:
            for metric_name in metrics:
                metric = create_metric(metric_name)
                result = await metric.evaluate(
                    response.content,
                    test_case.expected,
                    test_case.context
                )
                metric_results[metric_name] = result.value
        
        return ExperimentResult(
            variant_id=variant.id,
            test_case_id=test_case.id,
            response=response.content,
            metrics=metric_results,
            latency_ms=latency,
            cost_usd=response.cost_usd,
            timestamp=datetime.utcnow().isoformat(),
            raw_response={"tokens": response.total_tokens}
        )
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        return self._experiments.get(experiment_id)
    
    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None
    ) -> List[Experiment]:
        """List all experiments, optionally filtered by status."""
        experiments = list(self._experiments.values())
        if status:
            experiments = [e for e in experiments if e.status == status]
        return sorted(experiments, key=lambda e: e.created_at, reverse=True)
    
    def pause_experiment(self, experiment_id: str) -> Experiment:
        """Pause a running experiment."""
        experiment = self._experiments.get(experiment_id)
        if experiment and experiment.status == ExperimentStatus.RUNNING:
            experiment.status = ExperimentStatus.PAUSED
            self._save_experiment(experiment)
        return experiment
    
    def cancel_experiment(self, experiment_id: str) -> Experiment:
        """Cancel an experiment."""
        experiment = self._experiments.get(experiment_id)
        if experiment:
            experiment.status = ExperimentStatus.CANCELLED
            self._save_experiment(experiment)
        return experiment
    
    def _save_experiment(self, experiment: Experiment):
        """Save experiment to disk."""
        file_path = self.storage_path / f"{experiment.id}.json"
        with open(file_path, 'w') as f:
            # Convert to serializable format
            data = {
                "id": experiment.id,
                "name": experiment.name,
                "description": experiment.description,
                "variants": [
                    {
                        "id": v.id,
                        "name": v.name,
                        "is_control": v.is_control,
                        "prompt_content": v.prompt_variation.prompt_content,
                        "system_prompt": v.prompt_variation.system_prompt
                    }
                    for v in experiment.variants
                ],
                "test_dataset_id": experiment.test_dataset_id,
                "metrics": experiment.metrics,
                "status": experiment.status.value,
                "config": experiment.config,
                "results": [
                    {
                        "variant_id": r.variant_id,
                        "test_case_id": r.test_case_id,
                        "response": r.response,
                        "metrics": r.metrics,
                        "latency_ms": r.latency_ms,
                        "cost_usd": r.cost_usd,
                        "timestamp": r.timestamp
                    }
                    for r in experiment.results
                ],
                "created_at": experiment.created_at,
                "started_at": experiment.started_at,
                "completed_at": experiment.completed_at
            }
            json.dump(data, f, indent=2)
    
    def _load_experiments(self):
        """Load all experiments from disk."""
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                    # Reconstruct experiment (simplified)
                    experiment = Experiment(
                        id=data["id"],
                        name=data["name"],
                        description=data["description"],
                        variants=[],  # Would need full reconstruction
                        test_dataset_id=data["test_dataset_id"],
                        metrics=data["metrics"],
                        status=ExperimentStatus(data["status"]),
                        config=data["config"],
                        created_at=data["created_at"],
                        started_at=data.get("started_at"),
                        completed_at=data.get("completed_at")
                    )
                    self._experiments[experiment.id] = experiment
            except Exception as e:
                print(f"Failed to load experiment {file_path}: {e}")

Output the complete file.
```

---

## PROMPT 3B.5: Statistical Analysis Module

```
Create the statistical analysis module for experiment results.

File: src/prompt_optimizer/statistics/analyzer.py

Requirements:
1. Proper statistical tests (t-test, Mann-Whitney)
2. Multiple comparison correction
3. Effect size calculation
4. Confidence intervals
5. Power analysis
6. Sample size recommendations

Implementation:

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

Output the complete file with all statistical methods.
```

---

## PROMPT 3B.6: Best Prompt Selector

```
Create the best prompt selection module.

File: src/prompt_optimizer/selection/selector.py

Requirements:
1. Multi-metric selection criteria
2. Weighted scoring
3. Confidence-based selection
4. Selection explanation

Implementation:

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
        lines.append(f"Weighted score: {best_data['weighted_score']:.3f}")
        
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

Output the complete file.
```

---

## PROMPT 3B.7: API Endpoints

```
Create the API endpoints for PromptOptimizer.

File: src/prompt_optimizer/api/endpoints.py

Endpoints:

POST /templates - Create a new prompt template
GET /templates - List all templates
GET /templates/{name} - Get template with version history
POST /templates/{name}/render - Render template with variables

POST /variations/generate - Generate prompt variations
GET /variations/{set_id} - Get variation set

POST /experiments - Create new experiment
GET /experiments - List experiments
GET /experiments/{id} - Get experiment details
POST /experiments/{id}/start - Start experiment
POST /experiments/{id}/run - Run experiment
GET /experiments/{id}/results - Get raw results
GET /experiments/{id}/analysis - Get statistical analysis
POST /experiments/{id}/select-best - Select best prompt

GET /sample-size-calculator - Calculate required sample size

Include:
- Pydantic models for all requests/responses
- Proper error handling
- Input validation

Output the complete file.
```

---

## PROMPT 3B.8: Dashboard

```
Create the Streamlit dashboard for PromptOptimizer.

File: src/prompt_optimizer/dashboard/app.py

Features:
1. Template management UI
2. Variation generator UI
3. Experiment creation wizard
4. Live experiment monitoring
5. Results visualization
6. Statistical analysis display
7. Best prompt selection UI

Visualizations:
- Metric comparison bar charts
- Confidence interval plots
- Effect size forest plot
- Sample size progress
- Power analysis chart

Sections:
1. Sidebar: Navigation, quick actions
2. Templates: Create, view, compare versions
3. Experiments: Create, monitor, analyze
4. Results: Charts, statistics, recommendations
5. Selection: Best prompt with explanation

Output the complete file with professional styling.
```

---

## PROMPT 3B.9: Tests and Documentation

```
Create tests and documentation for PromptOptimizer.

Files:
1. tests/test_prompt_optimizer.py
2. README.md (for PromptOptimizer module)

Tests:
1. Template management
   - Create, version, render
   - Variable extraction
   - Validation

2. Variation generation
   - Each strategy type
   - Combined variations
   - Reproducibility

3. A/B testing
   - Experiment creation
   - Result collection
   - Status management

4. Statistical analysis
   - T-test calculations
   - Effect size
   - Power analysis
   - Sample size calculation
   - Multiple comparison correction

5. Selection
   - Weighted scoring
   - Confidence calculation
   - Explanation generation

README should include:
1. Overview
2. Key concepts (A/B testing, statistical significance)
3. Quick start
4. Template system usage
5. Running experiments
6. Interpreting results
7. Best practices for prompt optimization
8. API documentation

Output all files.
```

---

# SUMMARY

## Project 3B Components

| Component | File | Purpose |
|-----------|------|---------|
| Template Manager | `templates/template_manager.py` | Version-controlled prompt templates |
| Variation Generator | `variations/variation_generator.py` | Systematic prompt variations |
| A/B Testing | `experiments/ab_testing.py` | Experimental framework |
| Statistics | `statistics/analyzer.py` | Statistical analysis |
| Selector | `selection/selector.py` | Best prompt selection |
| API | `api/endpoints.py` | REST endpoints |
| Dashboard | `dashboard/app.py` | Streamlit UI |

## Key Features

1. **Template System**: Version-controlled Jinja2 templates
2. **6 Variation Strategies**: Instruction, few-shot, format, CoT, system prompt, emphasis
3. **Proper Statistics**: t-tests, effect sizes, power analysis, Bonferroni correction
4. **Multi-Metric Selection**: Weighted scoring across metrics
5. **Confidence Scores**: Know how certain you are about selections

## Usage Example

```python
# 1. Create template
template = manager.create_template(
    name="qa_prompt",
    template_string="Answer: {{ question }}\nContext: {{ context }}"
)

# 2. Generate variations
variations = generator.generate(
    base_prompt="Answer the question based on context.",
    strategies=[VariationStrategy.COT_STYLE, VariationStrategy.OUTPUT_FORMAT],
    variations_per_strategy=3
)

# 3. Create and run experiment
experiment = framework.create_experiment(
    name="QA Prompt Optimization",
    control_prompt=variations.variations[0],
    treatment_prompts=variations.variations[1:],
    test_dataset_id="qa_test",
    metrics=["accuracy", "latency"]
)
await framework.run_experiment(experiment.id, test_cases, llm_provider)

# 4. Analyze results
analysis = analyzer.analyze_experiment(experiment, "accuracy")

# 5. Select best
result = selector.select_best(experiment, criteria)
print(f"Best prompt: {result.selected_variant_id}")
print(f"Confidence: {result.confidence_score:.2%}")
```

## Time Estimate

| Component | Days |
|-----------|------|
| Template System | 0.5 |
| Variation Generator | 1 |
| A/B Testing Framework | 1.5 |
| Statistical Analysis | 1.5 |
| Best Prompt Selector | 0.5 |
| API + Dashboard | 1 |
| Tests + Docs | 0.5 |
| **Total** | **5-7 days** |
