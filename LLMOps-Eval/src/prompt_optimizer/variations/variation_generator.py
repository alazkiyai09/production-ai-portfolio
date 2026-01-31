"""
Prompt variation generator for systematic prompt optimization.

This module provides strategies for generating systematic prompt variations
for A/B testing with reproducibility and lineage tracking.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Set
from enum import Enum
import itertools
import random
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime
import logging
import re

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class VariationStrategy(Enum):
    """Types of variation strategies."""

    INSTRUCTION_REPHRASE = "instruction_rephrase"
    FEW_SHOT_SELECTION = "few_shot_selection"
    FEW_SHOT_ORDER = "few_shot_order"
    OUTPUT_FORMAT = "output_format"
    COT_STYLE = "cot_style"
    SYSTEM_PROMPT = "system_prompt"
    EMPHASIS = "emphasis"
    VERBOSITY = "verbosity"
    TEMPERATURE = "temperature"
    TOP_P = "top_p"
    CONTEXT_POSITION = "context_position"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PromptVariation:
    """
    A single prompt variation.

    Attributes:
        id: Unique variation identifier
        base_template_id: ID of the source template
        strategy: Which variation strategy was used
        variation_params: Parameters for this variation
        prompt_content: The actual prompt content
        system_prompt: Optional system prompt override
        description: Human-readable description
        parent_variations: List of parent variation IDs (for combined strategies)
        generation_seed: Seed used for generation
        metadata: Additional metadata
    """

    id: str
    base_template_id: str
    strategy: VariationStrategy
    variation_params: Dict[str, Any]
    prompt_content: str
    system_prompt: Optional[str] = None
    description: str = ""
    parent_variations: List[str] = field(default_factory=list)
    generation_seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def hash(self) -> str:
        """Generate hash for variation content."""
        content = self.prompt_content + (self.system_prompt or "")
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "base_template_id": self.base_template_id,
            "strategy": self.strategy.value,
            "variation_params": self.variation_params,
            "prompt_content": self.prompt_content,
            "system_prompt": self.system_prompt,
            "description": self.description,
            "parent_variations": self.parent_variations,
            "hash": self.hash,
            "generation_seed": self.generation_seed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptVariation":
        """Create PromptVariation from dictionary."""
        strategy = VariationStrategy(data["strategy"])
        return cls(
            id=data["id"],
            base_template_id=data["base_template_id"],
            strategy=strategy,
            variation_params=data["variation_params"],
            prompt_content=data["prompt_content"],
            system_prompt=data.get("system_prompt"),
            description=data.get("description", ""),
            parent_variations=data.get("parent_variations", []),
            generation_seed=data.get("generation_seed"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class VariationSet:
    """
    A set of related variations for A/B testing.

    Attributes:
        id: Unique variation set ID
        name: Human-readable name
        base_template_id: Source template
        variations: List of PromptVariation objects
        strategies_used: List of strategies applied
        created_at: Creation timestamp
        metadata: Additional metadata
    """

    id: str
    name: str
    base_template_id: str
    variations: List[PromptVariation] = field(default_factory=list)
    strategies_used: List[VariationStrategy] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def variation_count(self) -> int:
        """Get total number of variations."""
        return len(self.variations)

    @property
    def strategies(self) -> Set[VariationStrategy]:
        """Get set of strategies used."""
        return set(self.strategies_used)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "base_template_id": self.base_template_id,
            "variations": [v.to_dict() for v in self.variations],
            "strategies_used": [s.value for s in self.strategies_used],
            "variation_count": self.variation_count,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VariationSet":
        """Create VariationSet from dictionary."""
        strategies = [VariationStrategy(s) for s in data["strategies_used"]]
        variations = [PromptVariation.from_dict(v) for v in data.get("variations", [])]

        return cls(
            id=data["id"],
            name=data["name"],
            base_template_id=data["base_template_id"],
            variations=variations,
            strategies_used=strategies,
            created_at=data["created_at"],
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# Base Strategy Class
# ============================================================================

class BaseVariationStrategy(ABC):
    """
    Abstract base class for variation strategies.

    All strategies must inherit from this class and implement
    the generate_variations method.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize strategy with reproducible randomness.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self._random = random.Random(seed)

    @property
    @abstractmethod
    def strategy_type(self) -> VariationStrategy:
        """Get the strategy type enum value."""
        pass

    @abstractmethod
    def generate_variations(
        self,
        base_prompt: str,
        num_variations: int,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Generate variation parameters.

        Args:
            base_prompt: Original prompt
            num_variations: Number of variations to generate
            **kwargs: Strategy-specific parameters

        Returns:
            List of variation parameter dictionaries
        """
        pass

    def apply_variation(
        self,
        base_prompt: str,
        params: Dict[str, Any],
        variables: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Apply a variation to a base prompt.

        Args:
            base_prompt: Original prompt content
            params: Variation parameters from generate_variations
            variables: Optional variables for template rendering

        Returns:
            Modified prompt content
        """
        raise NotImplementedError("Subclasses must implement apply_variation")


# ============================================================================
# Strategy Implementations
# ============================================================================

class InstructionRephraseStrategy(BaseVariationStrategy):
    """
    Generate instruction rephrasing variations.

    Creates variations by changing how instructions are phrased
    while maintaining the core task.
    """

    # Rephrasing patterns
    REPHRASE_TEMPLATES = {
        "direct": "{{ instruction }}",
        "polite": "Please {{ instruction|lower }}.",
        "imperative": "You must {{ instruction|lower }}.",
        "question": "Can you {{ instruction|lower }}?",
        "task_framing": "Your task is to: {{ instruction|lower }}.",
        "role_expert": "As an expert in this domain, {{ instruction|lower }}.",
        "outcome_focused": "Provide a response that {{ instruction|lower }}.",
        "step_by_step": "Step by step, {{ instruction|lower }}.",
        "concise": "{{ instruction|lower }} (be brief).",
        "detailed": "{{ instruction|lower }} (provide detailed explanation).",
    }

    # Synonyms for common instruction words
    SYNONYMS = {
        "explain": [
            "describe", "clarify", "elaborate", "break down", "outline",
            "analyze", "examine", "evaluate", "assess", "investigate", "study",
        ],
        "write": [
            "compose", "create", "draft", "generate", "produce",
            "compose, "author", "craft",
        ],
        "list": [
            "enumerate", "itemize", "name", "specify", "detail",
            "provide", "give", "state",
        ],
    }

    @property
    def strategy_type(self) -> VariationStrategy:
        return VariationStrategy.INSTRUCTION_REPHRASE

    def generate_variations(
        self,
        base_prompt: str,
        num_variations: int,
        instruction: Optional[str] = None,
        rephrasing_styles: Optional[List[str]] = None,
        use_synonyms: bool = True,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Generate instruction rephrasing variations.

        Args:
            base_prompt: Original prompt
            num_variations: Number of variations
            instruction: The core instruction to rephrase
            rephrasing_styles: Which styles to use (None = all)
            use_synonyms: Whether to use synonym substitution

        Returns:
            List of variation parameter dictionaries
        """
        variations = []

        # If instruction not provided, try to extract it
        if not instruction:
            instruction = self._extract_instruction(base_prompt)

        # Select styles
        if rephrasing_styles is None:
            rephrasing_styles = list(self.REPHRASE_TEMPLATES.keys())[:num_variations]
        else:
            rephrasing_styles = [s for s in rephrasing_styles if s in self.REPHRASE_TEMPLATES]

        # Generate template variations
        for i, style in enumerate(rephrasing_styles):
            template = self.REPHRASE_TEMPLATES[style]
            variations.append({
                "rephrase_style": style,
                "template": template,
                "instruction": instruction,
                "description": f"Instruction rephrasing: {style}",
                "position": i,
            })

        # Add synonym variations if requested
        if use_synonyms and instruction:
            synonym_vars = self._generate_synonym_variations(instruction, num_variations // 2)
            variations.extend(synonym_vars)

        return variations

    def apply_variation(
        self,
        base_prompt: str,
        params: Dict[str, Any],
        variables: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Apply rephrasing variation to prompt."""
        template = params.get("template", "")
        instruction = params.get("instruction", "")

        if variables:
            return template.replace("{{ instruction }}", instruction).replace(
                "{{ instruction|lower }}", instruction.lower()
            )

        return template.replace("{{ instruction }}", instruction)

    def _extract_instruction(self, prompt: str) -> str:
        """
        Extract the core instruction from a prompt.

        Args:
            prompt: The prompt text

        Returns:
            Extracted instruction string
        """
        # Remove common prefixes
        prefixes = [
            r"^Please ",
            r"^Can you ",
            r"^I need you to ",
            r"^Help me ",
        ]
        for prefix in prefixes:
            prompt = re.sub(prefix, "", prompt, flags=re.IGNORECASE)

        # Remove punctuation at end
        prompt = prompt.rstrip().rstrip(".")

        # Extract first sentence (usually the instruction)
        first_period = prompt.find(". ")
        if first_period > 0:
            return prompt[:first_period + 1]

        return prompt

    def _generate_synonym_variations(
        self,
        instruction: str,
        num_variations: int,
    ) -> List[Dict[str, Any]]:
        """Generate variations using synonym substitution."""
        variations = []
        words = instruction.lower().split()

        # Find replaceable words
        for word in words:
            if word in self.SYNONYMS:
                for i, synonym in enumerate(self.SYNONMS[word][:num_variations]):
                    variation = instruction.replace(word, synonym, 1)
                    variations.append({
                        "synonym_word": word,
                        "synonym": synonym,
                        "template": variation,
                        "description": f"Synonym: {word} → {synonym}",
                    })

        return variations[:num_variations]


class FewShotSelectionStrategy(BaseVariationStrategy):
    """
    Generate variations by selecting different few-shot examples.

    Tests which subset of examples works best.
    """

    @property
    def strategy_type(self) -> VariationStrategy:
        return VariationStrategy.FEW_SHOT_SELECTION

    def generate_variations(
        self,
        base_prompt: str,
        num_variations: int,
        examples: List[Dict[str, Any]],
        examples_per_prompt: int = 3,
        selection_method: str = "diverse",
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Generate few-shot selection variations.

        Args:
            base_prompt: Original prompt
            num_variations: Number of variations
            examples: Pool of available examples
            examples_per_prompt: How many examples per variation
            selection_method: How to select examples (diverse, random, sequential)

        Returns:
            List of variation parameters
        """
        if not examples or len(examples) < examples_per_prompt:
            logger.warning("Not enough examples for few-shot selection")
            return []

        variations = []
        n_examples = len(examples)

        # Calculate max possible combinations
        from math import comb

        max_combos = comb(n_examples, examples_per_prompt)
        if max_combos < num_variations:
            num_variations = max_combos
            logger.warning(f"Reducing variations to {num_variations} (only {max_combos} combinations possible)")

        # Generate combinations based on method
        if selection_method == "sequential":
            # First N combinations
            combinations = list(itertools.combinations(range(n_examples), examples_per_prompt))
        elif selection_method == "random":
            # Random combinations
            all_combos = list(itertools.combinations(range(n_examples), examples_per_prompt))
            combinations = self._random.sample(all_combos, min(num_variations, len(all_combos)))
        else:  # "diverse"
            combinations = self._generate_diverse_combinations(
                examples, examples_per_prompt, num_variations
            )

        # Create variation objects
        for i, indices in enumerate(combinations[:num_variations]):
            selected = [examples[j] for j in indices]
            variations.append({
                "selection_method": selection_method,
                "example_indices": list(indices),
                "examples": selected,
                "description": f"Example set {i+1}: {[e.get('id', j) for j, e in enumerate(selected)]}",
            })

        return variations

    def _n_combinations(self, n: int, k: int) -> int:
        """Calculate number of combinations."""
        from math import comb
        return comb(n, k)

    def _generate_diverse_combinations(
        self,
        examples: List[Dict[str, Any]],
        k: int,
        num_combinations: int,
    ) -> List[tuple]:
        """
        Generate diverse combinations maximizing example coverage.

        Args:
            examples: Pool of examples
            k: Examples per prompt
            num_combinations: Number of combinations to generate

        Returns:
            List of example index tuples
        """
        # Score each combination by diversity
        all_combos = list(itertools.combinations(range(len(examples)), k))

        def score_combination(indices):
            selected = [examples[i] for i in indices]
            categories = set()
            tags = set()

            for ex in selected:
                categories.update(ex.get("category", []))
                tags.update(ex.get("tags", []))

            # Length variance
            lengths = [len(str(ex.get("input", "") + str(ex.get("output", ""))) for ex in selected]
            length_var = max(lengths) - min(lengths) if lengths else 0

            return len(categories) * 2 + len(tags) + length_var

        # Sort by diversity score (descending)
        all_combos.sort(key=score_combination, reverse=True)

        return all_combos[:num_combinations]


class FewShotOrderStrategy(BaseVariationStrategy):
    """
    Generate variations by reordering few-shot examples.

    Tests whether example order affects performance.
    """

    @property
    def strategy_type(self) -> VariationStrategy:
        return VariationStrategy.FEW_SHOT_ORDER

    ORDERING_STRATEGIES = {
        "original": lambda ex: ex,
        "reversed": lambda ex: list(reversed(ex)),
        "sort_by_input_length": lambda ex: sorted(ex, key=lambda x: len(str(x.get("input", "")))),
        "sort_by_output_length": lambda ex: sorted(ex, key=lambda x: len(str(x.get("output", "")))),
        "sort_by_difficulty": lambda ex: sorted(ex, key=lambda x: x.get("metadata", {}).get("difficulty", 5)),
        "alternate_categories": lambda ex: [],
        "random": lambda ex, seed=seed: random.Random(seed).sample(ex, len(ex)),
    }

    def generate_variations(
        self,
        base_prompt: str,
        num_variations: int,
        examples: List[Dict[str, Any]],
        ordering_strategies: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Generate ordering variations.

        Args:
            base_prompt: Original prompt
            num_variations: Number of variations
            examples: Examples to reorder
            ordering_strategies: Which ordering strategies to use

        Returns:
            List of variation parameters
        """
        if not examples:
            return []

        variations = []

        # Select strategies
        if ordering_strategies is None:
            ordering_strategies = list(self.ORDERING_STRATEGIES.keys())[:num_variations]
        else:
            ordering_strategies = [s for s in ordering_strategies if s in self.ORDERING_STRATEGIES]

        for strategy in ordering_strategies:
            order_fn = self.ORDERING_STRATEGIES.get(strategy)
            ordered = order_fn(examples.copy())
            variations.append({
                "ordering_strategy": strategy,
                "ordered_examples": ordered,
                "description": f"Ordering: {strategy}",
            })

        return variations


class OutputFormatStrategy(BaseVariationStrategy):
    """
    Generate output format variations.

    Tests which output format works best.
    """

    @property
    def strategy_type(self) -> VariationStrategy:
        return VariationStrategy.OUTPUT_FORMAT

    FORMAT_TEMPLATES = {
        "plain": "Provide a direct answer.",
        "paragraph": "Provide your answer as a single paragraph.",
        "bullet_points": "Provide your answer as bullet points.",
        "numbered_list": "Provide your answer as a numbered list.",
        "json": "Respond in JSON format.",
        "json_structured": "Respond in JSON format with proper keys and structure.",
        "xml": "Respond in XML format with appropriate tags.",
        "yaml": "Respond in YAML format.",
        "markdown": "Format your response using markdown.",
        "code": "Provide your response as code.",
        "table": "Present your answer in a table format.",
    }

    def generate_variations(
        self,
        base_prompt: str,
        num_variations: int,
        formats: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Generate output format variations.

        Args:
            base_prompt: Original prompt
            num_variations: Number of variations
            formats: Which formats to test

        Returns:
            List of variation parameters
        """
        if formats is None:
            formats = list(self.FORMAT_TEMPLATES.keys())[:num_variations]
        else:
            formats = [f for f in formats if f in self.FORMAT_TEMPLATES]

        variations = []
        for fmt in formats:
            template = self.FORMAT_TEMPLATES[fmt]
            variations.append({
                "output_format": fmt,
                "format_instruction": template,
                "description": f"Output format: {fmt}",
            })

        return variations

    def apply_variation(
        self,
        base_prompt: str,
        params: Dict[str, Any],
        variables: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Apply output format variation to prompt."""
        format_inst = params.get("format_instruction", "")
        if format_inst:
            return f"{base_prompt}\n\n{format_inst}"
        return base_prompt


class ChainOfThoughtStrategy(BaseVariationStrategy):
    """
    Generate chain-of-thought variations.

    Tests different CoT prompting styles.
    """

    @property
    def strategy_type(self) -> VariationStrategy:
        return VariationStrategy.COT_STYLE

    COT_TEMPLATES = {
        "none": "",
        "simple": "Let's think step by step.",
        "explicit_steps": """Let's approach this systematically:
1. Understand the question
2. Gather relevant information
3. Consider different perspectives
4. Formulate the answer""",
        "reasoning_trace": "Show your work. Explain your reasoning step by step before giving the final answer.",
        "verify_then_answer": "First verify the premises, then provide your answer.",
        "pros_cons": "Consider the pros and cons before concluding.",
        "decomposition": "Break this down into smaller parts and address each.",
        "analogy": "Use an analogy to help explain the concept.",
        "two_hopes": "First attempt one approach, then try another if needed.",
        "confidence_interval": "Provide your answer with a confidence score (0-100%)",
        "summary_first": "Start with a brief summary, then explain in detail.",
    }

    def generate_variations(
        self,
        base_prompt: str,
        num_variations: int,
        cot_styles: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Generate CoT variations.

        Args:
            base_prompt: Original prompt
            num_variations: Number of variations
            cot_styles: Which CoT styles to use

        Returns:
            List of variation parameters
        """
        if cot_styles is None:
            cot_styles = list(self.COT_TEMPLATES.keys())[:num_variations]
        else:
            cot_styles = [s for s in cot_styles if s in self.COT_TEMPLATES]

        variations = []
        for style in cot_styles:
            template = self.COT_TEMPLATES[style]
            variations.append({
                "cot_style": style,
                "cot_instruction": template,
                "description": f"CoT style: {style}",
            })

        return variations

    def apply_variation(
        self,
        base_prompt: str,
        params: Dict[str, Any],
        variables: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Apply CoT variation to prompt."""
        cot_instruction = params.get("cot_instruction", "")
        if cot_instruction:
            return f"{base_prompt}\n\n{cot_instruction}"
        return base_prompt


class SystemPromptStrategy(BaseVariationStrategy):
    """
    Generate system prompt variations.

    Tests different AI persona and behavior settings.
    """

    @property
    def strategy_type(self) -> VariationStrategy:
        return VariationStrategy.SYSTEM_PROMPT

    SYSTEM_PROMPTS = {
        "minimal": "You are a helpful AI assistant.",
        "standard": "You are a helpful, harmless, honest AI assistant.",
        "expert": "You are an expert AI assistant with deep knowledge across many domains.",
        "precise": "You are a precise AI assistant. Provide accurate, detailed responses.",
        "concise": "You are a concise AI assistant. Keep responses brief and to-the-point.",
        "creative": "You are a creative AI assistant. Think outside the box.",
        "analytical": "You are an analytical AI assistant. Show your reasoning process.",
        "teacher": "You are a patient teacher. Explain concepts step by step.",
        "professional": "You are a professional assistant. Maintain a formal tone.",
        "friendly": "You are a friendly AI assistant. Be conversational and helpful.",
        "socratic": "Guide users to answers through questioning rather than direct answers.",
    }

    def generate_variations(
        self,
        base_prompt: str,
        num_variations: int,
        system_prompts: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Generate system prompt variations.

        Args:
            base_prompt: Original prompt
            num_variations: Number of variations
            system_prompts: Which system prompts to test

        Returns:
            List of variation parameters
        """
        if system_prompts is None:
            system_prompts = list(self.SYSTEM_PROMPTS.keys())[:num_variations]
        else:
            system_prompts = [s for s in system_prompts if s in self.SYSTEM_PROMPTS]

        variations = []
        for sp in system_prompts:
            prompt = self.SYSTEM_PROMPTS[sp]
            variations.append({
                "system_style": sp,
                "system_prompt": prompt,
                "description": f"System prompt: {sp}",
            })

        return variations

    def apply_variation(
        self,
        base_prompt: str,
        params: Dict[str, Any],
        variables: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Apply system prompt variation."""
        # System prompts are separate from user prompts
        return base_prompt


class EmphasisStrategy(BaseVariationStrategy):
    """
    Generate emphasis and tone variations.

    Tests how emphasis words affect responses.
    """

    @property
    def strategy_type(self) -> VariationStrategy:
        return VariationStrategy.EMPHASIS

    EMPHASIS_PATTERNS = {
        "none": "",
        "strongly": "It is {{ adjective }} that you {{ verb }}.",
        "very": "It is {{ adjective }} that you {{ verb }}.",
        "quite": "It is {{ adjective }} that you {{ verb }}.",
        "somewhat": "It is {{ adjective }} that you {{ verb }}.",
        "definitely": "You definitely {{ verb }}.",
        "clearly": "You clearly {{ verb }}.",
        "must": "You must {{ verb }}.",
        "should": "You should {{ verb }}.",
        "can": "You can {{ verb }}.",
        "may": "You may {{ verb }}.",
    }

    def generate_variations(
        self,
        base_prompt: str,
        num_variations: int,
        adjectives: Optional[List[str]] = None,
        verbs: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Generate emphasis variations.

        Args:
            base_prompt: Original prompt
            num_variations: Number of variations
            adjectives: Adjectives to test
            verbs: Verbs to test

        Returns:
            List of variation parameters
        """
        if adjectives is None:
            adjectives = ["important", "essential", "crucial", "significant"]
        if verbs is None:
            verbs = ["remember", "note", "ensure", "consider"]

        variations = []
        patterns = list(self.EMPHASIS_PATTERNS.items())

        for i, (style, pattern) in enumerate(patterns[:num_variations]):
            adj = self._random.choice(adjectives)
            verb = self._random.choice(verbs)
            variations.append({
                "emphasis_style": style,
                "template": pattern,
                "adjective": adj,
                "verb": verb,
                "description": f"Emphasis: {style} ({adj} {verb})",
            })

        return variations


class VerbosityStrategy(BaseVariationStrategy):
    """
    Generate verbosity variations.

    Tests response length/detail level.
    """

    @property
    def strategy_type(self) -> VariationStrategy:
        return VariationStrategy.VERBOSITY

    VERBOSITY_LEVELS = {
        "minimal": "Be extremely brief. Answer in one sentence.",
        "concise": "Be concise. Provide short, direct answers.",
        "standard": "Provide clear, complete answers.",
        "detailed": "Provide comprehensive answers with explanations.",
        "thorough": "Provide exhaustive answers covering all aspects.",
        "elaborate": "Provide detailed explanations with examples and context.",
    }

    def generate_variations(
        self,
        base_prompt: str,
        num_variations: int,
        verbosity_levels: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Generate verbosity variations.

        Args:
            base_prompt: Original prompt
            num_variations: Number of variations
            verbosity_levels: Which verbosity levels to test

        Returns:
            List of variation parameters
        """
        if verbosity_levels is None:
            verbosity_levels = list(self.VERBOSITY_LEVELS.keys())[:num_variations]
        else:
            verbosity_levels = [v for v in verbosity_levels if v in self.VERBOSITY_LEVELS]

        variations = []
        for level in verbosity_levels:
            instruction = self.VERBOSITY_LEVELS[level]
            variations.append({
                "verbosity_level": level,
                "instruction": instruction,
                "description": f"Verbosity: {level}",
            })

        return variations

    def apply_variation(
        self,
        base_prompt: str,
        params: Dict[str, Any],
        variables: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Apply verbosity variation to prompt."""
        instruction = params.get("instruction", "")
        if instruction:
            return f"{instruction}\n\n{base_prompt}"
        return base_prompt


# ============================================================================
# Temperature Sweep Strategy
# ============================================================================

class TemperatureSweepStrategy(BaseVariationStrategy):
    """
    Generate temperature parameter variations.

    Tests which temperature setting works best.
    """

    @property
    def strategy_type(self) -> VariationStrategy:
        return VariationStrategy.TEMPERATURE

    def generate_variations(
        self,
        base_prompt: str,
        num_variations: int,
        temp_min: float = 0.0,
        temp_max: float = 1.0,
        steps: int = 5,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Generate temperature variations."""
        from numpy import linspace

        variations = []
        temperatures = list(linspace(temp_min, temp_max, steps))

        for i, temp in enumerate(temperatures):
            variations.append({
                "temperature": temp,
                "description": f"Temperature: {temp:.2f}",
            })

        return variations


class TopPSweepStrategy(BaseVariationStrategy):
    """
    Generate top_p parameter variations.

    Tests nucleus sampling temperature equivalent.
    """

    @property
    def strategy_type(self) -> VariationStrategy:
        return VariationStrategy.TOP_P

    def generate_variations(
        self,
        base_prompt: str,
        num_variations: int,
        top_p_min: float = 0.1,
        top_p_max: float = 1.0,
        steps: int = 5,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Generate top_p variations."""
        import numpy as np

        variations = []
        top_p_values = np.linspace(top_p_min, top_p_max, steps)

        for i, top_p in enumerate(top_p_values):
            variations.append({
                "top_p": top_p,
                "description": f"Top-P: {top_p:.2f}",
            })

        return variations


class ContextPositionStrategy(BaseVariationStrategy):
    """
    Generate context position variations.

    Tests whether context position affects responses.
    """

    @property
    def strategy_type(self) -> VariationStrategy:
        return VariationStrategy.CONTEXT_POSITION

    POSITION_TEMPLATES = {
        "context_first": "{{ context }}\n\n{{ instruction }}",
        "instruction_first": "{{ instruction }}\n\n{{ context }}",
        "context_middle": "{{ instruction }}\n\n{{ context }}\n\n{{ output_format }}",
        "sandwich": "{{ context }}\n\n{{ instruction }}\n\n{{ context }}",
        "inline_context": "{{ instruction }} (Context: {{ context }}) {{ output_format }}",
    }

    def generate_variations(
        self,
        base_prompt: str,
        num_variations: int,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Generate context position variations."""
        variations = []

        for position, template in self.POSITION_TEMPLATES.items():
            variations.append({
                "context_position": position,
                "template": template,
                "description": f"Context position: {position}",
            })

        return variations


# ============================================================================
# Main Variation Generator
# ============================================================================

class VariationGenerator:
    """
    Main class for generating prompt variations.

    Orchestrates multiple variation strategies and creates
    comprehensive variation sets for A/B testing.
    """

    def __init__(
        self,
        seed: int = 42,
        max_variations_per_strategy: int = 10,
        enable_temperature_sweeps: bool = True,
    ):
        """
        Initialize the variation generator.

        Args:
            seed: Random seed for reproducibility
            max_variations_per_strategy: Max variations per strategy
            enable_temperature_sweeps: Enable temperature sweeps
        """
        self.seed = seed
        self.max_variations_per_strategy = max_variations_per_strategy
        self.enable_temperature_sweeps = enable_temperature_sweeps

        # Initialize all strategies
        self.strategies: Dict[VariationStrategy, BaseVariationStrategy] = {
            VariationStrategy.INSTRUCTION_REPHRASE: InstructionRephraseStrategy(seed=seed),
            VariationStrategy.FEW_SHOT_SELECTION: FewShotSelectionStrategy(seed=seed),
            VariationStrategy.FEW_SHOT_ORDER: FewShotOrderStrategy(seed=seed),
            VariationStrategy.OUTPUT_FORMAT: OutputFormatStrategy(),
            VariationStrategy.COT_STYLE: ChainOfThoughtStrategy(),
            VariationStrategy.SYSTEM_PROMPT: SystemPromptStrategy(),
            VariationStrategy.EMPHASIS: EmphasisStrategy(seed=seed),
            VariationStrategy.VERBOSITY: VerbosityStrategy(),
            VariationStrategy.TEMPERATURE: TemperatureSweepStrategy(seed=seed),
            VariationStrategy.TOP_P: TopPSweepStrategy(seed=seed),
            VariationStrategy.CONTEXT_POSITION: ContextPositionStrategy(),
        }

    def generate(
        self,
        base_prompt: str,
        strategies: List[VariationStrategy],
        variations_per_strategy: int = 3,
        combine_strategies: bool = False,
        max_total_variations: int = 50,
        **strategy_kwargs: Dict[str, Any],
    ) -> VariationSet:
        """
        Generate comprehensive prompt variations.

        Args:
            base_prompt: The original prompt to vary
            strategies: Which variation strategies to use
            variations_per_strategy: Variations per strategy
            combine_strategies: Create combined strategy variations
            max_total_variations: Maximum total variations
            **strategy_kwargs: Additional parameters for specific strategies

        Returns:
            VariationSet containing all generated variations
        """
        all_variations: List[PromptVariation] = []
        strategies_used: List[VariationStrategy] = []

        # Generate variations for each strategy
        for strategy in strategies:
            if strategy not in self.strategies:
                logger.warning(f"Unknown strategy: {strategy}, skipping")
                continue

            strategy_impl = self.strategies[strategy]
            try:
                params_list = strategy_impl.generate_variations(
                    base_prompt=base_prompt,
                    num_variations=min(variations_per_strategy, self.max_variations_per_strategy),
                    **strategy_kwargs.get(strategy.value, {}),
                )

                for i, params in enumerate(params_list):
                    # Generate the actual prompt content
                    prompt_content = strategy_impl.apply_variation(
                        base_prompt=base_prompt,
                        params=params,
                    )

                    # Add system prompt if provided
                    system_prompt = params.get("system_prompt")

                    # Create PromptVariation
                    variation = PromptVariation(
                        id=f"{strategy.value}_{i}",
                        base_template_id="base",
                        strategy=strategy,
                        variation_params=params,
                        prompt_content=prompt_content,
                        system_prompt=system_prompt,
                        description=params.get("description", ""),
                        generation_seed=self.seed,
                    )
                    all_variations.append(variation)

                strategies_used.append(strategy)

            except Exception as e:
                logger.error(f"Error generating variations for {strategy}: {e}")

        # Generate combined strategy variations if requested
        if combine_strategies and len(strategies) > 1:
            combined = self._generate_combined_variations(
                base_prompt, strategies, variations_per_strategy, strategy_kwargs
            )
            all_variations.extend(combined)

        # Limit total variations
        if len(all_variations) > max_total_variations:
            all_variations = all_variations[:max_total_variations]

        # Generate metadata
        metadata = {
            "seed": self.seed,
            "strategies_requested": [s.value for s in strategies],
            "total_variations": len(all_variations),
            "combination_enabled": combine_strategies,
            "max_variations_per_strategy": self.max_variations_per_strategy,
        }

        return VariationSet(
            id=f"varset_{hashlib.sha256(base_prompt.encode()).hexdigest()[:12]}",
            name="Generated Variations",
            base_template_id="base",
            variations=all_variations,
            strategies_used=strategies_used,
            created_at=datetime.utcnow().isoformat(),
            metadata=metadata,
        )

    def _generate_combined_variations(
        self,
        base_prompt: str,
        strategies: List[VariationStrategy],
        variations_per_strategy: int,
        strategy_kwargs: Dict[str, Any],
    ) -> List[PromptVariation]:
        """
        Generate combinations of multiple strategies.

        Creates variations that combine 2+ strategies together.
        """
        if len(strategies) < 2:
            return []

        combined = []
        combo_count = 0
        max_combos = min(5, variations_per_strategy)

        # Generate 2-strategy combinations
        for i, strat1 in enumerate(strategies[:-1]):
            for strat2 in strategies[i + 1:]:
                if combo_count >= max_combos:
                    break

                # Get first variation from each strategy
                var1_params = self.strategies[strat1].generate_variations(
                    base_prompt, 1, **strategy_kwargs.get(strat1.value, {})
                )
                var2_params = self.strategies[strat2].generate_variations(
                    base_prompt, 1, **strategy_kwargs.get(strat2.value, {})
                )

                if var1_params and var2_params:
                    # Combine parameters
                    combined_params = {
                        **var1_params[0],
                        **var2_params[0],
                        "combined_strategies": [strat1.value, strat2.value],
                        "description": f"Combined: {strat1.value} + {strat2.value}",
                    }

                    # Apply first variation
                    prompt1 = self.strategies[strat1].apply_variation(
                        base_prompt, var1_params[0]
                    )

                    # Apply second variation
                    prompt2 = self.strategies[strat2].apply_variation(
                        prompt1, var2_params[0]
                    )

                    combined = PromptVariation(
                        id=f"combined_{strat1.value}_{strat2.value}_{combo_count}",
                        base_template_id="base",
                        strategy=VariationStrategy.INSTRUCTION_REPHRASE,  # Placeholder
                        variation_params=combined_params,
                        prompt_content=prompt2,
                        system_prompt=combined_params.get("system_prompt"),
                        description=combined_params["description"],
                        parent_variations=[
                            var1_params[0].get("id", ""),
                            var2_params[0].get("id", "")
                        ],
                        generation_seed=self.seed,
                    )
                    combined.append(combined)
                    combo_count += 1

        return combined

    def get_strategy(
        self,
        strategy: VariationStrategy,
    ) -> Optional[BaseVariationStrategy]:
        """Get a specific strategy implementation."""
        return self.strategies.get(strategy)

    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names."""
        return [s.value for s in VariationStrategy]


# ============================================================================
# Convenience Functions
# ============================================================================

def create_variation_generator(
    seed: int = 42,
    max_variations: int = 10,
    enable_temperature_sweeps: bool = True,
) -> VariationGenerator:
    """
    Factory function to create a variation generator.

    Args:
        seed: Random seed for reproducibility
        max_variations: Max variations per strategy
        enable_temperature_sweeps: Enable temperature sweeps

    Returns:
        Initialized VariationGenerator
    """
    return VariationGenerator(
        seed=seed,
        max_variations=max_variations,
        enable_temperature_sweeps=enable_temperature_sweeps,
    )


# Export main classes and functions
__all__ = [
    # Enums
    "VariationStrategy",
    # Data classes
    "PromptVariation",
    "VariationSet",
    # Strategies
    "InstructionRephraseStrategy",
    "FewShotSelectionStrategy",
    "FewShotOrderStrategy",
    "OutputFormatStrategy",
    "ChainOfThoughtStrategy",
    "SystemPromptStrategy",
    "EmphasisStrategy",
    "VerbosityStrategy",
    "TemperatureSweepStrategy",
    "TopPSweepStrategy",
    "ContextPositionStrategy",
    # Main class
    "VariationGenerator",
    "create_variation_generator",
]
