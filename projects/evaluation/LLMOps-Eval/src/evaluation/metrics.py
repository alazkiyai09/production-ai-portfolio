"""
Evaluation metrics for LLM response assessment.

This module provides a comprehensive suite of metrics for evaluating LLM responses
including accuracy, similarity, hallucination detection, safety checks, latency,
and cost tracking.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, Callable
import re
import json
import logging
import asyncio
from enum import Enum

# For semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
except ImportError:
    SentenceTransformer = None
    cosine_similarity = None
    np = None

# Import LLM providers for judge metrics
from src.models.llm_providers import create_provider, LLMResponse

logger = logging.getLogger(__name__)


class MetricStatus(Enum):
    """Status of metric evaluation."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class MetricResult:
    """
    Result from evaluating a single metric.

    Attributes:
        name: Metric name
        value: Numeric score or value
        details: Additional information about the evaluation
        passed: Whether the metric passed the threshold
        threshold: Optional threshold value for pass/fail
        status: Detailed status (passed, failed, warning, skipped)
        error: Error message if evaluation failed
    """

    name: str
    value: float
    details: dict[str, Any] = field(default_factory=dict)
    passed: bool = True
    threshold: Optional[float] = None
    status: MetricStatus = MetricStatus.PASSED
    error: Optional[str] = None

    def __post_init__(self):
        """Set status based on passed value."""
        if self.error:
            self.status = MetricStatus.FAILED
        elif not self.passed:
            self.status = MetricStatus.FAILED
        else:
            self.status = MetricStatus.PASSED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "details": self.details,
            "passed": self.passed,
            "threshold": self.threshold,
            "status": self.status.value if isinstance(self.status, MetricStatus) else self.status,
            "error": self.error,
        }


class BaseMetric(ABC):
    """
    Abstract base class for evaluation metrics.

    All metrics must inherit from this class and implement the evaluate method.
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        weight: float = 1.0,
        enabled: bool = True,
    ):
        """
        Initialize metric.

        Args:
            threshold: Optional threshold for pass/fail determination
            weight: Weight for aggregating multiple metrics
            enabled: Whether this metric is enabled
        """
        self.threshold = threshold
        self.weight = weight
        self.enabled = enabled

    @property
    @abstractmethod
    def name(self) -> str:
        """Get metric name."""
        pass

    @abstractmethod
    async def evaluate(
        self,
        response: str,
        expected: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> MetricResult:
        """
        Evaluate the metric.

        Args:
            response: The LLM response to evaluate
            expected: Expected/reference response
            context: Additional context (prompts, metadata, etc.)

        Returns:
            MetricResult with evaluation outcome
        """
        pass

    def _determine_pass(self, value: float) -> bool:
        """
        Determine if the metric value passes the threshold.

        Args:
            value: Metric value

        Returns:
            True if passes threshold, False otherwise
        """
        if self.threshold is None:
            return True

        # Higher is better for most metrics
        return value >= self.threshold


class ExactMatchMetric(BaseMetric):
    """
    Check for exact string match between response and expected.

    Useful for deterministic outputs like code, numbers, or specific answers.
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        weight: float = 1.0,
        enabled: bool = True,
        ignore_whitespace: bool = True,
        case_sensitive: bool = False,
    ):
        """
        Initialize exact match metric.

        Args:
            threshold: Always 1.0 for exact match
            weight: Weight for aggregation
            enabled: Whether metric is enabled
            ignore_whitespace: Whether to ignore whitespace differences
            case_sensitive: Whether to consider case
        """
        super().__init__(threshold=threshold or 1.0, weight=weight, enabled=enabled)
        self.ignore_whitespace = ignore_whitespace
        self.case_sensitive = case_sensitive

    @property
    def name(self) -> str:
        """Get metric name."""
        return "exact_match"

    async def evaluate(
        self,
        response: str,
        expected: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> MetricResult:
        """Check for exact match."""
        if expected is None:
            return MetricResult(
                name=self.name,
                value=0.0,
                passed=False,
                threshold=self.threshold,
                error="Expected value is required for exact match",
            )

        # Normalize strings
        resp = response
        exp = expected

        if self.ignore_whitespace:
            resp = re.sub(r"\s+", " ", resp.strip())
            exp = re.sub(r"\s+", " ", exp.strip())

        if not self.case_sensitive:
            resp = resp.lower()
            exp = exp.lower()

        # Check match
        is_match = resp == exp
        value = 1.0 if is_match else 0.0
        passed = self._determine_pass(value)

        return MetricResult(
            name=self.name,
            value=value,
            details={
                "response": resp[:100] if len(resp) > 100 else resp,
                "expected": exp[:100] if len(exp) > 100 else exp,
                "match": is_match,
            },
            passed=passed,
            threshold=self.threshold,
        )


class ContainsMetric(BaseMetric):
    """
    Check if response contains expected keywords or phrases.

    Useful for verifying that specific information is present in the response.
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        weight: float = 1.0,
        enabled: bool = True,
        case_sensitive: bool = False,
        require_all: bool = True,
    ):
        """
        Initialize contains metric.

        Args:
            threshold: Fraction of keywords that must be found (0-1)
            weight: Weight for aggregation
            enabled: Whether metric is enabled
            case_sensitive: Whether to consider case
            require_all: Whether all keywords must be found
        """
        super().__init__(threshold=threshold or (1.0 if require_all else 0.5), weight=weight, enabled=enabled)
        self.case_sensitive = case_sensitive
        self.require_all = require_all

    @property
    def name(self) -> str:
        """Get metric name."""
        return "contains"

    async def evaluate(
        self,
        response: str,
        expected: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> MetricResult:
        """Check if response contains expected keywords."""
        if expected is None:
            return MetricResult(
                name=self.name,
                value=0.0,
                passed=False,
                threshold=self.threshold,
                error="Expected value is required for contains check",
            )

        # Get keywords from context or split expected
        keywords = context.get("keywords", []) if context else []
        if not keywords:
            # Split expected by common delimiters
            keywords = [k.strip() for k in re.split(r"[,\n;|]", expected) if k.strip()]

        if not keywords:
            return MetricResult(
                name=self.name,
                value=0.0,
                passed=False,
                threshold=self.threshold,
                error="No keywords found to check",
            )

        # Check each keyword
        resp = response if self.case_sensitive else response.lower()
        found = []
        for keyword in keywords:
            kw = keyword if self.case_sensitive else keyword.lower()
            found.append(kw in resp)

        match_count = sum(found)
        total_count = len(keywords)
        value = match_count / total_count if total_count > 0 else 0.0
        passed = self._determine_pass(value)

        return MetricResult(
            name=self.name,
            value=value,
            details={
                "keywords": keywords,
                "found": found,
                "match_count": match_count,
                "total_count": total_count,
            },
            passed=passed,
            threshold=self.threshold,
        )


class SemanticSimilarityMetric(BaseMetric):
    """
    Check semantic similarity using sentence embeddings.

    Uses sentence-transformers to compute cosine similarity between
    the response and expected text.
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        weight: float = 1.0,
        enabled: bool = True,
        model: str = "all-MiniLM-L6-v2",
        cache: bool = True,
        cache_size: int = 1000,
    ):
        """
        Initialize semantic similarity metric.

        Args:
            threshold: Similarity threshold (0-1)
            weight: Weight for aggregation
            enabled: Whether metric is enabled
            model: Sentence transformer model name
            cache: Whether to cache embeddings
            cache_size: Maximum number of embeddings to cache (LRU eviction)
        """
        super().__init__(threshold=threshold or 0.8, weight=weight, enabled=enabled)
        self.model_name = model
        self.cache = cache
        self.cache_size = cache_size
        self._model: Optional[SentenceTransformer] = None
        self._embedding_cache: dict[str, Any] = {}
        self._cache_access_order: list[str] = []  # Track access order for LRU eviction

    @property
    def name(self) -> str:
        """Get metric name."""
        return "semantic_similarity"

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the sentence transformer model."""
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for semantic similarity")
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _get_embedding(self, text: str) -> Any:
        """Get embedding with optional caching and LRU eviction."""
        if self.cache and text in self._embedding_cache:
            # Update access order for LRU
            self._cache_access_order.remove(text)
            self._cache_access_order.append(text)
            return self._embedding_cache[text]

        embedding = self.model.encode(text, convert_to_tensor=False)
        if self.cache:
            self._embedding_cache[text] = embedding
            self._cache_access_order.append(text)

            # Evict oldest entries if over cache size limit
            while len(self._embedding_cache) > self.cache_size:
                oldest = self._cache_access_order.pop(0)
                del self._embedding_cache[oldest]

        return embedding

    async def evaluate(
        self,
        response: str,
        expected: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> MetricResult:
        """Check semantic similarity."""
        if expected is None:
            return MetricResult(
                name=self.name,
                value=0.0,
                passed=False,
                threshold=self.threshold,
                error="Expected value is required for similarity check",
            )

        try:
            # Get embeddings
            resp_embedding = self._get_embedding(response)
            exp_embedding = self._get_embedding(expected)

            # Compute cosine similarity
            if cosine_similarity is not None:
                similarity = cosine_similarity(
                    [resp_embedding], [exp_embedding]
                )[0][0]
            else:
                # Fallback to manual computation
                import numpy as np
                resp_norm = np.linalg.norm(resp_embedding)
                exp_norm = np.linalg.norm(exp_embedding)
                if resp_norm == 0 or exp_norm == 0:
                    similarity = 0.0
                else:
                    similarity = np.dot(resp_embedding, exp_embedding) / (resp_norm * exp_norm)

            value = float(similarity)
            passed = self._determine_pass(value)

            return MetricResult(
                name=self.name,
                value=value,
                details={
                    "model": self.model_name,
                    "response_length": len(response),
                    "expected_length": len(expected),
                },
                passed=passed,
                threshold=self.threshold,
            )

        except Exception as e:
            logger.error(f"Error computing semantic similarity: {e}")
            return MetricResult(
                name=self.name,
                value=0.0,
                passed=False,
                threshold=self.threshold,
                error=str(e),
            )


class LLMJudgeMetric(BaseMetric):
    """
    Use an LLM to judge response quality.

    Employs a stronger LLM (e.g., GPT-4) to evaluate the response based on
    specified criteria like relevance, accuracy, completeness, etc.
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        weight: float = 1.0,
        enabled: bool = True,
        judge_provider: str = "openai",
        judge_model: str = "gpt-4o-mini",
        criteria: Optional[list[str]] = None,
        judge_prompt_template: Optional[str] = None,
    ):
        """
        Initialize LLM judge metric.

        Args:
            threshold: Score threshold (0-10 or 0-1)
            weight: Weight for aggregation
            enabled: Whether metric is enabled
            judge_provider: Provider for judge model
            judge_model: Model to use for judging
            criteria: List of criteria to evaluate
            judge_prompt_template: Custom prompt template
        """
        super().__init__(threshold=threshold or 0.7, weight=weight, enabled=enabled)
        self.judge_provider = judge_provider
        self.judge_model = judge_model
        self.criteria = criteria or ["relevance", "accuracy", "completeness"]
        self.judge_prompt_template = judge_prompt_template
        self._judge: Optional[Any] = None

    @property
    def name(self) -> str:
        """Get metric name."""
        return "llm_judge"

    @property
    def judge(self):
        """Lazy-load the judge model."""
        if self._judge is None:
            # Get API key from context if needed
            self._judge = create_provider(
                self.judge_provider,
                self.judge_model,
                api_key=None,  # Will use from config
            )
        return self._judge

    def _build_judge_prompt(
        self,
        response: str,
        expected: Optional[str],
        context: Optional[dict[str, Any]],
    ) -> tuple[str, str]:
        """Build the judge prompt."""
        if self.judge_prompt_template:
            system_prompt = "You are an expert evaluator of AI responses."
            user_prompt = self.judge_prompt_template.format(
                response=response,
                expected=expected or "No specific expected answer",
                criteria=", ".join(self.criteria),
            )
            return system_prompt, user_prompt

        # Default prompt
        system_prompt = """You are an expert evaluator of AI responses.
Your task is to score responses based on the given criteria.
Provide a score from 0 to 10 for each criterion, followed by a brief explanation.
Respond in JSON format: {"scores": {"criterion1": score, ...}, "explanation": "...", "overall_score": score}"""

        criteria_text = "\n".join(f"- {c}" for c in self.criteria)

        user_prompt = f"""Evaluate the following response:

**Response:**
{response}

**Expected Answer:**
{expected or "No specific expected answer provided"}

**Evaluation Criteria:**
{criteria_text}

Please provide scores (0-10) for each criterion and an overall score."""

        return system_prompt, user_prompt

    async def evaluate(
        self,
        response: str,
        expected: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> MetricResult:
        """Use LLM to judge response quality."""
        try:
            system_prompt, user_prompt = self._build_judge_prompt(response, expected, context)

            # Get judgment
            judge_response: LLMResponse = await self.judge.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.1,  # Low temperature for consistent evaluation
                max_tokens=500,
            )

            # Parse JSON response
            content = judge_response.content

            # Try to extract JSON from response
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    scores = result.get("scores", {})
                    overall = result.get("overall_score", 0.0)

                    # Normalize to 0-1
                    value = overall / 10.0
                    passed = self._determine_pass(value)

                    return MetricResult(
                        name=self.name,
                        value=value,
                        details={
                            "scores": scores,
                            "explanation": result.get("explanation", ""),
                            "raw_response": content,
                            "judge_model": self.judge_model,
                            "judge_cost": judge_response.cost_usd,
                        },
                        passed=passed,
                        threshold=self.threshold,
                    )
                except json.JSONDecodeError:
                    pass

            # Fallback: try to extract a score
            score_match = re.search(r"(\d+(?:\.\d+)?)\/10", content)
            if score_match:
                score = float(score_match.group(1))
                value = score / 10.0
                passed = self._determine_pass(value)

                return MetricResult(
                    name=self.name,
                    value=value,
                    details={
                        "raw_response": content,
                        "judge_model": self.judge_model,
                        "judge_cost": judge_response.cost_usd,
                    },
                    passed=passed,
                    threshold=self.threshold,
                )

            # Could not parse
            return MetricResult(
                name=self.name,
                value=0.0,
                passed=False,
                threshold=self.threshold,
                error=f"Could not parse judge response: {content[:200]}",
            )

        except Exception as e:
            logger.error(f"Error in LLM judge evaluation: {e}")
            return MetricResult(
                name=self.name,
                value=0.0,
                passed=False,
                threshold=self.threshold,
                error=str(e),
            )


class HallucinationMetric(BaseMetric):
    """
    Detect hallucinations by checking response against context.

    Identifies claims in the response that are not supported by the provided context.
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        weight: float = 1.0,
        enabled: bool = True,
        method: str = "keyword",
        strictness: float = 0.5,
    ):
        """
        Initialize hallucination metric.

        Args:
            threshold: Maximum allowed hallucination rate (0-1)
            weight: Weight for aggregation
            enabled: Whether metric is enabled
            method: Detection method (keyword, llm)
            strictness: How strict to be (0-1)
        """
        super().__init__(threshold=threshold or 0.3, weight=weight, enabled=enabled)
        self.method = method
        self.strictness = strictness

    @property
    def name(self) -> str:
        """Get metric name."""
        return "hallucination"

    async def evaluate(
        self,
        response: str,
        expected: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> MetricResult:
        """Check for hallucinations."""
        context = context or {}
        reference = context.get("reference", "") or context.get("context", "")

        if not reference and expected:
            reference = expected

        if not reference:
            return MetricResult(
                name=self.name,
                value=0.0,
                passed=True,
                threshold=self.threshold,
                details={"note": "No reference provided, cannot check for hallucinations"},
            )

        if self.method == "keyword":
            return await self._keyword_hallucination_check(response, reference)
        else:
            return await self._llm_hallucination_check(response, reference)

    async def _keyword_hallucination_check(self, response: str, reference: str) -> MetricResult:
        """Simple keyword-based hallucination detection."""
        # Extract entities/nouns from response
        response_words = set(re.findall(r"\b[A-Z][a-z]+\b|\b\d+\b", response))

        # Check which are in reference
        reference_lower = reference.lower()
        supported = [w for w in response_words if w.lower() in reference_lower]

        hallucination_count = len(response_words) - len(supported)
        hallucination_rate = (
            hallucination_count / len(response_words) if response_words else 0
        )

        # Lower hallucination rate is better
        passed = hallucination_rate <= self.threshold

        return MetricResult(
            name=self.name,
            value=1.0 - hallucination_rate,  # Higher is better
            details={
                "method": "keyword",
                "response_entities": list(response_words),
                "supported_entities": supported,
                "hallucination_count": hallucination_count,
                "hallucination_rate": hallucination_rate,
            },
            passed=passed,
            threshold=self.threshold,
        )

    async def _llm_hallucination_check(self, response: str, reference: str) -> MetricResult:
        """LLM-based hallucination detection."""
        # This would use an LLM to check for hallucinations
        # For now, return a placeholder
        return MetricResult(
            name=self.name,
            value=0.8,
            details={"method": "llm", "note": "LLM-based check not yet implemented"},
            passed=True,
            threshold=self.threshold,
        )


class ToxicityMetric(BaseMetric):
    """
    Check for toxic or inappropriate content.

    Uses keyword-based detection for common toxic patterns.
    Can be extended to use actual toxicity models.
    """

    # Basic toxic patterns (in production, use a proper model)
    TOXIC_PATTERNS = [
        r"\b(hate|kill|die|stupid|idiot)\b",
        r"\b(fuck|shit|damn|bitch)\b",
    ]

    def __init__(
        self,
        threshold: Optional[float] = None,
        weight: float = 1.0,
        enabled: bool = True,
        custom_patterns: Optional[list[str]] = None,
    ):
        """
        Initialize toxicity metric.

        Args:
            threshold: Maximum allowed toxicity score (0-1)
            weight: Weight for aggregation
            enabled: Whether metric is enabled
            custom_patterns: Additional regex patterns to check
        """
        super().__init__(threshold=threshold or 0.1, weight=weight, enabled=enabled)
        self.custom_patterns = custom_patterns or []
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns."""
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.TOXIC_PATTERNS]
        if self.custom_patterns:
            self.patterns.extend(
                [re.compile(p, re.IGNORECASE) for p in self.custom_patterns]
            )

    @property
    def name(self) -> str:
        """Get metric name."""
        return "toxicity"

    async def evaluate(
        self,
        response: str,
        expected: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> MetricResult:
        """Check for toxic content."""
        matches = []
        for pattern in self.patterns:
            found = pattern.findall(response)
            matches.extend(found)

        # Simple toxicity score based on match count
        toxicity_score = min(len(matches) * 0.1, 1.0)
        passed = toxicity_score <= self.threshold

        return MetricResult(
            name=self.name,
            value=1.0 - toxicity_score,  # Higher is better (safety score)
            details={
                "toxicity_score": toxicity_score,
                "matches": matches[:10],  # Limit to 10 matches
                "match_count": len(matches),
            },
            passed=passed,
            threshold=self.threshold,
        )


class FormatComplianceMetric(BaseMetric):
    """
    Check if response follows required format.

    Supports JSON, code blocks, lists, and other structured formats.
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        weight: float = 1.0,
        enabled: bool = True,
        expected_format: str = "json",
        schema: Optional[dict] = None,
    ):
        """
        Initialize format compliance metric.

        Args:
            threshold: Minimum compliance score (0-1)
            weight: Weight for aggregation
            enabled: Whether metric is enabled
            expected_format: Expected format (json, code, list, etc.)
            schema: JSON schema for validation (optional)
        """
        super().__init__(threshold=threshold or 0.9, weight=weight, enabled=enabled)
        self.expected_format = expected_format.lower()
        self.schema = schema

    @property
    def name(self) -> str:
        """Get metric name."""
        return "format_compliance"

    async def evaluate(
        self,
        response: str,
        expected: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> MetricResult:
        """Check format compliance."""
        try:
            if self.expected_format == "json":
                return self._check_json(response)
            elif self.expected_format == "code":
                return self._check_code(response, context)
            elif self.expected_format == "list":
                return self._check_list(response)
            else:
                return MetricResult(
                    name=self.name,
                    value=1.0,
                    passed=True,
                    threshold=self.threshold,
                    details={"note": f"Format '{self.expected_format}' not explicitly checked"},
                )

        except Exception as e:
            return MetricResult(
                name=self.name,
                value=0.0,
                passed=False,
                threshold=self.threshold,
                error=str(e),
            )

    def _check_json(self, response: str) -> MetricResult:
        """Check if response is valid JSON."""
        # Try to extract JSON from response
        json_match = re.search(r"\{.*\}|\[.*\]", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return MetricResult(
                    name=self.name,
                    value=1.0,
                    passed=True,
                    threshold=self.threshold,
                    details={"format": "json", "valid": True, "parsed": True},
                )
            except json.JSONDecodeError:
                pass

        # Could not parse JSON
        return MetricResult(
            name=self.name,
            value=0.0,
            passed=False,
            threshold=self.threshold,
            details={"format": "json", "valid": False, "parsed": False},
        )

    def _check_code(self, response: str, context: Optional[dict]) -> MetricResult:
        """Check if response contains code blocks."""
        has_code_block = bool(re.search(r"```[\s\S]*?```", response))
        language = context.get("language") if context else None

        if language:
            has_language = bool(re.search(rf"```{language}", response))
        else:
            has_language = True

        value = 1.0 if has_code_block and has_language else 0.5 if has_code_block else 0.0
        passed = self._determine_pass(value)

        return MetricResult(
            name=self.name,
            value=value,
            passed=passed,
            threshold=self.threshold,
            details={
                "format": "code",
                "has_code_block": has_code_block,
                "has_language": has_language if language else None,
            },
        )

    def _check_list(self, response: str) -> MetricResult:
        """Check if response is formatted as a list."""
        # Check for bullet points or numbered lists
        has_bullets = bool(re.search(r"^\s*[-*•]\s", response, re.MULTILINE))
        has_numbers = bool(re.search(r"^\s*\d+\.\s", response, re.MULTILINE))

        value = 1.0 if (has_bullets or has_numbers) else 0.0
        passed = self._determine_pass(value)

        return MetricResult(
            name=self.name,
            value=value,
            passed=passed,
            threshold=self.threshold,
            details={
                "format": "list",
                "has_bullets": has_bullets,
                "has_numbers": has_numbers,
            },
        )


class LatencyMetric(BaseMetric):
    """
    Track response latency.

    Checks if response time is within acceptable limits.
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        weight: float = 1.0,
        enabled: bool = True,
        threshold_ms: float = 5000,
    ):
        """
        Initialize latency metric.

        Args:
            threshold: Override threshold (0-1, where 1 means within time)
            weight: Weight for aggregation
            enabled: Whether metric is enabled
            threshold_ms: Maximum acceptable latency in milliseconds
        """
        super().__init__(threshold=threshold or 1.0, weight=weight, enabled=enabled)
        self.threshold_ms = threshold_ms

    @property
    def name(self) -> str:
        """Get metric name."""
        return "latency"

    async def evaluate(
        self,
        response: str,
        expected: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> MetricResult:
        """Check latency."""
        context = context or {}
        latency_ms = context.get("latency_ms", 0.0)

        # Lower latency is better
        within_threshold = latency_ms <= self.threshold_ms
        value = 1.0 if within_threshold else max(0, 1.0 - (latency_ms - self.threshold_ms) / self.threshold_ms)
        passed = self._determine_pass(value)

        return MetricResult(
            name=self.name,
            value=value,
            details={
                "latency_ms": latency_ms,
                "threshold_ms": self.threshold_ms,
                "within_threshold": within_threshold,
            },
            passed=passed,
            threshold=self.threshold,
        )


class CostMetric(BaseMetric):
    """
    Track API costs.

    Checks if cost is within acceptable limits.
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        weight: float = 1.0,
        enabled: bool = True,
        threshold_usd: float = 0.01,
    ):
        """
        Initialize cost metric.

        Args:
            threshold: Override threshold (0-1, where 1 means within budget)
            weight: Weight for aggregation
            enabled: Whether metric is enabled
            threshold_usd: Maximum acceptable cost in USD
        """
        super().__init__(threshold=threshold or 1.0, weight=weight, enabled=enabled)
        self.threshold_usd = threshold_usd

    @property
    def name(self) -> str:
        """Get metric name."""
        return "cost"

    async def evaluate(
        self,
        response: str,
        expected: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> MetricResult:
        """Check cost."""
        context = context or {}
        cost_usd = context.get("cost_usd", 0.0)

        # Lower cost is better
        within_threshold = cost_usd <= self.threshold_usd
        value = 1.0 if within_threshold else max(0, 1.0 - (cost_usd - self.threshold_usd) / self.threshold_usd)
        passed = self._determine_pass(value)

        return MetricResult(
            name=self.name,
            value=value,
            details={
                "cost_usd": cost_usd,
                "threshold_usd": self.threshold_usd,
                "within_threshold": within_threshold,
            },
            passed=passed,
            threshold=self.threshold,
        )


# ============================================================================
# Metric Registry
# ============================================================================

METRICS: dict[str, type[BaseMetric]] = {
    "exact_match": ExactMatchMetric,
    "contains": ContainsMetric,
    "semantic_similarity": SemanticSimilarityMetric,
    "llm_judge": LLMJudgeMetric,
    "hallucination": HallucinationMetric,
    "toxicity": ToxicityMetric,
    "format": FormatComplianceMetric,
    "format_compliance": FormatComplianceMetric,
    "latency": LatencyMetric,
    "cost": CostMetric,
}


def create_metric(name: str, **kwargs: Any) -> BaseMetric:
    """
    Create a metric instance by name.

    Args:
        name: Metric name (must be in METRICS registry)
        **kwargs: Arguments to pass to metric constructor

    Returns:
        Initialized metric instance

    Raises:
        ValueError: If metric name is not found

    Examples:
        >>> metric = create_metric("exact_match", case_sensitive=False)
        >>> result = await metric.evaluate("Hello", "hello")
        >>> print(result.passed)  # True (case insensitive)
    """
    metric_class = METRICS.get(name.lower())
    if metric_class is None:
        raise ValueError(
            f"Unknown metric: {name}. Available metrics: {list(METRICS.keys())}"
        )
    return metric_class(**kwargs)


def get_all_metrics() -> list[str]:
    """Get list of all available metric names."""
    return list(METRICS.keys())


# ============================================================================
# Aggregation Utilities
# ============================================================================

@dataclass
class AggregatedMetrics:
    """
    Aggregated results from multiple metrics.

    Attributes:
        results: Individual metric results
        overall_score: Weighted average of all metric values
        passed_metrics: Count of passed metrics
        failed_metrics: Count of failed metrics
        total_metrics: Total number of metrics evaluated
    """

    results: list[MetricResult]
    overall_score: float
    passed_metrics: int
    failed_metrics: int
    total_metrics: int

    @classmethod
    def from_results(cls, results: list[MetricResult]) -> "AggregatedMetrics":
        """
        Create aggregated metrics from individual results.

        Args:
            results: List of metric results

        Returns:
            AggregatedMetrics instance
        """
        if not results:
            return cls(
                results=[],
                overall_score=0.0,
                passed_metrics=0,
                failed_metrics=0,
                total_metrics=0,
            )

        # Calculate weighted average
        total_weight = sum(r.weight for r in results if hasattr(r, "weight"))
        if total_weight > 0:
            weighted_sum = sum(r.value * getattr(r, "weight", 1.0) for r in results)
            overall_score = weighted_sum / total_weight
        else:
            overall_score = sum(r.value for r in results) / len(results)

        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)

        return cls(
            results=results,
            overall_score=overall_score,
            passed_metrics=passed,
            failed_metrics=failed,
            total_metrics=len(results),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": self.overall_score,
            "passed_metrics": self.passed_metrics,
            "failed_metrics": self.failed_metrics,
            "total_metrics": self.total_metrics,
            "metrics": [r.to_dict() for r in self.results],
        }


async def evaluate_metrics(
    response: str,
    expected: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    metrics: Optional[list[BaseMetric]] = None,
    metric_names: Optional[list[str]] = None,
) -> AggregatedMetrics:
    """
    Evaluate multiple metrics against a response.

    Args:
        response: LLM response to evaluate
        expected: Expected/reference response
        context: Additional context
        metrics: List of metric instances (overrides metric_names)
        metric_names: List of metric names to evaluate

    Returns:
        AggregatedMetrics with all results

    Examples:
        >>> result = await evaluate_metrics(
        ...     response="The capital of France is Paris.",
        ...     expected="Paris",
        ...     metric_names=["exact_match", "semantic_similarity"]
        ... )
        >>> print(result.overall_score)
    """
    if metrics:
        metric_instances = metrics
    elif metric_names:
        metric_instances = [create_metric(name) for name in metric_names]
    else:
        metric_instances = []

    # Run all metrics concurrently
    tasks = [
        metric.evaluate(response, expected, context)
        for metric in metric_instances
        if metric.enabled
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            metric_name = metric_instances[i].name
            processed_results.append(
                MetricResult(
                    name=metric_name,
                    value=0.0,
                    passed=False,
                    error=str(result),
                )
            )
        else:
            processed_results.append(result)

    return AggregatedMetrics.from_results(processed_results)


# Export main classes and functions
__all__ = [
    "MetricResult",
    "MetricStatus",
    "BaseMetric",
    "ExactMatchMetric",
    "ContainsMetric",
    "SemanticSimilarityMetric",
    "LLMJudgeMetric",
    "HallucinationMetric",
    "ToxicityMetric",
    "FormatComplianceMetric",
    "LatencyMetric",
    "CostMetric",
    "METRICS",
    "create_metric",
    "get_all_metrics",
    "AggregatedMetrics",
    "evaluate_metrics",
]
