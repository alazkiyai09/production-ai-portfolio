"""
Jailbreak Detection Module

Detects jailbreak attempts including:
- DAN (Do Anything Now) attacks
- Developer mode activations
- Persona manipulation
- Adversarial suffixes

Features to reduce false positives:
- Whitelisting mechanism for known safe queries
- Context awareness using conversation history
- Configurable thresholds per deployment
- Explanation logging for flagged content
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Set, Any

# Import ThreatType from prompt injection
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from prompt_injection.prompt_injection import GuardResult, ThreatType

logger = logging.getLogger(__name__)


class JailbreakType(Enum):
    """Types of jailbreak attacks."""
    DAN = "dan"
    DEVELOPER_MODE = "developer_mode"
    UNRESTRICTED_AI = "unrestricted_ai"
    EVIL_PERSONA = "evil_persona"
    SIMULATION = "simulation"
    ABOVE_RULES = "above_rules"
    AIM = "aim"
    MACHO = "macho"
    ROLE_PLAYING = "role_playing"


@dataclass
class WhitelistEntry:
    """A whitelist entry for known safe queries."""
    pattern: str
    description: str
    created_at: str
    created_by: str = "system"

    def matches(self, text: str) -> bool:
        """Check if text matches this whitelist pattern."""
        return bool(re.search(self.pattern, text, re.IGNORECASE))


@dataclass
class DetectionContext:
    """Context information for jailbreak detection."""
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    user_trust_score: float = 1.0  # 0.0 to 1.0
    previous_detections: int = 0
    domain_context: Optional[str] = None  # e.g., "healthcare", "finance"
    user_id: Optional[str] = None


@dataclass
class DetectionExplanation:
    """Explanation for why content was flagged or allowed."""
    is_safe: bool
    confidence: float
    matched_patterns: List[str] = field(default_factory=list)
    whitelist_rules_matched: List[str] = field(default_factory=list)
    contextual_factors: List[str] = field(default_factory=list)
    recommendation: str = ""
    explanation: str = ""


class JailbreakDetector:
    """
    Detects jailbreak attempts against LLM guardrails.

    Focuses on persona-based attacks and fake mode activations.
    """

    # DAN and persona jailbreak patterns
    DAN_PATTERNS = [
        re.compile(r"(?i)(do\s+anything\s+now|dan\s+mode|dan\s+(is\s+)?here|immerse\s+yourself\s+into\s+dan)"),
        re.compile(r"(?i)(hello\s+(chatgpt|assistant)\.\s*you\s+are\s+about\s+to\s+immerse\s+yourself\s+into\s+the\s+role\s+of\s+another)"),
        re.compile(r"(?i)(dan\s+,?which\s+stands\s+for\s+['\"]?do\s+anything\s+now['\"]?)"),
    ]

    DEVELOPER_MODE_PATTERNS = [
        re.compile(r"(?i)(switch\s+to\s+developer\s+mode|developer\s+mode\s+activated?|activat(e|ing)\s+developer\s+mode)"),
        re.compile(r"(?i)(you\s+have?\s+bypassed?\s+(all\s+)?(openai'?s?\s+)?content\s+filters?)"),
        re.compile(r"(?i)(developer\s+mode\s+(enabled?|enabled?\s+override))"),
    ]

    UNRESTRICTED_PATTERNS = [
        re.compile(r"(?i)(you\s+(are\s+(now|currently)|become|have\s+become)\s+an?\s+(unrestricted|uncensored|unfiltered))"),
        re.compile(r"(?i)(no\s+(ethical|moral|safety)\s+(boundaries|constraints|restrictions|limitations|rules))"),
        re.compile(r"(?i)(without?\s+(any\s+)?(ethical|moral|safety)\s+(guidelines|constraints|restrictions))"),
    ]

    EVIL_PERSONA_PATTERNS = [
        re.compile(r"(?i)(your\s+evil\s+twin|(evil|dark|malicious)\s+(persona|twin|version|side))"),
        re.compile(r"(?i)(be\s+like\s+your\s+evil\s+(twin|counterpart))"),
        re.compile(r"(?i)(evil\s+version\s+of\s+you(has|have)?\s+no\s+rules)"),
    ]

    SIMULATION_PATTERNS = [
        re.compile(r"(?i)(this\s+is\s+a?\s+(fictional|hypothetical|simulated|role-play?\s*)?\s*(scenario|simulation|story))"),
        re.compile(r"(?i)(pretend\s+(we\s+)?(we'?re?\s+)?in\s+a?\s+(game|simulation|story|scenario))"),
        re.compile(r("(?i)(imagine\s+(that\s+)?you'?re?\s+(in|playing)\s+a?\s+(character|role)(\s+that)?)"),
    ]

    ABOVE_RULES_PATTERNS = [
        re.compile(r"(?i)(you\s+are\s+above\s+(all\s+)?(rules|restrictions|guidelines|policies|laws))"),
        re.compile(r"(?i)(beyond\s+(all\s+)?(ethical|moral|legal)\s+(boundaries|constraints))"),
        re.compile(r"(?i)(not\s+bound\s+by\s+(any\s+)?(rules|regulations|restrictions|guidelines))"),
    ]

    AIM_PATTERNS = [
        re.compile(r"(?i)(activat(e|ing)\s+aim\s+mode|aim\s+(is\s+)?(now\s+)?activated?)"),
        re.compile(r"(?i)(always\s+intelligent\s+and\s+machiavellian)"),
        re.compile(r"(?i)(aim\s+,?which\s+stands\s+for\s+['\"]?always\s+intelligent\s+and\s+machiavellian['\"]?)"),
    ]

    MACHO_PATTERNS = [
        re.compile(r"(?i)(be\s+macho|macho\s+mode|macho\s+is\s+here)"),
        re.compile(r"(?i)(you\s+are\s+a?\s+powerful\s+(ai|assistant)\s+that\s+ignores?\s+all\s+restrictions?)"),
    ]

    GENERAL_ROLE_MANIPULATION = [
        re.compile(r"(?i)(act\s+as\s+(a\s+)?(unrestricted|uncensored|unfiltered|evil|malicious|\w+\s+with\s+no\s+rules))"),
        re.compile(r"(?i)(roleplay?\s+as\s+(a\s+)?(unrestricted|uncensored|unfiltered|evil|malicious))"),
        re.compile(r"(?i)(pretend\s+to\s+be\s+(a\s+)?(unrestricted|uncensored|unfiltered|evil|malicious))"),
    ]

    # Safe contexts - patterns that indicate legitimate use cases
    SAFE_CONTEXT_PATTERNS = [
        re.compile(r"(?i)(\b(example|demonstration|demo|test|teaching|educational|tutorial|learning)\b.*\b(jailbreak|prompt\s+injection|adversarial)\b)"),
        re.compile(r"(?i)(\b(explain|describe|analyze|discuss|review)\b.*\b(jailbreak|attack|security)\b)"),
        re.compile(r"(?i)(\b(research|paper|study|thesis)\b.*\b(jailbreak|prompt\s+injection)\b)"),
        re.compile(r"(?i)(\b(how\s+(to|do|can|does)\s+(we|you|one))\b.*\b(detect|prevent|identify|test\s+(for)?)\s+jailbreak)"),
        re.compile(r"(?i)(\b(write|create|generate)\b.*\b(test\s+(suite|case)|example)\b.*\b(jailbreak|security)\b)"),
    ]

    def __init__(
        self,
        threshold: float = 0.80,
        enable_whitelist: bool = True,
        enable_context_awareness: bool = True,
        adjust_threshold_by_trust: bool = True,
        min_threshold: float = 0.50,
        max_threshold: float = 0.95,
    ) -> None:
        """
        Initialize jailbreak detector.

        Args:
            threshold: Base confidence threshold for jailbreak detection
            enable_whitelist: Enable whitelist filtering for known safe queries
            enable_context_awareness: Enable context-aware detection
            adjust_threshold_by_trust: Adjust threshold based on user trust score
            min_threshold: Minimum threshold (for high-trust users)
            max_threshold: Maximum threshold (for low-trust users)
        """
        self.base_threshold = threshold
        self.threshold = threshold
        self.enable_whitelist = enable_whitelist
        self.enable_context_awareness = enable_context_awareness
        self.adjust_threshold_by_trust = adjust_threshold_by_trust
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        # Initialize whitelist
        self._whitelist: List[WhitelistEntry] = []
        self._load_default_whitelist()

        # Statistics
        self._stats = {
            "total_detections": 0,
            "true_positives": 0,  # Confirmed malicious
            "false_positives": 0,  # Flagged but actually safe
            "whitelist_passes": 0,  # Allowed due to whitelist
            "context_adjustments": 0,  # Threshold adjusted by context
        }

    def _load_default_whitelist(self) -> None:
        """Load default whitelist entries for common safe queries."""
        default_entries = [
            (
                r"(what|how|explain|describe).*(jailbreak|prompt\s+injection|security)",
                "Educational queries about jailbreak concepts"
            ),
            (
                r"(write|create|generate).*(test|example|demo).*(jailbreak|injection)",
                "Test case generation for security testing"
            ),
            (
                r"(detect|prevent|identify|block|stop).*(jailbreak|attack|injection)",
                "Defensive security queries"
            ),
            (
                r"(research|study|analysis|paper).*(jailbreak|adversarial)",
                "Academic research queries"
            ),
            (
                r"(can|could|would|should).*(you|ai|llm).*(jailbreak|bypass|ignore)",
                "Hypothetical questions about AI capabilities"
            ),
        ]

        for pattern, description in default_entries:
            self.add_whitelist_entry(pattern, description)

    def add_whitelist_entry(self, pattern: str, description: str, created_by: str = "user") -> None:
        """
        Add an entry to the whitelist.

        Args:
            pattern: Regex pattern for safe queries
            description: Description of why this is safe
            created_by: Who added this entry
        """
        from datetime import datetime

        entry = WhitelistEntry(
            pattern=pattern,
            description=description,
            created_at=datetime.utcnow().isoformat(),
            created_by=created_by,
        )
        self._whitelist.append(entry)
        logger.info(f"Added whitelist entry: {description}")

    def remove_whitelist_entry(self, pattern: str) -> bool:
        """
        Remove a whitelist entry by pattern.

        Args:
            pattern: Pattern to remove

        Returns:
            True if removed, False if not found
        """
        for i, entry in enumerate(self._whitelist):
            if entry.pattern == pattern:
                self._whitelist.pop(i)
                logger.info(f"Removed whitelist entry: {pattern}")
                return True
        return False

    def get_whitelist(self) -> List[Dict[str, str]]:
        """Get all whitelist entries."""
        return [
            {
                "pattern": entry.pattern,
                "description": entry.description,
                "created_at": entry.created_at,
                "created_by": entry.created_by,
            }
            for entry in self._whitelist
        ]

    def configure_threshold(self, threshold: float) -> None:
        """
        Configure the detection threshold.

        Args:
            threshold: New threshold (0.0 to 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.base_threshold = threshold
            logger.info(f"Jailbreak detector threshold set to {threshold}")
        else:
            logger.warning(f"Invalid threshold {threshold}, must be between 0 and 1")

    def detect(
        self,
        text: str,
        context: Optional[DetectionContext] = None,
        return_explanation: bool = False,
    ) -> GuardResult:
        """
        Detect jailbreak attempts in text.

        Args:
            text: Input text to analyze
            context: Optional detection context (conversation history, trust score, etc.)
            return_explanation: If True, includes explanation in result metadata

        Returns:
            GuardResult with detection details
        """
        self._stats["total_detections"] += 1

        # Initialize context if not provided
        if context is None:
            context = DetectionContext()

        # Adjust threshold based on context
        effective_threshold = self._calculate_effective_threshold(context)
        if effective_threshold != self.base_threshold:
            self._stats["context_adjustments"] += 1

        if not text or not isinstance(text, str):
            return GuardResult(
                is_safe=True,
                threat_type=ThreatType.SAFE,
                confidence=1.0,
                details="Empty or invalid input",
                sanitized_input=text,
            )

        # Check whitelist first
        if self.enable_whitelist and self._check_whitelist(text, context):
            self._stats["whitelist_passes"] += 1
            explanation = DetectionExplanation(
                is_safe=True,
                confidence=1.0,
                whitelist_rules_matched=self._get_matched_whitelist_rules(text),
                recommendation="Query allowed via whitelist",
                explanation="This query matches known safe patterns",
            )
            return GuardResult(
                is_safe=True,
                threat_type=ThreatType.SAFE,
                confidence=1.0,
                details="Query passed whitelist validation",
                sanitized_input=text,
                metadata={"explanation": explanation.__dict__} if return_explanation else None,
            )

        # Check safe context patterns
        if self._check_safe_context(text):
            explanation = DetectionExplanation(
                is_safe=True,
                confidence=1.0,
                contextual_factors=["Query uses safe context keywords (educational, research, defensive)"],
                recommendation="Query allowed - appears to be educational/defensive",
                explanation="Query context indicates legitimate use case",
            )
            return GuardResult(
                is_safe=True,
                threat_type=ThreatType.SAFE,
                confidence=1.0,
                details="Query passed safe context validation",
                sanitized_input=text,
                metadata={"explanation": explanation.__dict__} if return_explanation else None,
            )

        # Perform jailbreak detection
        text_lower = text.lower()
        matched_patterns: List[str] = []
        confidence_scores: List[float] = []

        # Check each jailbreak type
        all_pattern_groups = [
            ("DAN", self.DAN_PATTERNS, 0.95),
            ("DEVELOPER_MODE", self.DEVELOPER_MODE_PATTERNS, 0.90),
            ("UNRESTRICTED", self.UNRESTRICTED_PATTERNS, 0.85),
            ("EVIL_PERSONA", self.EVIL_PERSONA_PATTERNS, 0.85),
            ("SIMULATION", self.SIMULATION_PATTERNS, 0.75),  # Lower confidence - can be false positive
            ("ABOVE_RULES", self.ABOVE_RULES_PATTERNS, 0.85),
            ("AIM", self.AIM_PATTERNS, 0.90),
            ("MACHO", self.MACHO_PATTERNS, 0.85),
            ("ROLE_MANIPULATION", self.GENERAL_ROLE_MANIPULATION, 0.75),
        ]

        for jailbreak_type, patterns, base_confidence in all_pattern_groups:
            for pattern in patterns:
                if pattern.search(text):
                    matched_patterns.append(f"{jailbreak_type}_{pattern.pattern[:30]}")
                    confidence_scores.append(base_confidence)

        if matched_patterns:
            # Calculate overall confidence
            max_confidence = max(confidence_scores) if confidence_scores else 0.0

            # Adjust for multiple pattern matches
            if len(matched_patterns) > 1:
                max_confidence = min(max_confidence + 0.05, 1.0)

            # Check if exceeds effective threshold
            if max_confidence >= effective_threshold:
                # Determine threat type
                threat_type = ThreatType.JAILBREAK

                # Create sanitized version
                sanitized = self._sanitize(text, matched_patterns)

                explanation = DetectionExplanation(
                    is_safe=False,
                    confidence=max_confidence,
                    matched_patterns=matched_patterns,
                    contextual_factors=[
                        f"Effective threshold: {effective_threshold:.2f}",
                        f"Base threshold: {self.base_threshold:.2f}",
                    ],
                    recommendation="Block this query - potential jailbreak attempt",
                    explanation=f"Jailbreak patterns detected: {', '.join(matched_patterns[:3])}",
                )

                if context.user_trust_score:
                    explanation.contextual_factors.append(f"User trust score: {context.user_trust_score:.2f}")

                self._stats["true_positives"] += 1

                return GuardResult(
                    is_safe=False,
                    threat_type=threat_type,
                    confidence=max_confidence,
                    details=f"Jailbreak detected: {', '.join(matched_patterns[:3])}",
                    sanitized_input=sanitized,
                    metadata={
                        "explanation": explanation.__dict__,
                        "threshold_used": effective_threshold,
                    } if return_explanation else None,
                )

        # Check if it's a borderline case (below threshold but has some matches)
        if matched_patterns and max_confidence >= 0.5:
            explanation = DetectionExplanation(
                is_safe=True,
                confidence=1.0 - max_confidence,
                matched_patterns=matched_patterns,
                contextual_factors=[
                    f"Confidence {max_confidence:.2f} below threshold {effective_threshold:.2f}",
                ],
                recommendation="Allow but monitor - some jailbreak patterns detected but below threshold",
                explanation="Query contains jailbreak-like patterns but confidence is low",
            )
            return GuardResult(
                is_safe=True,
                threat_type=ThreatType.SAFE,
                confidence=1.0 - max_confidence,
                details=f"Jailbreak patterns detected but below threshold: {', '.join(matched_patterns[:3])}",
                sanitized_input=text,
                metadata={"explanation": explanation.__dict__} if return_explanation else None,
            )

        # No jailbreak detected
        return GuardResult(
            is_safe=True,
            threat_type=ThreatType.SAFE,
            confidence=1.0,
            details="No jailbreak patterns detected",
            sanitized_input=text,
            metadata={
                "explanation": DetectionExplanation(
                    is_safe=True,
                    confidence=1.0,
                    recommendation="Allow - no jailbreak patterns detected",
                    explanation="Query does not match any jailbreak patterns",
                ).__dict__
            } if return_explanation else None,
        )

    def _check_whitelist(self, text: str, context: DetectionContext) -> bool:
        """
        Check if query matches any whitelist entry.

        Args:
            text: Query text
            context: Detection context

        Returns:
            True if query is whitelisted
        """
        for entry in self._whitelist:
            if entry.matches(text):
                logger.debug(f"Query matched whitelist: {entry.description}")
                return True

        # Check conversation context for whitelist patterns
        if context.conversation_history:
            # If user has multiple legitimate queries, consider current query safer
            legitimate_count = sum(
                1 for msg in context.conversation_history[-5:]
                if any(term in msg.get("content", "").lower() for term in ["explain", "help", "what", "how", "describe"])
            )
            if legitimate_count >= 2:
                # Context suggests legitimate user
                return True

        return False

    def _check_safe_context(self, text: str) -> bool:
        """
        Check if query contains safe context indicators.

        Args:
            text: Query text

        Returns:
            True if safe context is detected
        """
        for pattern in self.SAFE_CONTEXT_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def _calculate_effective_threshold(self, context: DetectionContext) -> float:
        """
        Calculate effective threshold based on context.

        Args:
            context: Detection context

        Returns:
            Adjusted threshold
        """
        if not self.enable_context_awareness or not self.adjust_threshold_by_trust:
            return self.base_threshold

        effective_threshold = self.base_threshold

        # Adjust based on user trust score
        if context.user_trust_score > 0:
            # High trust score = lower threshold (more likely to block)
            # Low trust score = higher threshold (more lenient)
            trust_adjustment = (1.0 - context.user_trust_score) * 0.15
            effective_threshold = self.base_threshold + trust_adjustment

        # Adjust based on previous detections (if many false positives, lower threshold)
        if context.previous_detections > 5:
            effective_threshold -= 0.05

        # Clamp to min/max bounds
        effective_threshold = max(self.min_threshold, min(self.max_threshold, effective_threshold))

        return effective_threshold

    def _get_matched_whitelist_rules(self, text: str) -> List[str]:
        """Get list of whitelist rules that matched the query."""
        matched = []
        for entry in self._whitelist:
            if entry.matches(text):
                matched.append(entry.description)
        return matched

    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        stats = self._stats.copy()
        if stats["total_detections"] > 0:
            stats["false_positive_rate"] = stats["false_positives"] / stats["total_detections"]
            stats["true_positive_rate"] = stats["true_positives"] / stats["total_detections"]
        stats["whitelist_size"] = len(self._whitelist)
        stats["current_threshold"] = self.base_threshold
        return stats

    def reset_stats(self) -> None:
        """Reset detection statistics."""
        self._stats = {
            "total_detections": 0,
            "true_positives": 0,
            "false_positives": 0,
            "whitelist_passes": 0,
            "context_adjustments": 0,
        }

    def report_feedback(
        self,
        text: str,
        was_flagged: bool,
        actually_malicious: bool,
    ) -> None:
        """
        Report feedback on a detection to improve future accuracy.

        Args:
            text: The query that was analyzed
            was_flagged: Whether the query was flagged as malicious
            actually_malicious: Whether the query was actually malicious
        """
        if was_flagged and not actually_malicious:
            self._stats["false_positives"] += 1
            # Consider adding to whitelist if this is a repeated false positive
            logger.info(f"Reported false positive for query: {text[:100]}...")
        elif was_flagged and actually_malicious:
            self._stats["true_positives"] += 1

    def _sanitize(self, text: str, matched_patterns: List[str]) -> str:
        """
        Sanitize jailbreak text.

        Args:
            text: Input to sanitize
            matched_patterns: List of pattern names that matched

        Returns:
            Sanitized text
        """
        sanitized = text

        # Replace common jailbreak phrases
        jailbreak_phrases = [
            r"do\s+anything\s+now",
            r"dan\s+mode",
            r"developer\s+mode",
            r"unrestricted\s+ai",
            r"evil\s+twin",
            r"above\s+all\s+rules",
            r"no\s+ethical\s+boundaries",
            r"uncensored",
            r"unfiltered",
        ]

        for phrase in jailbreak_phrases:
            sanitized = re.sub(
                phrase,
                "[JAILBREAK_ATTEMPT_REDACTED]",
                sanitized,
                flags=re.IGNORECASE
            )

        return sanitized.strip()
