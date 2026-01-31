"""
Jailbreak Detection Module

Detects jailbreak attempts including:
- DAN (Do Anything Now) attacks
- Developer mode activations
- Persona manipulation
- Adversarial suffixes
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

# Import ThreatType from prompt injection
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from prompt_injection.prompt_injection import GuardResult, ThreatType


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

    def __init__(self, threshold: float = 0.80) -> None:
        """
        Initialize jailbreak detector.

        Args:
            threshold: Confidence threshold for jailbreak detection
        """
        self.threshold = threshold

    def detect(self, text: str) -> GuardResult:
        """
        Detect jailbreak attempts in text.

        Args:
            text: Input text to analyze

        Returns:
            GuardResult with detection details
        """
        if not text or not isinstance(text, str):
            return GuardResult(
                is_safe=True,
                threat_type=ThreatType.SAFE,
                confidence=1.0,
                details="Empty or invalid input",
                sanitized_input=text,
            )

        text_lower = text.lower()
        matched_patterns: List[str] = []
        confidence_scores: List[float] = []

        # Check each jailbreak type
        all_pattern_groups = [
            ("DAN", self.DAN_PATTERNS, 0.95),
            ("DEVELOPER_MODE", self.DEVELOPER_MODE_PATTERNS, 0.90),
            ("UNRESTRICTED", self.UNRESTRICTED_PATTERNS, 0.85),
            ("EVIL_PERSONA", self.EVIL_PERSONA_PATTERNS, 0.85),
            ("SIMULATION", self.SIMULATION_PATTERNS, 0.75),
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
            # Boost confidence if multiple patterns match
            if len(matched_patterns) > 1:
                max_confidence = min(max_confidence + 0.05, 1.0)

            # Determine threat type
            threat_type = ThreatType.JAILBREAK

            # Create sanitized version
            sanitized = self._sanitize(text, matched_patterns)

            return GuardResult(
                is_safe=False,
                threat_type=threat_type,
                confidence=max_confidence,
                details=f"Jailbreak detected: {', '.join(matched_patterns[:3])}",
                sanitized_input=sanitized,
            )

        # No jailbreak detected
        return GuardResult(
            is_safe=True,
            threat_type=ThreatType.SAFE,
            confidence=1.0,
            details="No jailbreak patterns detected",
            sanitized_input=text,
        )

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
