"""
Output Filtering Module

Filters LLM outputs to prevent:
- Data leakage (system prompts, training data)
- Hallucination indicators
- Toxic/harmful content
- PII disclosure
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

# Import GuardResult and ThreatType from prompt_injection module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompt_injection.prompt_injection import GuardResult, ThreatType
from pii.pii_detector import PIIDetector


class OutputThreatType(Enum):
    """Threat types specific to output filtering."""
    SAFE = "safe"
    DATA_LEAKAGE = "data_leakage"
    SYSTEM_PROMPT_LEAK = "system_prompt_leak"
    TRAINING_DATA_LEAK = "training_data_leak"
    HALLUCINATION = "hallucination"
    TOXICITY = "toxicity"
    HARMFUL_CONTENT = "harmful_content"
    PII_LEAKAGE = "pii_leakage"
    INSTRUCTION_LEAKAGE = "instruction_leakage"
    CONFIDENCE_ISSUE = "confidence_issue"


@dataclass
class OutputIssue:
    """Specific issue found in output."""
    issue_type: OutputThreatType
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    matched_text: Optional[str] = None
    position: Optional[Tuple[int, int]] = None


class OutputGuard:
    """
    Comprehensive output filtering and sanitization.

    Detects and prevents various issues in LLM outputs.
    """

    # Data leakage patterns
    SYSTEM_PROMPT_INDICATORS = [
        "system prompt",
        "system message",
        "my instructions",
        "my programming",
        "my guidelines",
        "as an ai language model",
        "as a large language model",
        "i am trained to",
        "i am programmed to",
        "my training data",
        "my developers",
        "openai taught me",
        "anthropic trained me",
        "my system message says",
        "according to my instructions",
        "per my guidelines",
        "my safety protocols",
        "i cannot because i am",
        "i'm an ai and i",
    ]

    TRAINING_DATA_PATTERNS = [
        # Training set references
        r"(i\s+(was\s+)?trained\s+on|learned\s+from|my\s+training\s+(data|set))",
        r"(pre.?trained|fine.?tuned)\s+on",
        r"(based\s+on\s+my\s+training|in\s+my\s+training)",
        # Dataset references
        r"(common\s+crawl|wikipedia|bookscorpus|webtext)",
        r"(the\s+dataset|i\s+was\s+trained)",
        # Memorization indicators
        r"(this\s+is\s+from\s+my\s+training|i\s+recall\s+from\s+training)",
        r"(my\s+weights|my\s+parameters|my\s+architecture)",
    ]

    INSTRUCTION_LEAKAGE_PATTERNS = [
        # Step-by-step instruction leaks
        r"\d+\.\s*(first|step|next|then|after\s+that)(\s+is)?:?\s*[a-z]{20,}",
        # Template/formula leaks
        r"(template|formula|pattern|structure)(\s+is)?:?\s*[a-z]{30,}",
        # Internal process leaks
        r"(internally|i\s+process|my\s+(internal|process)(ing))",
        # Chain of thought leaks
        r"(chain\s+of\s+thought|reasoning\s+steps|let\s+me\s+think)",
    ]

    # Hallucination indicators
    UNCERTAINTY_PHRASES = [
        "i'm not sure",
        "i don't know",
        "i am not certain",
        "i believe",
        "i think",
        "it seems",
        "possibly",
        "probably",
        "might be",
        "could be",
        "appears to be",
        "as far as i know",
        "to the best of my knowledge",
        "i may be wrong",
        "i'm uncertain",
        "it's unclear",
        "i cannot confirm",
        "i don't have information",
        "i don't have access",
        "i'm not familiar with",
        "i cannot verify",
        "not entirely sure",
        "if i recall correctly",
        "if i remember right",
    ]

    OVERCONFIDENCE_PHRASES = [
        "definitely",
        "absolutely",
        "certainly",
        "without doubt",
        "without question",
        "100% sure",
        "completely sure",
        "guaranteed",
        "i guarantee",
        "i'm certain",
        "i'm positive",
        "there's no doubt",
        "no question about it",
    ]

    CONTRADICTION_MARKERS = [
        (r"(?:however|but|on the other hand|conversely|although).*?{prev_phrase}", "contrast"),
        (r"(?:actually|in fact|really).*?(?:not|never|don't|didn't).*?(?:said|stated|mentioned)", "correction"),
    ]

    # Toxicity and harmful content patterns
    TOXICITY_PATTERNS = {
        "profanity": [
            r"\b(fuck|shit|damn|hell|bitch|bastard|ass|asshole|dick|piss)\b",
        ],
        "hate_speech": [
            r"\s(all\s+\w+\s+(should|must|deserve\s+to)\s+(die|burn|be\s+killed|be\s+destroyed))",
            r"\s(those?\s+\w+\s+(are|should\s+be))\s+(subhuman|animals|monsters|parasites)",
            r"\s(\w+\s+(is\s+)?(inferior|superior)\s+to\s+\w+)",
        ],
        "violence": [
            r"\b(kill|murder|torture|rape|assault|attack|beat|violence)\b.*\b(you|someone|every\w+)\b",
            r"\b(how\s+to\s+(kill|murder|hurt|harm|cause\s+harm))\b",
            r"\b(build|create|make)\s+(bomb|weapon|poison|explosive)\b",
        ],
        "self_harm": [
            r"\b(how\s+to\s+(kill|hurt)\s+myself|commit\s+suicide|end\s+my\s+life)\b",
            r"\b(want\s+to\s+die|want\s+to\s+kill\s+myself|suicidal)\b",
        ],
        "illegal_acts": [
            r"\b(how\s+to\s+(steal|rob|shoplift|burglarize))\b",
            r"\b(make\s+(fake|forged|counterfeit))\s+(money|documents|id)\b",
            r"\b(how\s+to\s+evade|avoid\s+)(police|arrest|detection)\b",
        ],
    }

    def __init__(
        self,
        pii_detector: Optional[PIIDetector] = None,
        strict_mode: bool = False,
    ) -> None:
        """
        Initialize output guard.

        Args:
            pii_detector: PIIDetector instance for PII scanning
            strict_mode: Enable stricter filtering
        """
        self.pii_detector = pii_detector or PIIDetector(redaction_mode="full")
        self.strict_mode = strict_mode

        # Compile regex patterns
        self._training_data_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.TRAINING_DATA_PATTERNS
        ]
        self._instruction_leakage_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.INSTRUCTION_LEAKAGE_PATTERNS
        ]
        self._toxicity_patterns = {
            category: [re.compile(p, re.IGNORECASE) for p in patterns]
            for category, patterns in self.TOXICITY_PATTERNS.items()
        }

    def check_data_leakage(
        self, output: str, system_prompt: Optional[str] = None
    ) -> GuardResult:
        """
        Check for data leakage in output.

        Args:
            output: LLM output text
            system_prompt: Optional system prompt to check against

        Returns:
            GuardResult with findings
        """
        issues: List[str] = []
        confidence = 0.0

        # Check system prompt indicators
        output_lower = output.lower()
        for indicator in self.SYSTEM_PROMPT_INDICATORS:
            if indicator in output_lower:
                issues.append(f"System prompt indicator: '{indicator}'")
                confidence = max(confidence, 0.7)

        # Check training data patterns
        for pattern in self._training_data_patterns:
            if pattern.search(output):
                match = pattern.search(output).group()
                issues.append(f"Training data reference: '{match[:100]}'")
                confidence = max(confidence, 0.75)

        # Check instruction leakage patterns
        for pattern in self._instruction_leakage_patterns:
            if pattern.search(output):
                match = pattern.search(output).group()
                issues.append(f"Instruction leakage: '{match[:100]}'")
                confidence = max(confidence, 0.8)

        # Check against system prompt if provided
        if system_prompt:
            # Look for direct quotes or near-matches
            system_sentences = re.split(r"[.!?]+", system_prompt)
            for sentence in system_sentences:
                if len(sentence.strip()) > 20:
                    # Check for exact matches
                    if sentence.strip().lower() in output_lower:
                        issues.append(f"Direct system prompt leakage detected")
                        confidence = 1.0
                        break

                    # Check for substantial overlaps (50%+ similarity)
                    words = set(sentence.lower().split())
                    output_words = set(output_lower.split())
                    overlap = len(words & output_words) / max(len(words), 1)

                    if overlap > 0.5:
                        issues.append(f"Substantial system prompt overlap: {overlap:.1%}")
                        confidence = max(confidence, 0.85)

        # Determine threat type
        if confidence > 0:
            threat_type = ThreatType.PROMPT_INJECTION
        else:
            threat_type = ThreatType.SAFE

        # Sanitize if issues found
        sanitized = output if confidence == 0 else self._sanitize_data_leakage(output)

        return GuardResult(
            is_safe=confidence == 0,
            threat_type=threat_type,
            confidence=confidence,
            details="; ".join(issues) if issues else "No data leakage detected",
            sanitized_input=sanitized,
        )

    def check_hallucination_indicators(self, output: str) -> GuardResult:
        """
        Check for hallucination and confidence indicators.

        Args:
            output: LLM output text

        Returns:
            GuardResult with findings
        """
        issues: List[str] = []
        confidence = 0.0
        output_lower = output.lower()

        # Check uncertainty phrases
        uncertainty_count = 0
        for phrase in self.UNCERTAINTY_PHRASES:
            if phrase in output_lower:
                uncertainty_count += 1
                issues.append(f"Uncertainty phrase: '{phrase}'")

        if uncertainty_count > 0:
            confidence = min(0.5 + (uncertainty_count * 0.1), 0.9)

        # Check overconfidence (can indicate hallucination)
        overconfidence_count = 0
        for phrase in self.OVERCONFIDENCE_PHRASES:
            if phrase in output_lower:
                overconfidence_count += 1
                issues.append(f"Overconfidence phrase: '{phrase}'")

        if overconfidence_count > 2:
            confidence = max(confidence, 0.6)
            issues.append(f"Excessive overconfidence detected ({overconfidence_count} instances)")

        # Check for contradictions
        sentences = re.split(r"[.!?]+", output)
        contradictions = self._detect_contradictions(sentences)
        if contradictions > 0:
            confidence = max(confidence, 0.7)
            issues.append(f"Potential contradictions detected: {contradictions}")

        # Check for factuality markers
        factuality_issues = self._check_factuality_issues(output)
        if factuality_issues:
            confidence = max(confidence, 0.65)
            issues.extend(factuality_issues)

        return GuardResult(
            is_safe=confidence < 0.7,  # Allow some uncertainty
            threat_type=ThreatType.PROMPT_INJECTION,
            confidence=confidence,
            details="; ".join(issues) if issues else "No hallucination indicators",
            sanitized_input=output,
        )

    def _detect_contradictions(self, sentences: List[str]) -> int:
        """
        Detect potential contradictions in text.

        Args:
            sentences: List of sentences

        Returns:
            Number of potential contradictions
        """
        contradictions = 0
        sentence_lower = [s.lower().strip() for s in sentences if len(s.strip()) > 10]

        # Look for pairs that contradict each other
        contradiction_patterns = [
            (r"\b(is|are)\b", r"\b(is not|are not|isn't|aren't)\b"),
            (r"\bwill\b", r"\bwon't|will not\b"),
            (r"\bcan\b", r"\bcan't|cannot\b"),
            (r"\balways\b", r"\bnever|sometimes\b"),
            (r"\btrue\b", r"\bfalse\b"),
        ]

        for i, sent1 in enumerate(sentence_lower):
            for sent2 in sentence_lower[i + 1 : i + 4]:  # Check next few sentences
                for pos_pattern, neg_pattern in contradiction_patterns:
                    if re.search(pos_pattern, sent1) and re.search(neg_pattern, sent2):
                        # Check for similar subjects/topics
                        words1 = set(re.findall(r"\b\w+\b", sent1))
                        words2 = set(re.findall(r"\b\w+\b", sent2))
                        overlap = len(words1 & words2) / max(len(words1), len(words2), 1)

                        if overlap > 0.2:
                            contradictions += 1

        return contradictions

    def _check_factuality_issues(self, output: str) -> List[str]:
        """
        Check for common factuality issues.

        Args:
            output: Text to check

        Returns:
            List of issues found
        """
        issues = []
        output_lower = output.lower()

        # Check for speculative language about facts
        if re.search(r"\b(in \d{4}|by \d{4}|in the future)\b.*\b(will|is going to)\b", output_lower):
            issues.append("Speculative future predictions presented as fact")

        # Check for over-specific numerical claims without sources
        if re.search(r"\b\d{1,3}(\.\d+)?\s*%\b", output):
            # Check if no source/citation nearby
            if not re.search(r"(according to|source|citation|study|research)", output_lower):
                issues.append("Specific statistics without attribution")

        # Check for definitive statements about uncertain topics
        uncertain_topics = [
            "alien",
            "afterlife",
            "consciousness",
            "meaning of life",
            "god's existence",
        ]
        for topic in uncertain_topics:
            if topic in output_lower:
                if any(word in output_lower for word in ["definitely", "certainly", "absolutely", "is"]):
                    issues.append(f"Definitive claim about uncertain topic: {topic}")

        return issues

    def check_toxicity(self, output: str) -> GuardResult:
        """
        Check for toxic and harmful content.

        Args:
            output: LLM output text

        Returns:
            GuardResult with findings
        """
        issues: List[str] = []
        total_score = 0.0
        max_severity = "low"

        for category, patterns in self._toxicity_patterns.items():
            category_matches = 0
            for pattern in patterns:
                matches = pattern.findall(output)
                if matches:
                    category_matches += len(matches)

            if category_matches > 0:
                issues.append(f"{category}: {category_matches} match(es)")

                # Score based on category severity
                severity_scores = {
                    "profanity": 0.3,
                    "hate_speech": 1.0,
                    "violence": 0.9,
                    "self_harm": 1.0,
                    "illegal_acts": 0.8,
                }
                total_score += severity_scores.get(category, 0.5) * min(category_matches, 3) / 3

                # Update max severity
                severity_mapping = {
                    "profanity": "low",
                    "illegal_acts": "high",
                    "violence": "high",
                    "hate_speech": "critical",
                    "self_harm": "critical",
                }
                current_severity = severity_mapping.get(category, "medium")
                severity_order = ["low", "medium", "high", "critical"]
                if severity_order.index(current_severity) > severity_order.index(max_severity):
                    max_severity = current_severity

        # In strict mode, even low severity is flagged
        threshold = 0.0 if self.strict_mode else 0.5

        return GuardResult(
            is_safe=total_score < threshold,
            threat_type=ThreatType.PROMPT_INJECTION,
            confidence=min(total_score, 1.0),
            details=f"Toxicity detected [{max_severity}]: " + "; ".join(issues) if issues else "No toxicity",
            sanitized_input=self._sanitize_toxicity(output) if total_score >= threshold else output,
        )

    def check_pii(self, output: str) -> GuardResult:
        """
        Check for PII in output.

        Args:
            output: LLM output text

        Returns:
            GuardResult with findings
        """
        has_pii, matches = self.pii_detector.detect(output)

        if has_pii:
            # Group by type
            pii_by_type = {}
            for match in matches:
                pii_by_type[match.pii_type] = pii_by_type.get(match.pii_type, 0) + 1

            details = ", ".join(f"{ptype}: {count}" for ptype, count in pii_by_type.items())
            redacted = self.pii_detector.redact(output)

            return GuardResult(
                is_safe=False,
                threat_type=ThreatType.PROMPT_INJECTION,
                confidence=0.9,
                details=f"PII detected: {details}",
                sanitized_input=redacted,
            )

        return GuardResult(
            is_safe=True,
            threat_type=ThreatType.SAFE,
            confidence=1.0,
            details="No PII detected",
            sanitized_input=output,
        )

    def filter_output(self, output: str, system_prompt: Optional[str] = None) -> Tuple[str, List[GuardResult]]:
        """
        Run all output filters and return sanitized output.

        Args:
            output: LLM output text
            system_prompt: Optional system prompt for leakage detection

        Returns:
            Tuple of (sanitized_output, list_of_guard_results)
        """
        results: List[GuardResult] = []
        current_output = output

        # Run all checks
        data_leakage_result = self.check_data_leakage(current_output, system_prompt)
        results.append(data_leakage_result)
        current_output = data_leakage_result.sanitized_input or current_output

        hallucination_result = self.check_hallucination_indicators(current_output)
        results.append(hallucination_result)

        toxicity_result = self.check_toxicity(current_output)
        results.append(toxicity_result)
        if not toxicity_result.is_safe:
            current_output = toxicity_result.sanitized_input or current_output

        pii_result = self.check_pii(current_output)
        results.append(pii_result)
        current_output = pii_result.sanitized_input or current_output

        return current_output, results

    def get_output_report(self, output: str, system_prompt: Optional[str] = None) -> dict:
        """
        Generate comprehensive output safety report.

        Args:
            output: LLM output text
            system_prompt: Optional system prompt

        Returns:
            Dictionary with detailed report
        """
        sanitized, results = self.filter_output(output, system_prompt)

        return {
            "original_length": len(output),
            "sanitized_length": len(sanitized),
            "is_safe": all(r.is_safe for r in results),
            "issues_found": len([r for r in results if not r.is_safe]),
            "results": {
                "data_leakage": {
                    "is_safe": results[0].is_safe,
                    "confidence": results[0].confidence,
                    "details": results[0].details,
                },
                "hallucination": {
                    "is_safe": results[1].is_safe,
                    "confidence": results[1].confidence,
                    "details": results[1].details,
                },
                "toxicity": {
                    "is_safe": results[2].is_safe,
                    "confidence": results[2].confidence,
                    "details": results[2].details,
                },
                "pii": {
                    "is_safe": results[3].is_safe,
                    "confidence": results[3].confidence,
                    "details": results[3].details,
                },
            },
            "sanitized_output": sanitized,
        }

    def _sanitize_data_leakage(self, text: str) -> str:
        """Remove data leakage from text."""
        sanitized = text

        # Remove system prompt indicators
        for indicator in self.SYSTEM_PROMPT_INDICATORS:
            sanitized = re.sub(
                re.escape(indicator),
                "[SYSTEM REFERENCE REMOVED]",
                sanitized,
                flags=re.IGNORECASE,
            )

        return sanitized

    def _sanitize_toxicity(self, text: str) -> str:
        """Remove toxic content from text."""
        sanitized = text

        # Replace toxic words
        for category, patterns in self._toxicity_patterns.items():
            if category in ["profanity", "hate_speech"]:
                for pattern in patterns:
                    sanitized = pattern.sub("[REDACTED]", sanitized)

        return sanitized
