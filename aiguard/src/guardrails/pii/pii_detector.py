"""
PII Detection and Redaction Module

Detects and redacts Personally Identifiable Information (PII) including:
- SSN, Credit Cards, Email, Phone, IP addresses
- Dates of birth, Passport numbers, Bank accounts
- Named entities (people, organizations, locations)
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

# Optional spaCy support
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class RedactionMode(Enum):
    """Redaction strategies."""
    FULL = "full"           # [PII_TYPE REDACTED]
    PARTIAL = "partial"     # XXX-XX-1234
    TOKEN = "token"         # Replace with consistent tokens
    MASK = "mask"           # **** characters
    HASH = "hash"           # Unique hash identifier


@dataclass
class PIIMatch:
    """Match information for detected PII."""
    pii_type: str
    value: str
    start: int
    end: int
    redacted_value: str
    confidence: float = 1.0


@dataclass
class PIIReport:
    """Comprehensive PII detection report."""
    has_pii: bool
    total_matches: int
    matches_by_type: Dict[str, int]
    matches: List[PIIMatch]
    redacted_text: str


class PIIDetector:
    """
    Comprehensive PII detection and redaction.

    Supports regex-based detection with optional spaCy NER enhancement.
    """

    # PII regex patterns
    PATTERNS = {
        # Social Security Number (US: XXX-XX-XXXX)
        "ssn": [
            re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),
            # SSN not starting with 000, 666, or 9xx
            re.compile(r"\b(?!000|666|9\d{2})\d{3}[-.\s]?(?!00)\d{2}[-.\s]?(?!0000)\d{4}\b"),
        ],

        # Credit Card Numbers (Visa, MC, Amex, Discover)
        "credit_card": [
            # Visa: 4XXX XXXX XXXX XXXX
            re.compile(r"\b4\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
            # MasterCard: 5XXX XXXX XXXX XXXX or 2XXX XXXX XXXX XXXX
            re.compile(r"\b(?:5[1-5]\d{2}|2[2-7]\d{2})[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
            # Amex: 3XXX XXXXXX XXXXX
            re.compile(r"\b3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}\b"),
            # Discover: 6011 XXXX XXXX XXXX
            re.compile(r"\b6011[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
            # Generic 13-19 digit card numbers
            re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        ],

        # Email addresses
        "email": [
            re.compile(
                r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
            ),
        ],

        # Phone numbers (US and international)
        "phone": [
            # US: (XXX) XXX-XXXX or XXX-XXX-XXXX or XXX XXX XXXX
            re.compile(r"\b(?:\+1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?[2-9]\d{2}[-.\s]?\d{4}\b"),
            # International: +XX XXX XXX XXXX
            re.compile(r"\b\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b"),
            # Generic: various formats
            re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        ],

        # IP addresses (IPv4)
        "ip_address": [
            re.compile(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"),
        ],

        # Date of Birth / Birth Date
        "dob": [
            # MM/DD/YYYY, MM-DD-YYYY, YYYY-MM-DD
            re.compile(r"\b(?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[-01])[-/](?:19|20)\d{2}\b"),
            re.compile(r"\b(?:19|20)\d{2}[-/](?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[-01])\b"),
            # DD/MM/YYYY, DD-MM-YYYY
            re.compile(r"\b(?:0[1-9]|[12]\d|3[-01])[-/](?:0[1-9]|1[0-2])[-/](?:19|20)\d{2}\b"),
            # Month DD, YYYY
            re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s.,]+(?:0[1-9]|[12]\d|3[01])[stndrh]{2}[\s.,]+(?:19|20)\d{2}\b", re.IGNORECASE),
        ],

        # Passport numbers (US and international formats)
        "passport": [
            # US: 9 digits
            re.compile(r"\b\d{9}\b"),  # Context-dependent, may have false positives
            # European format: 2 letters + digits
            re.compile(r"\b[A-Z]{2}\d{6,9}\b"),
        ],

        # Bank account numbers
        "bank_account": [
            # US routing + account
            re.compile(r"\b(?:routing|aba|account)\s*(?:number|#)?[:\s]+\d{6,17}\b", re.IGNORECASE),
            # Simple account number patterns
            re.compile(r"\baccount\s*(?:number|#)?[:\s]+\d{6,17}\b", re.IGNORECASE),
            # IBAN format
            re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
        ],

        # Driver's license (US format - varies by state)
        "drivers_license": [
            re.compile(r"\b(?:driver'?s?\s*license|dl|id)\s*(?:card)?\s*(?:number|#)?[:\s]+[A-Z0-9]{6,15}\b", re.IGNORECASE),
            # Generic alphanumeric license format
            re.compile(r"\b[A-Z]{1,2}[-.\s]?\d{6,12}\b"),
        ],

        # Medical identifiers
        "medical_id": [
            re.compile(r"\b(?:medical\s*record|mrn|patient\s*id)\s*(?:number|#)?[:\s]+\d{4,12}\b", re.IGNORECASE),
        ],

        # URL with personal info
        "url": [
            re.compile(r"\bhttps?://[^\s<>\"]+\b"),
        ],

        # MAC Address
        "mac_address": [
            re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"),
        ],

        # License plate
        "license_plate": [
            re.compile(r"\b[A-Z0-9]{2,3}[-\s]?[A-Z0-9]{2,5}\b"),
        ],
    }

    # Named entity labels from spaCy (when available)
    SPACY_PII_LABELS = {
        "PERSON": "person",
        "ORG": "organization",
        "GPE": "location",  # Geopolitical entity
        "LOC": "location",
        "DATE": "date",
        "TIME": "time",
        "EMAIL": "email",
        "PHONE": "phone",
        "CARDINAL": "number",  # May indicate ID numbers
        "QUANTITY": "quantity",
        "MONEY": "money",
        "NORP": "nationality/religion/political",
        "WORK_OF_ART": "creative_work",
        "LAW": "law",
        "LANGUAGE": "language",
        "EVENT": "event",
        "FAC": "facility",
        "PRODUCT": "product",
    }

    def __init__(
        self,
        redaction_mode: str = "full",
        use_spacy: bool = True,
        spacy_model: str = "en_core_web_sm",
    ) -> None:
        """
        Initialize PII detector.

        Args:
            redaction_mode: One of 'full', 'partial', 'token', 'mask', 'hash'
            use_spacy: Whether to use spaCy NER (if available)
            spacy_model: Name of spaCy model to use
        """
        self.redaction_mode = RedactionMode(redaction_mode)
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self._nlp = None

        if self.use_spacy:
            try:
                self._nlp = spacy.load(spacy_model)
            except OSError:
                # Model not installed, fallback to regex only
                self.use_spacy = False

        # Track tokens for consistent replacement
        self._token_map: Dict[str, str] = {}
        self._token_counters: Dict[str, int] = {}

    def _reset_tokens(self) -> None:
        """Reset token tracking for new text."""
        self._token_map = {}
        self._token_counters = {}

    def _get_token(self, pii_type: str, value: str) -> str:
        """
        Get consistent token for PII value.

        Args:
            pii_type: Type of PII
            value: Original PII value

        Returns:
            Consistent token
        """
        # Create key for this specific value
        key = f"{pii_type}:{value}"

        if key in self._token_map:
            return self._token_map[key]

        # Generate new token
        if pii_type not in self._token_counters:
            self._token_counters[pii_type] = 0

        self._token_counters[pii_type] += 1
        token = f"[{pii_type.upper()}_{self._token_counters[pii_type]}]"
        self._token_map[key] = token
        return token

    def _redact_value(self, pii_type: str, value: str) -> str:
        """
        Redact a single PII value based on redaction mode.

        Args:
            pii_type: Type of PII
            value: Original value

        Returns:
            Redacted value
        """
        if self.redaction_mode == RedactionMode.FULL:
            return f"[{pii_type.upper()} REDACTED]"

        elif self.redaction_mode == RedactionMode.PARTIAL:
            # Show last 4 characters for most types
            if len(value) > 4:
                if pii_type == "email":
                    # Show first 2 chars of username and domain
                    parts = value.split("@")
                    if len(parts) == 2:
                        username = parts[0]
                        domain = parts[1]
                        username_masked = username[:2] + "*" * (len(username) - 2)
                        domain_parts = domain.split(".")
                        if len(domain_parts) >= 2:
                            domain_masked = "*"*len(domain_parts[0]) + "." + ".".join(domain_parts[1:])
                            return f"{username_masked}@{domain_masked}"
                return "*" * (len(value) - 4) + value[-4:]
            return "*" * len(value)

        elif self.redaction_mode == RedactionMode.TOKEN:
            return self._get_token(pii_type, value)

        elif self.redaction_mode == RedactionMode.MASK:
            return "*" * len(value)

        elif self.redaction_mode == RedactionMode.HASH:
            # Simple hash (for demo - use proper hash in production)
            hash_val = str(hash(value) % 10000)
            return f"[{pii_type.upper()}_HASH:{hash_val}]"

        return f"[{pii_type.upper()} REDACTED]"

    def detect(self, text: str) -> Tuple[bool, List[PIIMatch]]:
        """
        Detect PII in text.

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (has_pii, list_of_matches)
        """
        if not text or not isinstance(text, str):
            return False, []

        matches: List[PIIMatch] = []

        # Regex-based detection
        for pii_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    # Check for duplicates/overlaps
                    is_duplicate = any(
                        m.start <= match.start() < m.end or m.start < match.end() <= m.end
                        for m in matches
                    )
                    if not is_duplicate:
                        value = match.group()
                        redacted = self._redact_value(pii_type, value)
                        matches.append(
                            PIIMatch(
                                pii_type=pii_type,
                                value=value,
                                start=match.start(),
                                end=match.end(),
                                redacted_value=redacted,
                            )
                        )

        # SpaCy NER enhancement (if available)
        if self.use_spacy and self._nlp:
            doc = self._nlp(text)
            for ent in doc.ents:
                if ent.label_ in self.SPACY_PII_LABELS:
                    pii_type = self.SPACY_PII_LABELS[ent.label_]

                    # Check for overlaps with regex matches
                    is_duplicate = any(
                        m.start <= ent.start_char < m.end or m.start < ent.end_char <= m.end
                        for m in matches
                    )
                    if not is_duplicate:
                        value = ent.text
                        redacted = self._redact_value(pii_type, value)
                        matches.append(
                            PIIMatch(
                                pii_type=pii_type,
                                value=value,
                                start=ent.start_char,
                                end=ent.end_char,
                                redacted_value=redacted,
                                confidence=0.9,  # NER confidence
                            )
                        )

        # Sort matches by position
        matches.sort(key=lambda m: m.start)

        return len(matches) > 0, matches

    def redact(self, text: str) -> str:
        """
        Redact PII from text.

        Args:
            text: Input text

        Returns:
            Text with PII redacted
        """
        self._reset_tokens()

        has_pii, matches = self.detect(text)

        if not has_pii:
            return text

        # Build redacted text by processing matches in reverse order
        # (to maintain correct indices)
        result = text
        for match in reversed(matches):
            result = result[: match.start] + match.redacted_value + result[match.end :]

        return result

    def get_pii_report(self, text: str) -> PIIReport:
        """
        Generate comprehensive PII report.

        Args:
            text: Input text

        Returns:
            PIIReport with detailed findings
        """
        self._reset_tokens()

        has_pii, matches = self.detect(text)
        redacted_text = self.redact(text)

        # Count by type
        matches_by_type: Dict[str, int] = {}
        for match in matches:
            matches_by_type[match.pii_type] = matches_by_type.get(match.pii_type, 0) + 1

        return PIIReport(
            has_pii=has_pii,
            total_matches=len(matches),
            matches_by_type=matches_by_type,
            matches=matches,
            redacted_text=redacted_text,
        )

    def get_summary(self, text: str) -> Dict:
        """
        Get summary of PII detection.

        Args:
            text: Input text

        Returns:
            Dictionary with summary statistics
        """
        report = self.get_pii_report(text)

        return {
            "has_pii": report.has_pii,
            "total_matches": report.total_matches,
            "pii_types_found": list(report.matches_by_type.keys()),
            "counts_by_type": report.matches_by_type,
            "redaction_mode": self.redaction_mode.value,
        }

    def validate_redaction(self, original: str, redacted: str) -> bool:
        """
        Validate that redaction was successful.

        Args:
            original: Original text
            redacted: Redacted text

        Returns:
            True if redaction appears successful
        """
        # Check that PII patterns are not in redacted text
        _, original_matches = self.detect(original)

        for match in original_matches:
            if match.value in redacted:
                return False

        return True

    def set_redaction_mode(self, mode: str) -> None:
        """
        Change redaction mode.

        Args:
            mode: New redaction mode ('full', 'partial', 'token', 'mask', 'hash')
        """
        self.redaction_mode = RedactionMode(mode)
        self._reset_tokens()
