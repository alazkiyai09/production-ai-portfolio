"""
Data transformation utilities for StreamProcess-Pipeline.

Provides data cleaning, normalization, and enrichment.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Configuration
# ============================================================================

class TransformerConfig(BaseModel):
    """Configuration for data transformer."""
    max_content_length: int = Field(default=10000, description="Maximum content length")
    normalize_whitespace: bool = Field(default=True, description="Normalize whitespace")
    strip_html: bool = Field(default=True, description="Strip HTML tags")
    lowercase: bool = Field(default=False, description="Convert to lowercase")
    remove_special_chars: bool = Field(default=False, description="Remove special characters")
    extract_features: bool = Field(default=True, description="Extract derived features")


# ============================================================================
# Data Transformer
# ============================================================================

class DataTransformer:
    """
    Transform and clean event data.

    Features:
    - Text cleaning and normalization
    - HTML tag removal
    - Whitespace normalization
    - Feature extraction
    - Data validation
    """

    def __init__(self, config: Optional[TransformerConfig] = None):
        """
        Initialize data transformer.

        Args:
            config: Optional transformer configuration
        """
        self.config = config or TransformerConfig()

    def transform_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a single event.

        Args:
            event: Raw event data

        Returns:
            Transformed event
        """
        transformed = event.copy()

        # Transform content
        if "content" in event:
            transformed["content"] = self.transform_text(event["content"])

        # Extract features
        if self.config.extract_features:
            transformed["derived"] = self.extract_features(event)

        # Normalize metadata
        if "metadata" in event:
            transformed["metadata"] = self.normalize_metadata(event["metadata"])

        # Add content hash
        if "content" in transformed:
            from src.processing.worker import hash_content
            transformed["content_hash"] = hash_content(transformed["content"])

        return transformed

    def transform_batch(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform a batch of events.

        Args:
            events: List of raw events

        Returns:
            List of transformed events
        """
        return [self.transform_event(event) for event in events]

    def transform_text(self, text: str) -> str:
        """
        Clean and normalize text content.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Strip HTML if enabled
        if self.config.strip_html:
            text = self._strip_html(text)

        # Remove special characters if enabled
        if self.config.remove_special_chars:
            text = self._remove_special_chars(text)

        # Normalize whitespace if enabled
        if self.config.normalize_whitespace:
            text = self._normalize_whitespace(text)

        # Lowercase if enabled
        if self.config.lowercase:
            text = text.lower()

        # Truncate to max length
        if len(text) > self.config.max_content_length:
            text = text[: self.config.max_content_length]

        return text.strip()

    def extract_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract derived features from event.

        Args:
            event: Event data

        Returns:
            Dictionary of derived features
        """
        features = {}

        # Content features
        if "content" in event:
            content = event["content"]
            features["content_length"] = len(content)
            features["word_count"] = len(content.split())
            features["sentence_count"] = len(re.split(r"[.!?]+", content))
            features["avg_word_length"] = (
                sum(len(word) for word in content.split()) / features["word_count"]
                if features["word_count"] > 0
                else 0
            )

        # Timestamp features
        if "timestamp" in event:
            timestamp = event["timestamp"]
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            elif isinstance(timestamp, datetime):
                pass
            else:
                timestamp = datetime.utcnow()

            features["hour_of_day"] = timestamp.hour
            features["day_of_week"] = timestamp.weekday()
            features["is_weekend"] = timestamp.weekday() >= 5

        # Event type features
        if "event_type" in event:
            features["event_type_category"] = self._categorize_event_type(event["event_type"])

        return features

    def normalize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize metadata fields.

        Args:
            metadata: Raw metadata

        Returns:
            Normalized metadata
        """
        if not metadata:
            return {}

        normalized = {}

        for key, value in metadata.items():
            # Normalize key
            norm_key = key.lower().replace(" ", "_").replace("-", "_")

            # Normalize value based on type
            if isinstance(value, str):
                # Trim whitespace
                normalized[norm_key] = value.strip()
            elif isinstance(value, (int, float, bool, list, dict)):
                # Keep as-is
                normalized[norm_key] = value
            else:
                # Convert to string
                normalized[norm_key] = str(value)

        return normalized

    def _strip_html(self, text: str) -> str:
        """Strip HTML tags from text."""
        # Simple regex-based HTML tag removal
        # For production, consider using BeautifulSoup
        html_pattern = re.compile(r"<[^>]+>")
        return html_pattern.sub("", text)

    def _remove_special_chars(self, text: str) -> str:
        """Remove special characters, keeping alphanumeric and basic punctuation."""
        # Keep alphanumeric, spaces, and basic punctuation
        pattern = re.compile(r"[^a-zA-Z0-9\s.,!?;:'\"-]")
        return pattern.sub("", text)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        # Replace multiple whitespace with single space
        return re.sub(r"\s+", " ", text)

    def _categorize_event_type(self, event_type: str) -> str:
        """Categorize event type."""
        categories = {
            "impression": "view",
            "click": "interaction",
            "conversion": "action",
        }
        return categories.get(event_type.lower(), "unknown")


# ============================================================================
# Validation Helpers
# ============================================================================

class ValidationError(Exception):
    """Validation error."""
    pass


def validate_transformed_event(event: Dict[str, Any]) -> bool:
    """
    Validate a transformed event.

    Args:
        event: Transformed event

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    required_fields = ["event_id", "event_type", "timestamp", "campaign_id", "user_id", "content"]

    for field in required_fields:
        if field not in event:
            raise ValidationError(f"Missing required field: {field}")

        if not event[field]:
            raise ValidationError(f"Empty required field: {field}")

    # Validate content
    if len(event["content"]) == 0:
        raise ValidationError("Content cannot be empty")

    # Validate event type
    valid_types = ["impression", "click", "conversion"]
    if event["event_type"] not in valid_types:
        raise ValidationError(f"Invalid event_type: {event['event_type']}")

    return True


# ============================================================================
# Pipeline
# ============================================================================

class TransformPipeline:
    """
    Pipeline for transforming events.

    Chains multiple transformers together.
    """

    def __init__(self, transformers: List[DataTransformer]):
        """
        Initialize pipeline.

        Args:
            transformers: List of transformers to apply in order
        """
        self.transformers = transformers

    def transform(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply all transformers to event.

        Args:
            event: Raw event

        Returns:
            Transformed event
        """
        result = event
        for transformer in self.transformers:
            result = transformer.transform_event(result)
        return result

    def transform_batch(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply all transformers to batch.

        Args:
            events: List of raw events

        Returns:
            List of transformed events
        """
        return [self.transform(event) for event in events]


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "TransformerConfig",
    "DataTransformer",
    "TransformPipeline",
    "ValidationError",
    "validate_transformed_event",
]
