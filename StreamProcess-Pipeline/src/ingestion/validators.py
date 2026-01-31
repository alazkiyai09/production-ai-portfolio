"""
Input validators for ingestion pipeline.
"""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError


# ============================================================================
# Validation Errors
# ============================================================================

class ValidationErrorDetail(BaseModel):
    """Validation error detail."""
    field: str
    message: str
    value: Any = None


class ValidationResult(BaseModel):
    """Validation result."""
    is_valid: bool
    errors: List[ValidationErrorDetail] = Field(default_factory=list)


# ============================================================================
# Base Schema Validator
# ============================================================================

class SchemaValidator:
    """Base schema validator."""

    @staticmethod
    def validate(record: dict, schema: type[BaseModel]) -> ValidationResult:
        """
        Validate a record against a Pydantic schema.

        Args:
            record: Record to validate
            schema: Pydantic model class

        Returns:
            ValidationResult with validation status
        """
        try:
            schema(**record)
            return ValidationResult(is_valid=True)
        except ValidationError as e:
            errors = [
                ValidationErrorDetail(
                    field=".".join(str(loc) for loc in error["loc"]),
                    message=error["msg"],
                    value=error.get("input"),
                )
                for error in e.errors()
            ]
            return ValidationResult(is_valid=False, errors=errors)

    @staticmethod
    def validate_batch(records: List[dict], schema: type[BaseModel]) -> ValidationResult:
        """
        Validate a batch of records.

        Args:
            records: List of records to validate
            schema: Pydantic model class

        Returns:
            ValidationResult with all validation errors
        """
        all_errors = []

        for idx, record in enumerate(records):
            result = SchemaValidator.validate(record, schema)
            if not result.is_valid:
                for error in result.errors:
                    error.field = f"[{idx}].{error.field}"
                all_errors.extend(result.errors)

        return ValidationResult(is_valid=len(all_errors) == 0, errors=all_errors)


# ============================================================================
# AdTech Event Validator
# ============================================================================

class AdTechEventValidator(SchemaValidator):
    """Validator for AdTech events."""

    REQUIRED_FIELDS = [
        "event_id",
        "event_type",
        "timestamp",
        "campaign_id",
        "user_id",
        "content",
    ]

    VALID_EVENT_TYPES = ["impression", "click", "conversion"]

    @staticmethod
    def validate_event_type(event_type: str) -> Tuple[bool, Optional[str]]:
        """
        Validate event type.

        Args:
            event_type: Event type to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if event_type not in AdTechEventValidator.VALID_EVENT_TYPES:
            return (
                False,
                f"Invalid event_type '{event_type}'. Must be one of: {AdTechEventValidator.VALID_EVENT_TYPES}"
            )
        return True, None

    @staticmethod
    def validate_content(content: str, min_length: int = 1, max_length: int = 10000) -> Tuple[bool, Optional[str]]:
        """
        Validate content field.

        Args:
            content: Content to validate
            min_length: Minimum length
            max_length: Maximum length

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not content or not content.strip():
            return False, "Content cannot be empty or whitespace"

        content_length = len(content)
        if content_length < min_length:
            return False, f"Content too short: {content_length} < {min_length}"

        if content_length > max_length:
            return False, f"Content too long: {content_length} > {max_length}"

        return True, None

    @staticmethod
    def validate_metadata(metadata: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate metadata for sensitive fields.

        Args:
            metadata: Metadata dictionary

        Returns:
            Tuple of (is_valid, error_message)
        """
        sensitive_keywords = [
            "email", "phone", "ssn", "social_security",
            "credit_card", "cc_number", "password", "secret",
        ]

        found_sensitive = []
        for key in metadata.keys():
            key_lower = key.lower()
            if any(keyword in key_lower for keyword in sensitive_keywords):
                found_sensitive.append(key)

        if found_sensitive:
            return (
                False,
                f"Potentially sensitive fields in metadata: {found_sensitive}"
            )

        return True, None

    @classmethod
    def validate_record(cls, record: Dict[str, Any]) -> ValidationResult:
        """
        Validate an AdTech event record with custom rules.

        Args:
            record: Event record to validate

        Returns:
            ValidationResult
        """
        errors = []

        # Check required fields
        for field in cls.REQUIRED_FIELDS:
            if field not in record:
                errors.append(
                    ValidationErrorDetail(
                        field=field,
                        message=f"Missing required field: {field}"
                    )
                )

        if errors:
            return ValidationResult(is_valid=False, errors=errors)

        # Validate event_type
        if "event_type" in record:
            is_valid, error = cls.validate_event_type(record["event_type"])
            if not is_valid:
                errors.append(
                    ValidationErrorDetail(field="event_type", message=error)
                )

        # Validate content
        if "content" in record:
            is_valid, error = cls.validate_content(record["content"])
            if not is_valid:
                errors.append(
                    ValidationErrorDetail(field="content", message=error, value=record["content"])
                )

        # Validate metadata
        if "metadata" in record and isinstance(record["metadata"], dict):
            is_valid, error = cls.validate_metadata(record["metadata"])
            if not is_valid:
                errors.append(
                    ValidationErrorDetail(field="metadata", message=error)
                )

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    @classmethod
    def validate_batch(cls, records: List[Dict[str, Any]]) -> ValidationResult:
        """
        Validate a batch of AdTech event records.

        Args:
            records: List of event records

        Returns:
            ValidationResult
        """
        all_errors = []

        for idx, record in enumerate(records):
            result = cls.validate_record(record)
            if not result.is_valid:
                for error in result.errors:
                    error.field = f"[{idx}].{error.field}"
                all_errors.extend(result.errors)

        return ValidationResult(is_valid=len(all_errors) == 0, errors=all_errors)
