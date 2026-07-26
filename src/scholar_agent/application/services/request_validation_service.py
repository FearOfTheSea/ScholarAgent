"""Validation service shared by future use cases."""

from scholar_agent.application.validators.common import (
    require_non_blank,
    require_positive,
)


class RequestValidationService:
    """Exposes explicit validation operations to application use cases."""

    def validate_text(self, value: str, field_name: str) -> str:
        """Validate a required text field."""
        return require_non_blank(value, field_name)

    def validate_count(self, value: int, field_name: str) -> int:
        """Validate a required positive count."""
        return require_positive(value, field_name)
