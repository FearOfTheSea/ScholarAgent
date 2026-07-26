"""Domain-specific exceptions."""

from scholar_agent.domain.exceptions.document_not_found_error import (
    DocumentNotFoundError,
)
from scholar_agent.domain.exceptions.document_processing_error import (
    DocumentProcessingError,
)
from scholar_agent.domain.exceptions.domain_validation_error import (
    DomainValidationError,
)

__all__ = [
    "DomainValidationError",
    "DocumentNotFoundError",
    "DocumentProcessingError",
]
