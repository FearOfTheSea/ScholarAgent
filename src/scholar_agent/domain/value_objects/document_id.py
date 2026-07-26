"""Document identity value object."""

from dataclasses import dataclass

from scholar_agent.domain.exceptions.domain_validation_error import (
    DomainValidationError,
)


@dataclass(frozen=True, slots=True)
class DocumentId:
    """A non-empty identifier for a document."""

    value: str

    def __post_init__(self) -> None:
        if not self.value.strip():
            message = "Document identifier must not be blank."
            raise DomainValidationError(message)
