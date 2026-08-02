"""Privacy-preserving source identities used by learner evidence."""

from dataclasses import dataclass

from scholar_agent.domain.value_objects.document_id import DocumentId
from scholar_agent.domain.value_objects.source_reference import SourceReference


@dataclass(frozen=True, slots=True)
class CitationIdentity:
    """A source location without an excerpt or other source text."""

    document_id: DocumentId
    chunk_id: str
    page_number: int | None

    def __post_init__(self) -> None:
        if not self.chunk_id.strip():
            raise ValueError("Citation chunk_id must not be blank.")

    @classmethod
    def from_reference(cls, reference: SourceReference) -> "CitationIdentity":
        """Strip a runtime citation down to its stable identity."""
        return cls(reference.document_id, reference.chunk_id, reference.page_number)
