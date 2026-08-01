"""Stable source location values used by every cited study result."""

from dataclasses import dataclass

from scholar_agent.domain.value_objects.document_id import DocumentId


@dataclass(frozen=True, slots=True)
class SourceReference:
    """A source location inside the session's selected document."""

    document_id: DocumentId
    chunk_id: str
    page_number: int | None
    excerpt: str
