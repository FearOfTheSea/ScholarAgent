"""DTOs used by retrieval-related output ports."""

from dataclasses import dataclass

from scholar_agent.domain.value_objects.document_id import DocumentId


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A text chunk returned by a retriever."""

    document_id: DocumentId
    content: str
    page_number: int | None
    section: str | None
    chunk_id: str
    similarity_score: float


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A chunk ready for embedding and persistent storage."""

    document_id: DocumentId
    content: str
    page_number: int | None
    section: str | None
    chunk_id: str
    ordinal: int


@dataclass(frozen=True, slots=True)
class LoadedPage:
    """Text extracted from one source document page."""

    document_id: DocumentId
    page_number: int
    content: str
