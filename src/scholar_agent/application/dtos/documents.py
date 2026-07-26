"""Requests and results for local document-library actions."""

from dataclasses import dataclass

from scholar_agent.domain.entities.document import Document
from scholar_agent.domain.value_objects.document_id import DocumentId


@dataclass(frozen=True, slots=True)
class IngestDocumentRequest:
    """A PDF submitted to the local study library."""

    original_filename: str
    content: bytes


@dataclass(frozen=True, slots=True)
class DeleteDocumentRequest:
    """Request to remove a document and all local derived data."""

    document_id: DocumentId


@dataclass(frozen=True, slots=True)
class IngestDocumentResult:
    """A successfully ingested document."""

    document: Document


@dataclass(frozen=True, slots=True)
class ListDocumentsResult:
    """Documents available in the local library."""

    documents: tuple[Document, ...]


@dataclass(frozen=True, slots=True)
class DeleteDocumentResult:
    """The result of a local document deletion."""

    document_id: DocumentId
    deleted: bool
