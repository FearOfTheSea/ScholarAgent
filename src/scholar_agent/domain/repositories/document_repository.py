"""Document repository contract."""

from abc import ABC, abstractmethod

from scholar_agent.domain.entities.document import Document
from scholar_agent.domain.value_objects.document_id import DocumentId


class DocumentRepository(ABC):
    """Provides access to documents without prescribing storage."""

    @abstractmethod
    def save(self, document: Document) -> None:
        """Persist a document."""

    @abstractmethod
    def get_by_id(self, document_id: DocumentId) -> Document | None:
        """Return a document when it exists."""

    @abstractmethod
    def list_all(self) -> tuple[Document, ...]:
        """Return all known documents in display order."""

    @abstractmethod
    def delete(self, document_id: DocumentId) -> bool:
        """Delete a document record and report whether it existed."""
