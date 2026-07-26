"""Local source-document library port."""

from abc import ABC, abstractmethod
from pathlib import Path

from scholar_agent.domain.value_objects.document_id import DocumentId


class IDocumentLibrary(ABC):
    """Stores original source documents on the local machine."""

    @abstractmethod
    def store(
        self,
        document_id: DocumentId,
        original_filename: str,
        content: bytes,
    ) -> Path:
        """Persist a source document and return its local path."""

    @abstractmethod
    def delete(self, document_id: DocumentId) -> None:
        """Remove a source document when it exists."""
