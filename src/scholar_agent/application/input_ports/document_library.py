"""Input ports for local document-library actions."""

from abc import ABC, abstractmethod

from scholar_agent.application.dtos.documents import (
    DeleteDocumentRequest,
    DeleteDocumentResult,
    IngestDocumentRequest,
    IngestDocumentResult,
    ListDocumentsResult,
)


class IngestDocument(ABC):
    """Adds a PDF to the local study library."""

    @abstractmethod
    def execute(self, request: IngestDocumentRequest) -> IngestDocumentResult:
        """Ingest a PDF and its derived retrieval data."""


class ListDocuments(ABC):
    """Lists documents in the local study library."""

    @abstractmethod
    def execute(self) -> ListDocumentsResult:
        """Return locally available documents."""


class DeleteDocument(ABC):
    """Removes a PDF and its derived retrieval data."""

    @abstractmethod
    def execute(self, request: DeleteDocumentRequest) -> DeleteDocumentResult:
        """Delete a local document."""
