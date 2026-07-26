"""PDF-loader port."""

from abc import ABC, abstractmethod
from pathlib import Path

from scholar_agent.application.dtos.retrieval import LoadedPage
from scholar_agent.domain.value_objects.document_id import DocumentId


class IPDFLoader(ABC):
    """Loads textual pages from a PDF document."""

    @abstractmethod
    def load(self, document_id: DocumentId, file_path: Path) -> tuple[LoadedPage, ...]:
        """Extract pages from a PDF."""
