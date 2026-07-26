"""Local source-document library implementation."""

from pathlib import Path

from scholar_agent.application.output_ports.document_library import IDocumentLibrary
from scholar_agent.domain.value_objects.document_id import DocumentId


class LocalDocumentLibrary(IDocumentLibrary):
    """Stores original PDFs in a private local application directory."""

    def __init__(self, library_path: Path) -> None:
        self._library_path = library_path

    def store(
        self,
        document_id: DocumentId,
        original_filename: str,
        content: bytes,
    ) -> Path:
        """Persist the uploaded PDF using the document identifier as its name."""
        self._library_path.mkdir(parents=True, exist_ok=True)
        file_path = self._file_path(document_id)
        file_path.write_bytes(content)
        return file_path

    def delete(self, document_id: DocumentId) -> None:
        """Delete a stored PDF when it exists."""
        file_path = self._file_path(document_id)
        if file_path.exists():
            file_path.unlink()

    def _file_path(self, document_id: DocumentId) -> Path:
        return self._library_path / f"{document_id.value}.pdf"
