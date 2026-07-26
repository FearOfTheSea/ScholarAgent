"""PyMuPDF implementation of the PDF-loader port."""

from pathlib import Path

from scholar_agent.application.dtos.retrieval import LoadedPage
from scholar_agent.application.output_ports.pdf_loader import IPDFLoader
from scholar_agent.domain.value_objects.document_id import DocumentId


class PyMuPDFLoader(IPDFLoader):
    """Extracts text from local PDF pages with PyMuPDF."""

    def load(self, document_id: DocumentId, file_path: Path) -> tuple[LoadedPage, ...]:
        """Extract text from every page in a local PDF file."""
        import fitz  # type: ignore[import-untyped]

        try:
            with fitz.open(stream=file_path.read_bytes(), filetype="pdf") as pdf:
                return tuple(
                    LoadedPage(
                        document_id=document_id,
                        page_number=page_number,
                        content=page.get_text("text").strip(),
                    )
                    for page_number, page in enumerate(pdf, start=1)
                )
        except (OSError, RuntimeError) as error:
            message = f"Unable to extract text from '{file_path.name}'."
            raise ValueError(message) from error
