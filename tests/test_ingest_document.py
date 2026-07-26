"""Tests for the document-ingestion application boundary."""

from pathlib import Path

from scholar_agent.application.dtos.documents import IngestDocumentRequest
from scholar_agent.application.dtos.retrieval import (
    DocumentChunk,
    LoadedPage,
    RetrievedChunk,
)
from scholar_agent.application.output_ports.document_library import IDocumentLibrary
from scholar_agent.application.output_ports.embedding_provider import IEmbeddingProvider
from scholar_agent.application.output_ports.pdf_loader import IPDFLoader
from scholar_agent.application.output_ports.text_chunker import ITextChunker
from scholar_agent.application.output_ports.vector_store import IVectorStore
from scholar_agent.application.services.request_validation_service import (
    RequestValidationService,
)
from scholar_agent.application.use_cases.ingest_document import IngestDocumentUseCase
from scholar_agent.domain.entities.document import Document
from scholar_agent.domain.repositories.document_repository import DocumentRepository
from scholar_agent.domain.value_objects.document_id import DocumentId


class MemoryDocumentRepository(DocumentRepository):
    """In-memory document catalog for application tests."""

    def __init__(self) -> None:
        self.document: Document | None = None

    def save(self, document: Document) -> None:
        self.document = document

    def get_by_id(self, document_id: DocumentId) -> Document | None:
        if self.document and self.document.identifier == document_id:
            return self.document
        return None

    def list_all(self) -> tuple[Document, ...]:
        return (self.document,) if self.document else ()

    def delete(self, document_id: DocumentId) -> bool:
        if self.get_by_id(document_id) is None:
            return False
        self.document = None
        return True


class MemoryLibrary(IDocumentLibrary):
    """Captures stored source content without writing a PDF."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.deleted = False

    def store(
        self,
        document_id: DocumentId,
        original_filename: str,
        content: bytes,
    ) -> Path:
        return self.path

    def delete(self, document_id: DocumentId) -> None:
        self.deleted = True


class FakePDFLoader(IPDFLoader):
    """Returns one extractable page."""

    def load(self, document_id: DocumentId, file_path: Path) -> tuple[LoadedPage, ...]:
        return (LoadedPage(document_id, 1, "Local study material."),)


class FakeChunker(ITextChunker):
    """Creates one source-aware chunk."""

    def chunk(self, pages: tuple[LoadedPage, ...]) -> tuple[DocumentChunk, ...]:
        page = pages[0]
        return (DocumentChunk(page.document_id, page.content, 1, None, "chunk-1", 0),)


class FakeEmbeddingProvider(IEmbeddingProvider):
    """Returns deterministic non-zero embeddings."""

    def embed(self, text: str) -> tuple[float, ...]:
        return (1.0, 0.0)

    def embed_many(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple(self.embed(text) for text in texts)


class FakeVectorStore(IVectorStore):
    """Captures indexed chunks for application tests."""

    def __init__(self) -> None:
        self.chunks: tuple[DocumentChunk, ...] = ()

    def add(
        self,
        chunks: tuple[DocumentChunk, ...],
        embeddings: tuple[tuple[float, ...], ...],
    ) -> None:
        self.chunks = chunks

    def search(
        self,
        embedding: tuple[float, ...],
        limit: int = 5,
        document_ids: tuple[DocumentId, ...] = (),
    ) -> tuple[RetrievedChunk, ...]:
        return ()

    def list_document_chunks(
        self,
        document_id: DocumentId,
    ) -> tuple[DocumentChunk, ...]:
        return self.chunks

    def get_chunk(
        self,
        document_id: DocumentId,
        chunk_id: str,
    ) -> DocumentChunk | None:
        return None

    def delete_document(self, document_id: DocumentId) -> None:
        self.chunks = ()


def test_ingest_document_persists_source_catalog_and_vectors(tmp_path: Path) -> None:
    """Ingestion coordinates external ports without framework dependencies."""
    repository = MemoryDocumentRepository()
    vector_store = FakeVectorStore()
    use_case = IngestDocumentUseCase(
        document_repository=repository,
        document_library=MemoryLibrary(tmp_path / "source.pdf"),
        pdf_loader=FakePDFLoader(),
        text_chunker=FakeChunker(),
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        validation_service=RequestValidationService(),
        maximum_upload_bytes=1024,
    )

    result = use_case.execute(IngestDocumentRequest("notes.pdf", b"%PDF local"))

    assert result.document.title == "notes"
    assert repository.document == result.document
    assert vector_store.chunks[0].page_number == 1
