"""End-to-end persistence and deletion tests for the local PDF library."""

from pathlib import Path

from scholar_agent.application.dtos.documents import (
    DeleteDocumentRequest,
    IngestDocumentRequest,
)
from scholar_agent.application.output_ports.embedding_provider import IEmbeddingProvider
from scholar_agent.application.services.request_validation_service import (
    RequestValidationService,
)
from scholar_agent.application.use_cases.delete_document import DeleteDocumentUseCase
from scholar_agent.application.use_cases.ingest_document import IngestDocumentUseCase
from scholar_agent.infrastructure.adapters.faiss_repository import FAISSRepository
from scholar_agent.infrastructure.adapters.langchain_text_chunker import (
    LangChainTextChunker,
)
from scholar_agent.infrastructure.adapters.local_document_library import (
    LocalDocumentLibrary,
)
from scholar_agent.infrastructure.adapters.pymupdf_loader import PyMuPDFLoader
from scholar_agent.infrastructure.adapters.sqlite_document_repository import (
    SQLiteDocumentRepository,
)

FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "evaluation"
    / "lecture_03_linear_regression.pdf"
)


class DeterministicEmbeddingProvider(IEmbeddingProvider):
    """Provides non-zero local vectors without downloading model weights."""

    def embed(self, text: str) -> tuple[float, ...]:
        total = sum(ord(character) for character in text)
        return (1.0, float(len(text) + 1), float(total % 997 + 1))

    def embed_many(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple(self.embed(text) for text in texts)


def test_document_data_survives_reopen_and_is_fully_deleted(tmp_path: Path) -> None:
    """Ingestion and deletion keep every local store aligned after reopening."""
    library_path = tmp_path / "documents"
    catalog_path = tmp_path / "catalog.sqlite3"
    vector_path = tmp_path / "vectors"
    embedding_provider = DeterministicEmbeddingProvider()
    document_repository = SQLiteDocumentRepository(catalog_path)
    document_library = LocalDocumentLibrary(library_path)
    vector_store = FAISSRepository(vector_path)
    ingest_document = IngestDocumentUseCase(
        document_repository=document_repository,
        document_library=document_library,
        pdf_loader=PyMuPDFLoader(),
        text_chunker=LangChainTextChunker(chunk_size=800, chunk_overlap=120),
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        validation_service=RequestValidationService(),
        maximum_upload_bytes=50 * 1024 * 1024,
    )

    result = ingest_document.execute(
        IngestDocumentRequest(FIXTURE_PATH.name, FIXTURE_PATH.read_bytes()),
    )
    document_id = result.document.identifier
    stored_pdf = library_path / f"{document_id.value}.pdf"

    reopened_catalog = SQLiteDocumentRepository(catalog_path)
    reopened_vectors = FAISSRepository(vector_path)
    assert stored_pdf.exists()
    assert reopened_catalog.get_by_id(document_id) == result.document
    assert reopened_vectors.list_document_chunks(document_id)
    assert reopened_vectors.search(
        embedding_provider.embed("gradient descent"),
        document_ids=(document_id,),
    )

    result = DeleteDocumentUseCase(
        document_repository=reopened_catalog,
        document_library=LocalDocumentLibrary(library_path),
        vector_store=reopened_vectors,
    ).execute(DeleteDocumentRequest(document_id))

    assert result.deleted is True
    assert not stored_pdf.exists()
    assert reopened_catalog.get_by_id(document_id) is None
    assert reopened_vectors.list_document_chunks(document_id) == ()
