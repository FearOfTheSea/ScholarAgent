"""Ingest-document use case."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from scholar_agent.application.dtos.documents import (
    IngestDocumentRequest,
    IngestDocumentResult,
)
from scholar_agent.application.input_ports.document_library import IngestDocument
from scholar_agent.application.output_ports.document_library import IDocumentLibrary
from scholar_agent.application.output_ports.embedding_provider import IEmbeddingProvider
from scholar_agent.application.output_ports.pdf_loader import IPDFLoader
from scholar_agent.application.output_ports.text_chunker import ITextChunker
from scholar_agent.application.output_ports.vector_store import IVectorStore
from scholar_agent.application.services.request_validation_service import (
    RequestValidationService,
)
from scholar_agent.domain.entities.document import Document
from scholar_agent.domain.exceptions.document_processing_error import (
    DocumentProcessingError,
)
from scholar_agent.domain.repositories.document_repository import DocumentRepository
from scholar_agent.domain.value_objects.document_id import DocumentId


class IngestDocumentUseCase(IngestDocument):
    """Persists a PDF and its local retrieval representation."""

    def __init__(
        self,
        document_repository: DocumentRepository,
        document_library: IDocumentLibrary,
        pdf_loader: IPDFLoader,
        text_chunker: ITextChunker,
        embedding_provider: IEmbeddingProvider,
        vector_store: IVectorStore,
        validation_service: RequestValidationService,
        maximum_upload_bytes: int,
    ) -> None:
        self._document_repository = document_repository
        self._document_library = document_library
        self._pdf_loader = pdf_loader
        self._text_chunker = text_chunker
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._validation_service = validation_service
        self._maximum_upload_bytes = maximum_upload_bytes

    def execute(self, request: IngestDocumentRequest) -> IngestDocumentResult:
        """Store, extract, chunk, embed, and index a local PDF."""
        filename = self._validation_service.validate_text(
            request.original_filename,
            "original_filename",
        )
        self._validate_pdf(filename, request.content)
        document_id = DocumentId(str(uuid4()))
        document_path = self._document_library.store(
            document_id, filename, request.content
        )

        try:
            pages = self._pdf_loader.load(document_id, document_path)
            chunks = self._text_chunker.chunk(pages)
            if not chunks:
                raise DocumentProcessingError(
                    "The PDF does not contain extractable text."
                )
            embeddings = self._embedding_provider.embed_many(
                tuple(chunk.content for chunk in chunks),
            )
            self._vector_store.add(chunks, embeddings)
            document = Document(
                identifier=document_id,
                title=Path(filename).stem,
                source=filename,
                page_count=len(pages),
                created_at=datetime.now(UTC),
            )
            self._document_repository.save(document)
            return IngestDocumentResult(document=document)
        except Exception:
            self._vector_store.delete_document(document_id)
            self._document_library.delete(document_id)
            raise

    def _validate_pdf(self, filename: str, content: bytes) -> None:
        if not filename.lower().endswith(".pdf"):
            raise DocumentProcessingError("Only PDF files are supported.")
        if not content:
            raise DocumentProcessingError("The uploaded PDF is empty.")
        if len(content) > self._maximum_upload_bytes:
            raise DocumentProcessingError(
                "The uploaded PDF exceeds the configured size limit."
            )
        if not content.startswith(b"%PDF"):
            raise DocumentProcessingError("The uploaded file is not a valid PDF.")
