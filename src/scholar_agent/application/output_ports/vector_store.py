"""Vector-store port."""

from abc import ABC, abstractmethod

from scholar_agent.application.dtos.retrieval import DocumentChunk, RetrievedChunk
from scholar_agent.domain.value_objects.document_id import DocumentId


class IVectorStore(ABC):
    """Stores and searches document embeddings."""

    @abstractmethod
    def add(
        self,
        chunks: tuple[DocumentChunk, ...],
        embeddings: tuple[tuple[float, ...], ...],
    ) -> None:
        """Store embeddings and their document-chunk metadata."""

    @abstractmethod
    def search(
        self,
        embedding: tuple[float, ...],
        limit: int = 5,
        document_ids: tuple[DocumentId, ...] = (),
    ) -> tuple[RetrievedChunk, ...]:
        """Return the nearest stored vectors."""

    @abstractmethod
    def list_document_chunks(
        self, document_id: DocumentId
    ) -> tuple[DocumentChunk, ...]:
        """Return a document's chunks in source order."""

    @abstractmethod
    def get_chunk(
        self,
        document_id: DocumentId,
        chunk_id: str,
    ) -> DocumentChunk | None:
        """Return one stored chunk when it exists."""

    @abstractmethod
    def delete_document(self, document_id: DocumentId) -> None:
        """Remove a document's vectors and metadata."""
