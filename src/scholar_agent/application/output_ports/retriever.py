"""Retriever port."""

from abc import ABC, abstractmethod

from scholar_agent.application.dtos.retrieval import RetrievedChunk
from scholar_agent.domain.value_objects.document_id import DocumentId


class IRetriever(ABC):
    """Finds relevant chunks for a textual query."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        limit: int = 5,
        document_ids: tuple[DocumentId, ...] = (),
    ) -> tuple[RetrievedChunk, ...]:
        """Return up to ``limit`` chunks relevant to a query."""
