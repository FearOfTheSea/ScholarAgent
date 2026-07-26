"""Retriever implementation backed by local embeddings and FAISS."""

from scholar_agent.application.dtos.retrieval import RetrievedChunk
from scholar_agent.application.output_ports.embedding_provider import IEmbeddingProvider
from scholar_agent.application.output_ports.retriever import IRetriever
from scholar_agent.application.output_ports.vector_store import IVectorStore
from scholar_agent.domain.value_objects.document_id import DocumentId


class LangChainRetriever(IRetriever):
    """Performs semantic retrieval while remaining independent of use cases."""

    def __init__(
        self,
        embedding_provider: IEmbeddingProvider,
        vector_store: IVectorStore,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        document_ids: tuple[DocumentId, ...] = (),
    ) -> tuple[RetrievedChunk, ...]:
        """Embed a query and retrieve its closest local chunks."""
        return self._vector_store.search(
            embedding=self._embedding_provider.embed(query),
            limit=limit,
            document_ids=document_ids,
        )
