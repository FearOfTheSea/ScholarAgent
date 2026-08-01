"""Semantic-search structured tool."""

from collections.abc import Mapping

from scholar_agent.application.output_ports.retriever import IRetriever
from scholar_agent.domain.value_objects.document_id import DocumentId


class SemanticSearchTool:
    """Returns retrieved chunk data without generating an answer."""

    def __init__(self, retriever: IRetriever) -> None:
        self._retriever = retriever

    def execute(self, arguments: Mapping[str, object]) -> Mapping[str, object]:
        """Run semantic search with optional document filters."""
        query = _required_text(arguments, "query")
        limit = _optional_positive_int(arguments, "limit", default=4)
        document_ids = _document_ids(arguments)
        chunks = self._retriever.retrieve(query, limit, document_ids)
        return {
            "chunks": [
                {
                    "document_id": chunk.document_id.value,
                    "chunk_id": chunk.chunk_id,
                    "page_number": chunk.page_number,
                    "content": chunk.content,
                    "similarity_score": chunk.similarity_score,
                }
                for chunk in chunks
            ],
        }


def _required_text(arguments: Mapping[str, object], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' must be a non-blank string.")
    return value.strip()


def _optional_positive_int(
    arguments: Mapping[str, object], key: str, default: int
) -> int:
    value = arguments.get(key, default)
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"'{key}' must be a positive integer.")
    return value


def _document_ids(arguments: Mapping[str, object]) -> tuple[DocumentId, ...]:
    raw_document_id = arguments.get("document_id")
    if isinstance(raw_document_id, str) and raw_document_id.strip():
        return (DocumentId(raw_document_id.strip()),)
    raw_document_ids = arguments.get("document_ids")
    if raw_document_ids is None:
        raise ValueError("'document_id' is required for semantic search.")
    if not isinstance(raw_document_ids, list) or not all(
        isinstance(document_id, str) and document_id.strip()
        for document_id in raw_document_ids
    ):
        raise ValueError("'document_ids' must be a list of strings.")
    if not raw_document_ids:
        raise ValueError("'document_ids' must contain at least one document.")
    return tuple(DocumentId(document_id.strip()) for document_id in raw_document_ids)
