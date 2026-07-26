"""Citation-lookup structured tool."""

from collections.abc import Mapping

from scholar_agent.application.output_ports.vector_store import IVectorStore
from scholar_agent.domain.value_objects.document_id import DocumentId


class CitationLookupTool:
    """Returns source text for a cited document chunk."""

    def __init__(self, vector_store: IVectorStore) -> None:
        self._vector_store = vector_store

    def execute(self, arguments: Mapping[str, object]) -> Mapping[str, object]:
        """Look up a cited chunk by its document and chunk identifier."""
        document_id = _document_id(arguments)
        chunk_id = _required_text(arguments, "chunk_id")
        chunk = self._vector_store.get_chunk(document_id, chunk_id)
        if chunk is None:
            return {"found": False}
        return {
            "found": True,
            "document_id": chunk.document_id.value,
            "chunk_id": chunk.chunk_id,
            "page_number": chunk.page_number,
            "content": chunk.content,
        }


def _document_id(arguments: Mapping[str, object]) -> DocumentId:
    value = arguments.get("document_id")
    if not isinstance(value, str):
        raise ValueError("'document_id' must be a string.")
    return DocumentId(value)


def _required_text(arguments: Mapping[str, object], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' must be a non-blank string.")
    return value.strip()
