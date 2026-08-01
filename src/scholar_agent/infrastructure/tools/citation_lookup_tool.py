"""Citation-lookup structured tool."""

from collections.abc import Mapping

from scholar_agent.application.output_ports.vector_store import IVectorStore
from scholar_agent.domain.value_objects.document_id import DocumentId


class CitationLookupTool:
    """Returns source text for a cited document chunk."""

    def __init__(self, vector_store: IVectorStore) -> None:
        self._vector_store = vector_store

    def execute(self, arguments: Mapping[str, object]) -> Mapping[str, object]:
        """Look up one to twenty cited chunks by their identifiers."""
        document_id = _document_id(arguments)
        raw_chunk_ids = arguments.get("chunk_ids")
        if raw_chunk_ids is None:
            raw_chunk_ids = [_required_text(arguments, "chunk_id")]
        if (
            not isinstance(raw_chunk_ids, list)
            or not 1 <= len(raw_chunk_ids) <= 20
            or not all(isinstance(item, str) and item.strip() for item in raw_chunk_ids)
        ):
            raise ValueError("'chunk_ids' must contain between 1 and 20 chunk IDs.")
        chunks = []
        for raw_chunk_id in raw_chunk_ids:
            chunk = self._vector_store.get_chunk(document_id, raw_chunk_id.strip())
            if chunk is not None:
                chunks.append(
                    {
                        "found": True,
                        "document_id": chunk.document_id.value,
                        "chunk_id": chunk.chunk_id,
                        "page_number": chunk.page_number,
                        "content": chunk.content,
                    }
                )
            else:
                chunks.append({"found": False, "chunk_id": raw_chunk_id.strip()})
        if len(chunks) == 1 and "chunk_ids" not in arguments:
            return chunks[0]
        return {"chunks": chunks}


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
