"""Summarize-document structured tool."""

from collections.abc import Mapping

from scholar_agent.application.dtos.study_requests import SummarizeDocumentRequest
from scholar_agent.application.input_ports.study_assistant import SummarizeDocument
from scholar_agent.domain.value_objects.document_id import DocumentId


class SummarizeDocumentTool:
    """Delegates a structured summary request to its use case."""

    def __init__(self, use_case: SummarizeDocument) -> None:
        self._use_case = use_case

    def execute(self, arguments: Mapping[str, object]) -> Mapping[str, object]:
        """Return a structured document summary."""
        document_id = _document_id(arguments)
        result = self._use_case.execute(SummarizeDocumentRequest(document_id))
        return {
            "summary": result.summary,
            "citations": [
                {
                    "document_id": reference.document_id.value,
                    "chunk_id": reference.chunk_id,
                    "page_number": reference.page_number,
                    "excerpt": reference.excerpt,
                }
                for reference in result.citations
            ],
        }


def _document_id(arguments: Mapping[str, object]) -> DocumentId:
    value = arguments.get("document_id")
    if not isinstance(value, str):
        raise ValueError("'document_id' must be a string.")
    return DocumentId(value)
