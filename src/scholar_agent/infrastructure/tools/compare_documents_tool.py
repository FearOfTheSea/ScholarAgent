"""Compare-documents structured tool."""

from collections.abc import Mapping

from scholar_agent.application.dtos.study_requests import CompareDocumentsRequest
from scholar_agent.application.input_ports.study_assistant import CompareDocuments
from scholar_agent.domain.value_objects.document_id import DocumentId


class CompareDocumentsTool:
    """Delegates a structured comparison request to its use case."""

    def __init__(self, use_case: CompareDocuments) -> None:
        self._use_case = use_case

    def execute(self, arguments: Mapping[str, object]) -> Mapping[str, object]:
        """Return a comparison and its source citations."""
        result = self._use_case.execute(
            CompareDocumentsRequest(
                first_document_id=DocumentId(
                    _required_text(arguments, "first_document_id")
                ),
                second_document_id=DocumentId(
                    _required_text(arguments, "second_document_id")
                ),
            ),
        )
        return {
            "comparison": result.comparison,
            "citations": [chunk.chunk_id for chunk in result.citations],
        }


def _required_text(arguments: Mapping[str, object], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' must be a non-blank string.")
    return value.strip()
