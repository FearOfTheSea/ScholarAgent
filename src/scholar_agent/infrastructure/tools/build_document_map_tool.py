"""Structured document-map capability adapter."""

from collections.abc import Mapping

from scholar_agent.application.use_cases.build_document_brief import (
    BuildDocumentBriefUseCase,
)
from scholar_agent.domain.value_objects.document_id import DocumentId


class BuildDocumentMapTool:
    """Adapt the cited document-map use case to the tool port."""

    def __init__(self, use_case: BuildDocumentBriefUseCase) -> None:
        self._use_case = use_case

    def execute(self, arguments: Mapping[str, object]) -> Mapping[str, object]:
        document_id = _document_id(arguments)
        result = self._use_case.execute(document_id)
        return {
            "document_id": result.brief.document_id.value,
            "synopsis": result.brief.synopsis,
            "objectives": [
                {
                    "id": objective.identifier,
                    "title": objective.title,
                    "description": objective.description,
                    "prerequisites": list(objective.prerequisite_ids),
                    "citations": [
                        reference.chunk_id for reference in objective.citations
                    ],
                }
                for objective in result.brief.objectives
            ],
            "cached": result.cached,
        }


def _document_id(arguments: Mapping[str, object]) -> DocumentId:
    value = arguments.get("document_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("'document_id' must be non-blank text.")
    return DocumentId(value.strip())
