"""Structured explanation capability adapter."""

from collections.abc import Mapping

from scholar_agent.application.dtos.mission import ExplainConceptRequest
from scholar_agent.application.use_cases.explain_concept import ExplainConceptUseCase
from scholar_agent.domain.value_objects.document_id import DocumentId
from scholar_agent.domain.value_objects.source_reference import SourceReference


class ExplainConceptTool:
    """Adapt the explanation use case to the structured tool port."""

    def __init__(self, use_case: ExplainConceptUseCase) -> None:
        self._use_case = use_case

    def execute(self, arguments: Mapping[str, object]) -> Mapping[str, object]:
        request = ExplainConceptRequest(
            document_id=_document_id(arguments),
            objective_id=_text(arguments, "objective_id"),
            source_chunk_ids=_string_list(arguments, "source_chunk_ids"),
            learner_question=_optional_text(arguments, "learner_question"),
            style=_optional_text(arguments, "style") or "concise",
        )
        result = self._use_case.execute(request)
        return {
            "objective_id": result.objective_id,
            "explanation": result.explanation,
            "check_question": result.check_question,
            "citations": [_reference_payload(item) for item in result.citations],
        }


def _document_id(arguments: Mapping[str, object]) -> DocumentId:
    return DocumentId(_text(arguments, "document_id"))


def _text(arguments: Mapping[str, object], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' must be non-blank text.")
    return value.strip()


def _optional_text(arguments: Mapping[str, object], key: str) -> str | None:
    value = arguments.get(key)
    if value is None:
        return None
    return _text(arguments, key)


def _string_list(arguments: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = arguments.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"'{key}' must be a non-empty list of strings.")
    return tuple(item.strip() for item in value)


def _reference_payload(reference: SourceReference) -> dict[str, object]:
    return {
        "document_id": reference.document_id.value,
        "chunk_id": reference.chunk_id,
        "page_number": reference.page_number,
        "excerpt": reference.excerpt,
    }
