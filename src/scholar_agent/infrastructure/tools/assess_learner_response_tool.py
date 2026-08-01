"""Structured learner-assessment capability adapter."""

from collections.abc import Mapping

from scholar_agent.application.dtos.mission import AssessLearnerResponseRequest
from scholar_agent.application.use_cases.assess_learner_response import (
    AssessLearnerResponseUseCase,
)
from scholar_agent.domain.value_objects.document_id import DocumentId
from scholar_agent.domain.value_objects.source_reference import SourceReference


class AssessLearnerResponseTool:
    """Adapt the assessment use case to the structured tool port."""

    def __init__(self, use_case: AssessLearnerResponseUseCase) -> None:
        self._use_case = use_case

    def execute(self, arguments: Mapping[str, object]) -> Mapping[str, object]:
        request = AssessLearnerResponseRequest(
            document_id=DocumentId(_text(arguments, "document_id")),
            objective_id=_text(arguments, "objective_id"),
            pending_question=_text(arguments, "pending_question"),
            learner_response=_text(arguments, "learner_response"),
            source_chunk_ids=_string_list(arguments, "source_chunk_ids"),
        )
        result = self._use_case.execute(request)
        return {
            "objective_id": result.objective_id,
            "score": result.score,
            "feedback": result.feedback,
            "missing_concepts": list(result.missing_concepts),
            "next_question": result.next_question,
            "citations": [_reference_payload(item) for item in result.citations],
        }


def _text(arguments: Mapping[str, object], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' must be non-blank text.")
    return value.strip()


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
