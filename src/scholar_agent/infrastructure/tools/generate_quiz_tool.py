"""Generate-quiz structured tool."""

from collections.abc import Mapping

from scholar_agent.application.dtos.study_requests import GenerateQuizRequest
from scholar_agent.application.input_ports.study_assistant import GenerateQuiz
from scholar_agent.domain.value_objects.document_id import DocumentId


class GenerateQuizTool:
    """Delegates a structured quiz request to its use case."""

    def __init__(self, use_case: GenerateQuiz) -> None:
        self._use_case = use_case

    def execute(self, arguments: Mapping[str, object]) -> Mapping[str, object]:
        """Return a structured quiz."""
        document_id = _document_id(arguments)
        question_count = _count(arguments, "question_count", default=5)
        result = self._use_case.execute(
            GenerateQuizRequest(document_id, question_count)
        )
        return {
            "questions": [
                {"prompt": question.prompt, "answer": question.answer}
                for question in result.questions
            ],
            "requested_count": result.requested_count,
            "effective_count": result.effective_count,
            "maximum_count": result.maximum_count,
            "notice": result.notice,
        }


def _document_id(arguments: Mapping[str, object]) -> DocumentId:
    value = arguments.get("document_id")
    if not isinstance(value, str):
        raise ValueError("'document_id' must be a string.")
    return DocumentId(value)


def _count(arguments: Mapping[str, object], key: str, default: int) -> int:
    value = arguments.get(key, default)
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"'{key}' must be a positive integer.")
    return value
