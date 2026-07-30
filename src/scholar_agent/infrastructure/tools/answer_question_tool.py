"""Answer-question structured tool."""

from collections.abc import Mapping

from scholar_agent.application.dtos.study_requests import AnswerQuestionRequest
from scholar_agent.application.input_ports.study_assistant import AnswerQuestion
from scholar_agent.domain.value_objects.document_id import DocumentId


class AnswerQuestionTool:
    """Delegate a grounded question to its application use case."""

    def __init__(self, use_case: AnswerQuestion) -> None:
        self._use_case = use_case

    def execute(self, arguments: Mapping[str, object]) -> Mapping[str, object]:
        """Return a structured answer and its citations."""
        document_id = _document_id(arguments)
        question = _required_text(arguments, "question")
        result = self._use_case.execute(
            AnswerQuestionRequest(question=question, document_id=document_id)
        )
        return {
            "answer": result.answer,
            "citations": [
                {
                    "document_id": citation.document_id.value,
                    "content": citation.content,
                    "page_number": citation.page_number,
                    "section": citation.section,
                    "chunk_id": citation.chunk_id,
                    "similarity_score": citation.similarity_score,
                }
                for citation in result.citations
            ],
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
