"""Generate-flashcards structured tool."""

from collections.abc import Mapping

from scholar_agent.application.dtos.study_requests import GenerateFlashcardsRequest
from scholar_agent.application.input_ports.study_assistant import GenerateFlashcards
from scholar_agent.domain.value_objects.document_id import DocumentId


class GenerateFlashcardsTool:
    """Delegates a structured flashcard request to its use case."""

    def __init__(self, use_case: GenerateFlashcards) -> None:
        self._use_case = use_case

    def execute(self, arguments: Mapping[str, object]) -> Mapping[str, object]:
        """Return structured study flashcards."""
        document_id = _document_id(arguments)
        card_count = _count(arguments, "card_count", default=10)
        result = self._use_case.execute(
            GenerateFlashcardsRequest(document_id, card_count)
        )
        return {
            "cards": [
                {"front": card.front, "back": card.back} for card in result.cards
            ],
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
