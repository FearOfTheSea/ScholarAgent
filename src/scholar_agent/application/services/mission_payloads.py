"""Validate mission capability payloads and create domain material objects."""

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime

from scholar_agent.application.dtos.tutor import TutorActivity
from scholar_agent.domain.entities.study_material import (
    Flashcard,
    FlashcardArtifact,
    QuizArtifact,
    QuizQuestion,
    SummaryArtifact,
)
from scholar_agent.domain.entities.study_session import (
    PendingLearnerInteraction,
    SourceReference,
    TutorTurnKind,
)
from scholar_agent.domain.value_objects.document_id import DocumentId


def text_value(payload: Mapping[str, object], key: str) -> str:
    """Read one required non-blank text field."""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Capability field '{key}' must be non-blank text.")
    return value.strip()


def integer_value(payload: Mapping[str, object], key: str) -> int:
    """Read one bounded assessment score."""
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 3:
        raise ValueError(f"Capability field '{key}' must be an integer from 0 to 3.")
    return value


def string_values(value: object) -> tuple[str, ...]:
    """Read a non-null array of non-blank strings."""
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError("Capability missing_concepts must be a string array.")
    return tuple(item.strip() for item in value)


def mappings(value: list[object]) -> tuple[Mapping[str, object], ...]:
    """Validate an array of object payloads."""
    if not all(isinstance(item, Mapping) for item in value):
        raise ValueError("Capability items must be objects.")
    return tuple(item for item in value if isinstance(item, Mapping))


def references(value: object, document_id: DocumentId) -> tuple[SourceReference, ...]:
    """Validate cited payloads against the mission's selected document."""
    if not isinstance(value, list) or not value:
        raise ValueError("Every mission material must contain citations.")
    result: list[SourceReference] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("Mission citations must be objects.")
        raw_document_id = item.get("document_id")
        chunk_id = item.get("chunk_id")
        excerpt = item.get("excerpt")
        page_number = item.get("page_number")
        if (
            raw_document_id != document_id.value
            or not isinstance(chunk_id, str)
            or not chunk_id.strip()
            or not isinstance(excerpt, str)
            or not excerpt.strip()
            or (page_number is not None and not isinstance(page_number, int))
        ):
            raise ValueError("Mission citation does not match the selected document.")
        result.append(
            SourceReference(document_id, chunk_id.strip(), page_number, excerpt.strip())
        )
    return tuple(result)


def search_chunk_ids(value: Mapping[str, object], document_id: DocumentId) -> list[str]:
    """Validate semantic-search results before using their chunk IDs."""
    chunks = value.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError("Semantic search returned an invalid chunk list.")
    identifiers: list[str] = []
    for item in chunks:
        if not isinstance(item, Mapping):
            raise ValueError("Semantic search returned an invalid chunk.")
        if item.get("document_id") != document_id.value:
            raise ValueError("Semantic search returned a different document.")
        chunk_id = item.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("Semantic search returned an invalid chunk ID.")
        identifiers.append(chunk_id.strip())
    return identifiers


def summary_artifact(
    payload: Mapping[str, object], document_id: DocumentId
) -> SummaryArtifact:
    """Create a cited summary artifact."""
    return SummaryArtifact(
        text_value(payload, "summary"),
        references(payload.get("citations"), document_id),
        datetime.now(UTC),
    )


def quiz_artifact(
    payload: Mapping[str, object], document_id: DocumentId
) -> QuizArtifact:
    """Create a quiz artifact with citations retained on every question."""
    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError("Quiz capability returned no cited questions.")
    questions = tuple(
        QuizQuestion(
            text_value(item, "prompt"),
            text_value(item, "answer"),
            references(item.get("citations"), document_id),
        )
        for item in mappings(raw_questions)
    )
    return QuizArtifact(
        questions,
        unique_references(ref for item in questions for ref in item.citations),
    )


def flashcard_artifact(
    payload: Mapping[str, object], document_id: DocumentId
) -> FlashcardArtifact:
    """Create a flashcard artifact with citations retained on every card."""
    raw_cards = payload.get("cards")
    if not isinstance(raw_cards, list) or not raw_cards:
        raise ValueError("Flashcard capability returned no cited cards.")
    cards = tuple(
        Flashcard(
            text_value(item, "front"),
            text_value(item, "back"),
            references(item.get("citations"), document_id),
        )
        for item in mappings(raw_cards)
    )
    return FlashcardArtifact(
        cards, unique_references(ref for item in cards for ref in item.citations)
    )


def first_pending_question(
    questions: list[object],
    objective_id: str,
    document_id: DocumentId,
) -> PendingLearnerInteraction | None:
    """Turn the first cited quiz item into hidden-answer pending state."""
    if not questions or not isinstance(questions[0], Mapping):
        return None
    item = questions[0]
    return PendingLearnerInteraction(
        objective_id=objective_id,
        question=text_value(item, "prompt"),
        reference_answer=text_value(item, "answer"),
        citations=references(item.get("citations"), document_id),
    )


def question_activity(pending: PendingLearnerInteraction) -> TutorActivity:
    """Render a learner question without its hidden reference answer."""
    return TutorActivity(
        TutorTurnKind.QUESTION,
        pending.question,
        pending.objective_id,
        pending.citations,
    )


def unique_references(
    source: Iterable[SourceReference],
) -> tuple[SourceReference, ...]:
    """Deduplicate citations in first-seen order."""
    result: list[SourceReference] = []
    seen: set[str] = set()
    for reference in source:
        if reference.chunk_id not in seen:
            result.append(reference)
            seen.add(reference.chunk_id)
    return tuple(result)
