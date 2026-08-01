"""Convert structured tool payloads into typed application results."""

from collections.abc import Mapping

from scholar_agent.application.dtos.agent import (
    StudyAgentAnswerResult,
    StudyAgentFlashcardsResult,
    StudyAgentQuizResult,
    StudyAgentSummaryResult,
    StudyAgentTaskResult,
    StudyTask,
)
from scholar_agent.application.dtos.retrieval import RetrievedChunk
from scholar_agent.application.dtos.study_results import Flashcard, QuizQuestion
from scholar_agent.domain.value_objects.document_id import DocumentId
from scholar_agent.domain.value_objects.source_reference import SourceReference


def task_result(
    task: StudyTask,
    payload: Mapping[str, object],
) -> StudyAgentTaskResult:
    """Convert one validated tool payload into its typed result variant."""
    if task is StudyTask.ANSWER_QUESTION:
        return StudyAgentAnswerResult(
            answer=_required_text(payload, "answer"),
            citations=_citations(payload.get("citations")),
        )
    if task is StudyTask.SUMMARIZE_DOCUMENT:
        return StudyAgentSummaryResult(
            summary=_required_text(payload, "summary"),
            citations=_source_references(payload.get("citations")),
        )
    if task is StudyTask.GENERATE_QUIZ:
        return StudyAgentQuizResult(
            questions=_quiz_questions(payload.get("questions")),
            requested_count=_required_int(payload, "requested_count"),
            effective_count=_required_int(payload, "effective_count"),
            maximum_count=_required_int(payload, "maximum_count"),
        )
    return StudyAgentFlashcardsResult(
        cards=_flashcards(payload.get("cards")),
        requested_count=_required_int(payload, "requested_count"),
        effective_count=_required_int(payload, "effective_count"),
        maximum_count=_required_int(payload, "maximum_count"),
    )


def _required_text(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Tool result '{key}' must be non-blank text.")
    return value.strip()


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Tool result '{key}' must be a positive integer.")
    return value


def _quiz_questions(value: object) -> tuple[QuizQuestion, ...]:
    if not isinstance(value, list):
        raise ValueError("Quiz tool result must contain a question list.")
    questions: list[QuizQuestion] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("Every quiz question must be an object.")
        prompt = item.get("prompt")
        answer = item.get("answer")
        if not isinstance(prompt, str) or not isinstance(answer, str):
            raise ValueError("Quiz question fields must be strings.")
        questions.append(
            QuizQuestion(
                prompt=prompt,
                answer=answer,
                citations=_source_references(item.get("citations")),
            )
        )
    return tuple(questions)


def _flashcards(value: object) -> tuple[Flashcard, ...]:
    if not isinstance(value, list):
        raise ValueError("Flashcard tool result must contain a card list.")
    cards: list[Flashcard] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("Every flashcard must be an object.")
        front = item.get("front")
        back = item.get("back")
        if not isinstance(front, str) or not isinstance(back, str):
            raise ValueError("Flashcard fields must be strings.")
        cards.append(
            Flashcard(
                front=front,
                back=back,
                citations=_source_references(item.get("citations")),
            )
        )
    return tuple(cards)


def _citations(value: object) -> tuple[RetrievedChunk, ...]:
    if not isinstance(value, list):
        raise ValueError("Answer tool result must contain a citation list.")
    return tuple(_citation(item) for item in value)


def _citation(value: object) -> RetrievedChunk:
    if not isinstance(value, Mapping):
        raise ValueError("Every citation must be an object.")
    document_id = value.get("document_id")
    content = value.get("content")
    page_number = value.get("page_number")
    section = value.get("section")
    chunk_id = value.get("chunk_id")
    similarity_score = value.get("similarity_score")
    if (
        not isinstance(document_id, str)
        or not isinstance(content, str)
        or not isinstance(chunk_id, str)
        or (page_number is not None and not isinstance(page_number, int))
        or (section is not None and not isinstance(section, str))
        or isinstance(similarity_score, bool)
        or not isinstance(similarity_score, (int, float))
    ):
        raise ValueError("Citation fields have invalid types.")
    return RetrievedChunk(
        document_id=DocumentId(document_id),
        content=content,
        page_number=page_number,
        section=section,
        chunk_id=chunk_id,
        similarity_score=float(similarity_score),
    )


def _source_references(value: object) -> tuple[SourceReference, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("Material citations must be a list.")
    references: list[SourceReference] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("Every material citation must be an object.")
        document_id = item.get("document_id")
        chunk_id = item.get("chunk_id")
        page_number = item.get("page_number")
        excerpt = item.get("excerpt")
        if (
            not isinstance(document_id, str)
            or not isinstance(chunk_id, str)
            or not isinstance(excerpt, str)
            or (page_number is not None and not isinstance(page_number, int))
        ):
            raise ValueError("Material citation fields have invalid types.")
        references.append(
            SourceReference(DocumentId(document_id), chunk_id, page_number, excerpt)
        )
    return tuple(references)
