"""Requests accepted by study-assistance use cases."""

from dataclasses import dataclass

from scholar_agent.domain.value_objects.document_id import DocumentId


@dataclass(frozen=True, slots=True)
class AnswerQuestionRequest:
    """Request to answer a question using selected documents."""

    question: str
    document_ids: tuple[DocumentId, ...] = ()


@dataclass(frozen=True, slots=True)
class SummarizeDocumentRequest:
    """Request to summarize one document."""

    document_id: DocumentId


@dataclass(frozen=True, slots=True)
class CompareDocumentsRequest:
    """Request to compare two documents."""

    first_document_id: DocumentId
    second_document_id: DocumentId


@dataclass(frozen=True, slots=True)
class GenerateQuizRequest:
    """Request to generate a quiz for one document."""

    document_id: DocumentId
    question_count: int = 5


@dataclass(frozen=True, slots=True)
class GenerateFlashcardsRequest:
    """Request to generate flashcards for one document."""

    document_id: DocumentId
    card_count: int = 10
