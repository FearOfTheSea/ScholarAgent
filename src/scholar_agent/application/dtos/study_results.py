"""Results produced by direct study-assistance use cases."""

from dataclasses import dataclass

from scholar_agent.application.dtos.retrieval import RetrievedChunk


@dataclass(frozen=True, slots=True)
class AnswerQuestionResult:
    """An answer with the source chunks that support it."""

    answer: str
    citations: tuple[RetrievedChunk, ...]


@dataclass(frozen=True, slots=True)
class SummarizeDocumentResult:
    """A concise document summary."""

    summary: str


@dataclass(frozen=True, slots=True)
class QuizQuestion:
    """A quiz question and its answer."""

    prompt: str
    answer: str


@dataclass(frozen=True, slots=True)
class GenerateQuizResult:
    """A generated quiz and the count policy applied to it."""

    questions: tuple[QuizQuestion, ...]
    requested_count: int
    effective_count: int
    maximum_count: int
    notice: str | None = None


@dataclass(frozen=True, slots=True)
class Flashcard:
    """A study flashcard."""

    front: str
    back: str


@dataclass(frozen=True, slots=True)
class GenerateFlashcardsResult:
    """Generated flashcards and the count policy applied to them."""

    cards: tuple[Flashcard, ...]
    requested_count: int
    effective_count: int
    maximum_count: int
    notice: str | None = None
