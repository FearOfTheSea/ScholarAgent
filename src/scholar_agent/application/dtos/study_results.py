"""Results produced by study-assistance use cases."""

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
class CompareDocumentsResult:
    """A comparison of two documents."""

    comparison: str
    citations: tuple[RetrievedChunk, ...]


@dataclass(frozen=True, slots=True)
class QuizQuestion:
    """A quiz question and its answer."""

    prompt: str
    answer: str


@dataclass(frozen=True, slots=True)
class GenerateQuizResult:
    """A generated quiz."""

    questions: tuple[QuizQuestion, ...]


@dataclass(frozen=True, slots=True)
class Flashcard:
    """A study flashcard."""

    front: str
    back: str


@dataclass(frozen=True, slots=True)
class GenerateFlashcardsResult:
    """A generated flashcard set."""

    cards: tuple[Flashcard, ...]
