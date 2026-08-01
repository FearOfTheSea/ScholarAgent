"""Cited study materials produced for one selected document."""

from dataclasses import dataclass
from datetime import UTC, datetime

from scholar_agent.domain.value_objects.source_reference import SourceReference


@dataclass(frozen=True, slots=True)
class SummaryArtifact:
    """A concise summary with the source references used to produce it."""

    text: str
    citations: tuple[SourceReference, ...]
    created_at: datetime = datetime.min.replace(tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class QuizQuestion:
    """A quiz question, answer, and supporting source references."""

    prompt: str
    answer: str
    citations: tuple[SourceReference, ...] = ()


@dataclass(frozen=True, slots=True)
class Flashcard:
    """A flashcard and supporting source references."""

    front: str
    back: str
    citations: tuple[SourceReference, ...] = ()


@dataclass(frozen=True, slots=True)
class QuizArtifact:
    """A generated quiz artifact."""

    questions: tuple[QuizQuestion, ...]
    citations: tuple[SourceReference, ...] = ()
    created_at: datetime = datetime.min.replace(tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class FlashcardArtifact:
    """A generated flashcard artifact."""

    cards: tuple[Flashcard, ...]
    citations: tuple[SourceReference, ...] = ()
    created_at: datetime = datetime.min.replace(tzinfo=UTC)


StudyArtifact = SummaryArtifact | QuizArtifact | FlashcardArtifact
