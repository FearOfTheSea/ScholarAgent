"""Domain concepts for one-document adaptive study sessions."""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from scholar_agent.domain.value_objects.document_id import DocumentId


class StudyMode(StrEnum):
    """Supported tutoring styles."""

    GUIDED = "guided"
    EXAM = "exam"
    CRAM = "cram"


class LearnerLevel(StrEnum):
    """Learner-selected explanation depth."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class MasteryLabel(StrEnum):
    """Human-readable objective progress."""

    UNSEEN = "unseen"
    DEVELOPING = "developing"
    PROFICIENT = "proficient"
    MASTERED = "mastered"


class TutorTurnKind(StrEnum):
    """Learner-facing response variants."""

    EXPLANATION = "explanation"
    QUESTION = "question"
    ASSESSMENT = "assessment"
    HINT = "hint"
    RECAP = "recap"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class SourceReference:
    """A source location inside the session's selected document."""

    document_id: DocumentId
    chunk_id: str
    page_number: int | None
    excerpt: str


@dataclass(frozen=True, slots=True)
class LearningObjective:
    """A teachable objective derived from the selected document."""

    identifier: str
    title: str
    description: str
    prerequisite_ids: tuple[str, ...]
    citations: tuple[SourceReference, ...]


@dataclass(frozen=True, slots=True)
class ConceptNode:
    """One cited concept in a document knowledge map."""

    identifier: str
    label: str
    explanation: str
    prerequisite_ids: tuple[str, ...]
    citations: tuple[SourceReference, ...]


@dataclass(frozen=True, slots=True)
class GlossaryTerm:
    """A cited term and definition."""

    term: str
    definition: str
    citations: tuple[SourceReference, ...]


@dataclass(frozen=True, slots=True)
class DocumentBrief:
    """Cached, cited learning map for one document."""

    document_id: DocumentId
    synopsis: str
    objectives: tuple[LearningObjective, ...]
    concepts: tuple[ConceptNode, ...]
    glossary: tuple[GlossaryTerm, ...]
    misconceptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LearnerAttempt:
    """A scored learner response tied to one objective."""

    objective_id: str
    response: str
    score: int
    feedback: str
    missing_concepts: tuple[str, ...]
    citations: tuple[SourceReference, ...]
    created_at: datetime

    def __post_init__(self) -> None:
        if self.score < 0 or self.score > 3:
            raise ValueError("Attempt score must be between 0 and 3.")


@dataclass(frozen=True, slots=True)
class ObjectiveProgress:
    """Deterministic mastery derived from recent attempts."""

    objective_id: str
    percentage: int
    label: MasteryLabel
    attempt_count: int


@dataclass(frozen=True, slots=True)
class TutorTurn:
    """One persisted exchange with the tutor."""

    kind: TutorTurnKind
    learner_message: str
    tutor_message: str
    objective_id: str | None
    citations: tuple[SourceReference, ...]
    assessment: LearnerAttempt | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class StudySession:
    """A resumable adaptive session permanently bound to one document."""

    identifier: str
    document_id: DocumentId
    goal: str
    learner_level: LearnerLevel
    mode: StudyMode
    target_minutes: int
    brief: DocumentBrief
    attempts: tuple[LearnerAttempt, ...] = ()
    turns: tuple[TutorTurn, ...] = ()
    created_at: datetime = datetime.min.replace(tzinfo=UTC)
    updated_at: datetime = datetime.min.replace(tzinfo=UTC)


def objective_progress(
    objective_id: str,
    attempts: tuple[LearnerAttempt, ...],
) -> ObjectiveProgress:
    """Calculate mastery from the latest three attempts for an objective."""
    relevant = tuple(
        attempt for attempt in attempts if attempt.objective_id == objective_id
    )
    if not relevant:
        return ObjectiveProgress(objective_id, 0, MasteryLabel.UNSEEN, 0)
    recent = relevant[-3:]
    percentage = round(
        sum(attempt.score for attempt in recent) / (3 * len(recent)) * 100
    )
    if percentage >= 80 and len(relevant) >= 2:
        label = MasteryLabel.MASTERED
    elif percentage >= 50:
        label = MasteryLabel.PROFICIENT
    else:
        label = MasteryLabel.DEVELOPING
    return ObjectiveProgress(objective_id, percentage, label, len(relevant))
