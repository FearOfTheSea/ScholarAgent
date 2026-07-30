"""Application DTOs for the adaptive single-document tutor."""

from dataclasses import dataclass
from enum import StrEnum

from scholar_agent.domain.entities.study_session import (
    DocumentBrief,
    LearnerAttempt,
    LearnerLevel,
    ObjectiveProgress,
    SourceReference,
    StudyMode,
    StudySession,
    TutorTurnKind,
)
from scholar_agent.domain.value_objects.document_id import DocumentId


class TutorCapability(StrEnum):
    """Explicit single-document capabilities used by the adaptive tutor."""

    BUILD_DOCUMENT_MAP = "build_document_map"
    EXPLAIN_CONCEPT = "explain_concept"
    ASSESS_RESPONSE = "assess_response"


@dataclass(frozen=True, slots=True)
class StartStudySessionRequest:
    """Parameters for a new local tutoring session."""

    document_id: DocumentId
    goal: str = "Understand the document and retain its key ideas."
    learner_level: LearnerLevel = LearnerLevel.INTERMEDIATE
    mode: StudyMode = StudyMode.GUIDED
    target_minutes: int = 30


@dataclass(frozen=True, slots=True)
class TutorActivity:
    """The learner-facing activity returned for a turn."""

    kind: TutorTurnKind
    message: str
    objective_id: str | None
    citations: tuple[SourceReference, ...]


@dataclass(frozen=True, slots=True)
class StudySessionResult:
    """Complete resumable session state."""

    session: StudySession
    progress: tuple[ObjectiveProgress, ...]
    current_objective_id: str | None
    activity: TutorActivity | None = None


@dataclass(frozen=True, slots=True)
class ContinueStudySessionRequest:
    """One learner message in an existing session."""

    session_id: str
    message: str


@dataclass(frozen=True, slots=True)
class TutorTurnResult:
    """One completed tutor turn and updated state."""

    intent: str
    activity: TutorActivity
    assessment: LearnerAttempt | None
    progress: tuple[ObjectiveProgress, ...]
    current_objective_id: str | None


@dataclass(frozen=True, slots=True)
class DeleteStudySessionResult:
    """Result of deleting one local session."""

    session_id: str
    deleted: bool


@dataclass(frozen=True, slots=True)
class BuildDocumentBriefResult:
    """A newly generated or cached brief."""

    brief: DocumentBrief
    cached: bool
