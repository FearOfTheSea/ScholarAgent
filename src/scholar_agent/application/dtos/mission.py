"""Application DTOs for the bounded single-document mission capabilities."""

from dataclasses import dataclass

from scholar_agent.domain.entities.study_session import (
    MissionTraceEvent,
    SourceReference,
)
from scholar_agent.domain.value_objects.document_id import DocumentId


@dataclass(frozen=True, slots=True)
class ExplainConceptRequest:
    """Request a cited explanation for one objective."""

    document_id: DocumentId
    objective_id: str
    source_chunk_ids: tuple[str, ...]
    learner_question: str | None = None
    style: str = "concise"


@dataclass(frozen=True, slots=True)
class ExplainConceptResult:
    """A cited explanation and a comprehension check."""

    objective_id: str
    explanation: str
    check_question: str
    citations: tuple[SourceReference, ...]


@dataclass(frozen=True, slots=True)
class AssessLearnerResponseRequest:
    """Request a cited assessment of one pending learner response."""

    document_id: DocumentId
    objective_id: str
    pending_question: str
    learner_response: str
    source_chunk_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AssessLearnerResponseResult:
    """Structured learner assessment used by mission policies."""

    objective_id: str
    score: int
    feedback: str
    missing_concepts: tuple[str, ...]
    next_question: str
    citations: tuple[SourceReference, ...]


@dataclass(frozen=True, slots=True)
class AdvanceStudyMissionRequest:
    """An optional learner action for a persistent mission."""

    session_id: str
    message: str | None = None


@dataclass(frozen=True, slots=True)
class MissionAdvanceResult:
    """Updated mission state and the latest bounded learner activity."""

    session_id: str
    activity_message: str | None
    trace_events: tuple[MissionTraceEvent, ...]
