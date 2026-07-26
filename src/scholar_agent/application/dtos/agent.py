"""DTOs for goal-oriented study-agent workflows."""

from dataclasses import dataclass

from scholar_agent.application.dtos.retrieval import RetrievedChunk
from scholar_agent.application.dtos.study_results import QuizQuestion
from scholar_agent.domain.value_objects.document_id import DocumentId


@dataclass(frozen=True, slots=True)
class PrepareStudySessionRequest:
    """A broad study goal for the local study agent."""

    goal: str
    document_ids: tuple[DocumentId, ...]
    question_count: int = 5
    session_id: str | None = None


@dataclass(frozen=True, slots=True)
class StudyAgentPlanStep:
    """One planned tool action shown to the learner."""

    tool_name: str
    description: str


@dataclass(frozen=True, slots=True)
class PrepareStudySessionResult:
    """The structured result of an agent study session."""

    plan: tuple[StudyAgentPlanStep, ...]
    summary: str
    quiz: tuple[QuizQuestion, ...]
    recommendations: tuple[str, ...]
    completed_tools: tuple[str, ...]
    citations: tuple[RetrievedChunk, ...]
    errors: tuple[str, ...] = ()
