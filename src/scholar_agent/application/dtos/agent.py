"""DTOs for the unified study-agent workflow."""

from dataclasses import dataclass
from enum import StrEnum

from scholar_agent.application.dtos.retrieval import RetrievedChunk
from scholar_agent.application.dtos.study_results import Flashcard, QuizQuestion
from scholar_agent.domain.entities.study_session import SourceReference
from scholar_agent.domain.value_objects.document_id import DocumentId


class StudyTask(StrEnum):
    """A user-facing study capability available to the agent."""

    ANSWER_QUESTION = "answer_question"
    SEMANTIC_SEARCH = "semantic_search"
    SUMMARIZE_DOCUMENT = "summarize_document"
    GENERATE_QUIZ = "generate_quiz"
    GENERATE_FLASHCARDS = "generate_flashcards"
    CITATION_LOOKUP = "citation_lookup"
    BUILD_DOCUMENT_MAP = "build_document_map"
    EXPLAIN_CONCEPT = "explain_concept"
    ASSESS_LEARNER_RESPONSE = "assess_learner_response"


class StudyAgentStatus(StrEnum):
    """Completion state for a unified study request."""

    COMPLETED = "completed"
    NEEDS_CLARIFICATION = "needs_clarification"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class AskStudyAgentRequest:
    """One free-form request grounded in one selected document."""

    prompt: str
    document_id: DocumentId
    quiz_count_default: int = 5


@dataclass(frozen=True, slots=True)
class StudyAgentPlanStep:
    """One validated capability selected by the planner."""

    task: StudyTask
    description: str


@dataclass(frozen=True, slots=True)
class StudyAgentAnswerResult:
    """A grounded answer produced by the question-answering use case."""

    answer: str
    citations: tuple[RetrievedChunk, ...]
    task: StudyTask = StudyTask.ANSWER_QUESTION


@dataclass(frozen=True, slots=True)
class StudyAgentSummaryResult:
    """A document summary produced by the summarization use case."""

    summary: str
    citations: tuple[SourceReference, ...] = ()
    task: StudyTask = StudyTask.SUMMARIZE_DOCUMENT


@dataclass(frozen=True, slots=True)
class StudyAgentQuizResult:
    """A quiz and the count policy applied to its generation."""

    questions: tuple[QuizQuestion, ...]
    requested_count: int
    effective_count: int
    maximum_count: int
    task: StudyTask = StudyTask.GENERATE_QUIZ


@dataclass(frozen=True, slots=True)
class StudyAgentFlashcardsResult:
    """Flashcards and the count policy applied to their generation."""

    cards: tuple[Flashcard, ...]
    requested_count: int
    effective_count: int
    maximum_count: int
    task: StudyTask = StudyTask.GENERATE_FLASHCARDS


type StudyAgentTaskResult = (
    StudyAgentAnswerResult
    | StudyAgentSummaryResult
    | StudyAgentQuizResult
    | StudyAgentFlashcardsResult
)


@dataclass(frozen=True, slots=True)
class StudyAgentTaskError:
    """A runtime failure isolated to one planned task."""

    task: StudyTask
    message: str


@dataclass(frozen=True, slots=True)
class AskStudyAgentResult:
    """Typed result returned by the unified study-agent use case."""

    status: StudyAgentStatus
    plan: tuple[StudyAgentPlanStep, ...]
    results: tuple[StudyAgentTaskResult, ...]
    notices: tuple[str, ...] = ()
    errors: tuple[StudyAgentTaskError, ...] = ()
    message: str | None = None
