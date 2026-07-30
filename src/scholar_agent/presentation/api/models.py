"""HTTP request and response models."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field

StudyTaskName = Literal[
    "answer_question",
    "summarize_document",
    "generate_quiz",
    "generate_flashcards",
]
StudyAgentStatusName = Literal[
    "completed",
    "needs_clarification",
    "partial",
    "failed",
]


class HealthResponse(BaseModel):
    """Response returned by the health endpoint."""

    status: Literal["ok"]
    service: str
    version: str


class ReadinessResponse(BaseModel):
    """Availability of the configured local Ollama model."""

    status: Literal["ready", "unavailable"]
    ollama_available: bool
    model_available: bool


class DocumentResponse(BaseModel):
    """Document metadata exposed by the local library API."""

    id: str
    title: str
    source: str
    page_count: int
    created_at: datetime


class CitationResponse(BaseModel):
    """A source chunk that supports a generated result."""

    document_id: str
    chunk_id: str
    page_number: int | None
    section: str | None
    similarity_score: float


class AnswerQuestionRequestModel(BaseModel):
    """Request body for a grounded, single-document question."""

    question: str
    document_id: str


class AnswerQuestionResponse(BaseModel):
    """Grounded answer and citations."""

    answer: str
    citations: list[CitationResponse]


class GenerateQuizRequestModel(BaseModel):
    """Requested number of generated quiz questions."""

    question_count: int = 5


class QuizQuestionResponse(BaseModel):
    """One generated quiz question."""

    prompt: str
    answer: str


class GenerateQuizResponse(BaseModel):
    """Generated quiz response with applied count policy."""

    questions: list[QuizQuestionResponse]
    requested_count: int
    effective_count: int
    generated_count: int
    maximum_count: int
    notice: str | None = None


class GenerateFlashcardsRequestModel(BaseModel):
    """Requested number of generated flashcards."""

    card_count: int = 10


class FlashcardResponse(BaseModel):
    """One generated flashcard."""

    front: str
    back: str


class GenerateFlashcardsResponse(BaseModel):
    """Generated flashcard response with applied count policy."""

    cards: list[FlashcardResponse]
    requested_count: int
    effective_count: int
    generated_count: int
    maximum_count: int
    notice: str | None = None


class SummarizeDocumentResponse(BaseModel):
    """Generated document summary."""

    summary: str


class AskStudyAgentRequestModel(BaseModel):
    """A free-form request grounded in one selected document."""

    prompt: str
    document_id: str


class AgentPlanStepResponse(BaseModel):
    """One validated action selected by the study agent."""

    task: StudyTaskName
    description: str


class AgentAnswerResultResponse(BaseModel):
    """A grounded answer selected by the study agent."""

    task: Literal["answer_question"]
    answer: str
    citations: list[CitationResponse]


class AgentSummaryResultResponse(BaseModel):
    """A summary selected by the study agent."""

    task: Literal["summarize_document"]
    summary: str


class AgentQuizResultResponse(BaseModel):
    """A quiz selected by the study agent."""

    task: Literal["generate_quiz"]
    questions: list[QuizQuestionResponse]
    requested_count: int
    effective_count: int
    generated_count: int
    maximum_count: int


class AgentFlashcardsResultResponse(BaseModel):
    """Flashcards selected by the study agent."""

    task: Literal["generate_flashcards"]
    cards: list[FlashcardResponse]
    requested_count: int
    effective_count: int
    generated_count: int
    maximum_count: int


AgentResultResponse = Annotated[
    AgentAnswerResultResponse
    | AgentSummaryResultResponse
    | AgentQuizResultResponse
    | AgentFlashcardsResultResponse,
    Field(discriminator="task"),
]


class AgentTaskErrorResponse(BaseModel):
    """A failure isolated to one selected study task."""

    task: StudyTaskName
    message: str


class AskStudyAgentResponse(BaseModel):
    """Structured result of a unified study-agent request."""

    status: StudyAgentStatusName
    plan: list[AgentPlanStepResponse]
    results: list[AgentResultResponse]
    notices: list[str]
    errors: list[AgentTaskErrorResponse]
    message: str | None = None


class PrepareStudySessionRequestModel(BaseModel):
    """Legacy study-agent request retained during API migration."""

    goal: str
    document_ids: list[str]
    question_count: int = 5
    session_id: str | None = None


class LegacyAgentPlanStepResponse(BaseModel):
    """One planned action in the legacy response shape."""

    tool_name: str
    description: str


class AgentQuizQuestionResponse(BaseModel):
    """One quiz question in the legacy response shape."""

    prompt: str
    answer: str


class PrepareStudySessionResponse(BaseModel):
    """Legacy response enriched with the unified typed results."""

    plan: list[LegacyAgentPlanStepResponse]
    summary: str
    quiz: list[AgentQuizQuestionResponse]
    recommendations: list[str]
    completed_tools: list[str]
    citations: list[CitationResponse]
    errors: list[str]
    results: list[AgentResultResponse]
    notices: list[str]
    status: StudyAgentStatusName
    message: str | None = None
