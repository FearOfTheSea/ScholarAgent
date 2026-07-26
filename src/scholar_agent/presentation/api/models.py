"""HTTP request and response models."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


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
    """Request body for a grounded document question."""

    question: str
    document_ids: list[str] = []


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
    """Generated quiz response."""

    questions: list[QuizQuestionResponse]


class GenerateFlashcardsRequestModel(BaseModel):
    """Requested number of generated flashcards."""

    card_count: int = 10


class FlashcardResponse(BaseModel):
    """One generated flashcard."""

    front: str
    back: str


class GenerateFlashcardsResponse(BaseModel):
    """Generated flashcard response."""

    cards: list[FlashcardResponse]


class SummarizeDocumentResponse(BaseModel):
    """Generated document summary."""

    summary: str


class CompareDocumentsRequestModel(BaseModel):
    """Selected documents for a grounded comparison."""

    first_document_id: str
    second_document_id: str


class CompareDocumentsResponse(BaseModel):
    """Grounded comparison and citations."""

    comparison: str
    citations: list[CitationResponse]


class PrepareStudySessionRequestModel(BaseModel):
    """Goal and documents supplied to the study agent."""

    goal: str
    document_ids: list[str]
    question_count: int = 5
    session_id: str | None = None


class AgentPlanStepResponse(BaseModel):
    """One planned action returned by the study agent."""

    tool_name: str
    description: str


class AgentQuizQuestionResponse(BaseModel):
    """One quiz question returned by the study agent."""

    prompt: str
    answer: str


class PrepareStudySessionResponse(BaseModel):
    """Structured study-agent result."""

    plan: list[AgentPlanStepResponse]
    summary: str
    quiz: list[AgentQuizQuestionResponse]
    recommendations: list[str]
    completed_tools: list[str]
    citations: list[CitationResponse]
    errors: list[str]
