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


LearnerLevelName = Literal["beginner", "intermediate", "advanced"]
StudyModeName = Literal["guided", "exam", "cram"]
MasteryLabelName = Literal["unseen", "developing", "proficient", "mastered"]
TutorTurnKindName = Literal[
    "explanation",
    "question",
    "assessment",
    "hint",
    "recap",
    "unsupported",
]


class SourceReferenceResponse(BaseModel):
    """Evidence displayed in the adaptive tutor."""

    document_id: str
    chunk_id: str
    page_number: int | None
    excerpt: str


class LearningObjectiveResponse(BaseModel):
    """One cited learning objective."""

    id: str
    title: str
    description: str
    prerequisite_ids: list[str]
    citations: list[SourceReferenceResponse]


class ConceptNodeResponse(BaseModel):
    """One node in the document knowledge map."""

    id: str
    label: str
    explanation: str
    prerequisite_ids: list[str]
    citations: list[SourceReferenceResponse]


class GlossaryTermResponse(BaseModel):
    """One cited glossary definition."""

    term: str
    definition: str
    citations: list[SourceReferenceResponse]


class DocumentBriefResponse(BaseModel):
    """Cited learning map derived from one document."""

    document_id: str
    synopsis: str
    objectives: list[LearningObjectiveResponse]
    concepts: list[ConceptNodeResponse]
    glossary: list[GlossaryTermResponse]
    misconceptions: list[str]


class ObjectiveProgressResponse(BaseModel):
    """Current deterministic mastery for an objective."""

    objective_id: str
    percentage: int
    label: MasteryLabelName
    attempt_count: int


class TutorActivityResponse(BaseModel):
    """One learner-facing tutor activity."""

    kind: TutorTurnKindName
    message: str
    objective_id: str | None
    citations: list[SourceReferenceResponse]


class LearnerAttemptResponse(BaseModel):
    """Structured assessment of a learner response."""

    objective_id: str
    response: str
    score: int
    feedback: str
    missing_concepts: list[str]
    citations: list[SourceReferenceResponse]
    created_at: datetime


class TutorTurnResponse(BaseModel):
    """One persisted tutor exchange."""

    kind: TutorTurnKindName
    learner_message: str
    tutor_message: str
    objective_id: str | None
    citations: list[SourceReferenceResponse]
    assessment: LearnerAttemptResponse | None
    created_at: datetime


class StartTutorSessionRequestModel(BaseModel):
    """Start a persistent session over one document."""

    document_id: str
    goal: str = "Understand the document and retain its key ideas."
    learner_level: LearnerLevelName = "intermediate"
    mode: StudyModeName = "guided"
    target_minutes: int = 30


class TutorSessionResponse(BaseModel):
    """Complete resumable state for one tutor session."""

    session_id: str
    document_id: str
    goal: str
    learner_level: LearnerLevelName
    mode: StudyModeName
    target_minutes: int
    brief: DocumentBriefResponse
    progress: list[ObjectiveProgressResponse]
    current_objective_id: str | None
    activity: TutorActivityResponse | None
    turns: list[TutorTurnResponse]
    created_at: datetime
    updated_at: datetime


class ContinueTutorSessionRequestModel(BaseModel):
    """One learner message in a tutor session."""

    message: str


class TutorTurnResultResponse(BaseModel):
    """Result of one bounded adaptive tutor turn."""

    intent: str
    activity: TutorActivityResponse
    assessment: LearnerAttemptResponse | None
    progress: list[ObjectiveProgressResponse]
    current_objective_id: str | None
