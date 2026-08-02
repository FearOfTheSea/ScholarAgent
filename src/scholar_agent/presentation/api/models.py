"""HTTP request and response models."""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field

StudyTaskName = Literal[
    "answer_question",
    "semantic_search",
    "summarize_document",
    "generate_quiz",
    "generate_flashcards",
    "citation_lookup",
    "build_document_map",
    "explain_concept",
    "assess_learner_response",
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
    citations: list[SourceReferenceResponse] = Field(default_factory=list)


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
    citations: list[SourceReferenceResponse] = Field(default_factory=list)


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
    citations: list[SourceReferenceResponse] = Field(default_factory=list)


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
    citations: list[SourceReferenceResponse] = Field(default_factory=list)


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


MissionStatusName = Literal["active", "awaiting_learner", "completed", "failed"]
MilestoneKindName = Literal["orient", "learn", "practice", "review"]
MilestoneStatusName = Literal["pending", "active", "completed", "failed", "skipped"]


class StudyPlanResponse(BaseModel):
    """Bounded user-facing mission plan."""

    focus: str
    objective_ids: list[str]
    citations: list[SourceReferenceResponse]


class StudyMilestoneResponse(BaseModel):
    """One mission milestone."""

    id: str
    kind: MilestoneKindName
    title: str
    objective_id: str | None
    capability: str
    status: MilestoneStatusName
    citations: list[SourceReferenceResponse]


class PendingLearnerInteractionResponse(BaseModel):
    """A pending question without its hidden reference answer."""

    objective_id: str
    question: str
    capability: str
    citations: list[SourceReferenceResponse]
    attempts: int


class MissionTraceEventResponse(BaseModel):
    """A concise capability/state trace event."""

    event_type: str
    summary: str
    capability: str | None
    state: str | None
    created_at: datetime


class MissionLedgerProjectionResponse(BaseModel):
    """Replay-safe state projection stored in one ledger entry."""

    status: MissionStatusName
    active_milestone_id: str | None
    pending_objective_id: str | None
    action_count: int
    attempt_count: int
    artifact_count: int
    completed_milestone_count: int
    mastery_by_objective: list[dict[str, str]]
    next_milestone_id: str | None


class MissionCitationIdentityResponse(BaseModel):
    """A source identity without source text."""

    document_id: str
    chunk_id: str
    page_number: int | None


class MissionLedgerEntryResponse(BaseModel):
    """A redacted, verifiable mission transition."""

    sequence: int
    event_type: str
    summary: str
    objective_id: str | None
    capability: str | None
    citations: list[MissionCitationIdentityResponse]
    projection: MissionLedgerProjectionResponse
    previous_digest: str
    current_digest: str
    created_at: datetime


class MissionInsightsResponse(BaseModel):
    """Deterministic Mission Intelligence indicators."""

    progress_percent: float | None
    mastery_counts: dict[str, int]
    assessment_count: int
    first_pass_proficiency_rate: float | None
    remediation_cycles: int
    evidence_coverage: float | None
    action_budget_used: int
    action_budget_remaining: int
    ledger_verified: bool
    next_action: str
    signals: list[str]


class MissionLedgerVerificationResponse(BaseModel):
    """Result of checking the first broken ledger link."""

    valid: bool
    sequence: int | None
    reason: str | None


class MissionRecordSessionResponse(BaseModel):
    """Safe session identity and lifecycle metadata for export."""

    session_id: str
    document_id: str
    goal: str
    status: MissionStatusName
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


class MissionRecordPlanResponse(BaseModel):
    """Exported plan metadata and source identities."""

    focus: str
    objective_ids: list[str]
    citations: list[MissionCitationIdentityResponse]


class MissionRecordArtifactResponse(BaseModel):
    """Exported artifact metadata without generated content."""

    kind: Literal["summary", "quiz", "flashcards"]
    item_count: int
    citations: list[MissionCitationIdentityResponse]
    created_at: datetime


class MissionRecordResponse(BaseModel):
    """Versioned redacted mission record."""

    record_version: int
    session_schema_version: int
    session: MissionRecordSessionResponse
    plan: MissionRecordPlanResponse | None
    ledger: list[MissionLedgerEntryResponse]
    insights: MissionInsightsResponse
    citations: list[MissionCitationIdentityResponse]
    artifacts: list[MissionRecordArtifactResponse]


class StudyArtifactResponse(BaseModel):
    """A cited mission artifact."""

    kind: Literal["summary", "quiz", "flashcards"]
    summary: str | None = None
    questions: list[QuizQuestionResponse] = Field(default_factory=list)
    cards: list[FlashcardResponse] = Field(default_factory=list)
    citations: list[SourceReferenceResponse] = Field(default_factory=list)


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
    learner_profile_id: str | None = None


class TutorSessionResponse(BaseModel):
    """Complete resumable state for one tutor session."""

    session_id: str
    document_id: str
    learner_profile_id: str | None = None
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
    status: MissionStatusName = "active"
    plan: StudyPlanResponse | None = None
    milestones: list[StudyMilestoneResponse] = Field(default_factory=list)
    artifacts: list[StudyArtifactResponse] = Field(default_factory=list)
    pending_interaction: PendingLearnerInteractionResponse | None = None
    trace: list[MissionTraceEventResponse] = Field(default_factory=list)
    can_advance: bool = True
    completed_at: datetime | None = None


class ContinueTutorSessionRequestModel(BaseModel):
    """One learner message in a tutor session."""

    message: str


class AdvanceStudySessionRequestModel(BaseModel):
    """Optional learner action for a mission advance."""

    message: str | None = None


class TutorTurnResultResponse(BaseModel):
    """Result of one bounded adaptive tutor turn."""

    intent: str
    activity: TutorActivityResponse
    assessment: LearnerAttemptResponse | None
    progress: list[ObjectiveProgressResponse]
    current_objective_id: str | None
    status: MissionStatusName = "active"
    can_advance: bool = True
    trace: list[MissionTraceEventResponse] = Field(default_factory=list)


class LearnerProfileResponse(BaseModel):
    """Local learner profile metadata."""

    profile_id: str
    display_name: str
    target_date: date | None
    created_at: datetime
    updated_at: datetime


class CreateLearnerProfileRequestModel(BaseModel):
    """Create a private local profile."""

    display_name: str
    target_date: date | None = None


class ImportLearnerProfileRequestModel(BaseModel):
    """Import a complete profile only when replacement is explicit."""

    replace: bool = False
    payload: dict[str, object]


class ConceptFingerprintResponse(BaseModel):
    """Explainable stable concept identity."""

    algorithm_version: str
    fingerprint: str
    document_id: str
    normalized_title: str
    normalized_description: str


class ReviewQueueEntryResponse(BaseModel):
    """Plain-language review recommendation and learning signals."""

    fingerprint: ConceptFingerprintResponse
    document_id: str
    objective_id: str
    title: str
    description: str
    confidence: int
    uncertainty: int
    observation_count: int
    recall_count: int
    transfer_count: int
    last_observed_at: datetime | None
    due_at: datetime
    expected_minutes: int
    reason_codes: list[str]
    source_documents: list[str]


class CitationIdentityRequestModel(BaseModel):
    """Source identity without source excerpt."""

    document_id: str
    chunk_id: str
    page_number: int | None = None


class RecordReviewOutcomeRequestModel(BaseModel):
    """Strict redacted review result."""

    fingerprint: ConceptFingerprintResponse
    objective_id: str
    modality: Literal["recall", "transfer"]
    score: int = Field(ge=0, le=3)
    difficulty: int = Field(ge=1, le=3)
    citations: list[CitationIdentityRequestModel]
    observed_at: datetime | None = None


class EvidenceObservationResponse(BaseModel):
    """Stored review evidence without raw learner content."""

    observation_id: str
    profile_id: str
    fingerprint: ConceptFingerprintResponse
    document_id: str
    objective_id: str
    session_id: str | None
    source: Literal["mission", "review"]
    modality: Literal["recall", "transfer"]
    score: int
    difficulty: int
    citations: list[CitationIdentityRequestModel]
    observed_at: datetime


class EquivalenceCandidateResponse(BaseModel):
    """A proposed cross-document link awaiting consent."""

    profile_id: str
    source: ConceptFingerprintResponse
    target: ConceptFingerprintResponse
    similarity: float
    created_at: datetime


class EquivalenceDecisionRequestModel(BaseModel):
    """Accept or reject one listed equivalence candidate."""

    source_fingerprint: str
    target_fingerprint: str
    decision: Literal["accepted", "rejected"]


class EquivalenceDecisionResponse(BaseModel):
    """Persisted explicit equivalence decision."""

    profile_id: str
    source: ConceptFingerprintResponse
    target: ConceptFingerprintResponse
    decision: Literal["accepted", "rejected"]
    decided_at: datetime


class StartReviewMissionRequestModel(BaseModel):
    """Start a mission for one resolved queue concept."""

    fingerprint: str
    as_of: datetime | None = None
