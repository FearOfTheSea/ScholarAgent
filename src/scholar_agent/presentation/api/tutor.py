"""Persistent adaptive single-document tutor endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from scholar_agent.application.dtos.mission import AdvanceStudyMissionRequest
from scholar_agent.application.dtos.tutor import (
    ContinueStudySessionRequest,
    StartStudySessionRequest,
    StudySessionResult,
    TutorActivity,
    TutorTurnResult,
)
from scholar_agent.domain.entities.study_material import (
    FlashcardArtifact,
    QuizArtifact,
    SummaryArtifact,
)
from scholar_agent.domain.entities.study_session import (
    DocumentBrief,
    LearnerAttempt,
    LearnerLevel,
    MissionStatus,
    SourceReference,
    StudyArtifact,
    StudyMode,
    TutorTurn,
    objective_progress,
)
from scholar_agent.domain.exceptions.document_not_found_error import (
    DocumentNotFoundError,
)
from scholar_agent.domain.value_objects.document_id import DocumentId
from scholar_agent.infrastructure.di.container import Container
from scholar_agent.presentation.api.dependencies import get_container
from scholar_agent.presentation.api.models import (
    AdvanceStudySessionRequestModel,
    ConceptNodeResponse,
    ContinueTutorSessionRequestModel,
    DocumentBriefResponse,
    FlashcardResponse,
    GlossaryTermResponse,
    LearnerAttemptResponse,
    LearningObjectiveResponse,
    MissionInsightsResponse,
    MissionLedgerVerificationResponse,
    MissionRecordResponse,
    MissionTraceEventResponse,
    ObjectiveProgressResponse,
    PendingLearnerInteractionResponse,
    QuizQuestionResponse,
    SourceReferenceResponse,
    StartTutorSessionRequestModel,
    StudyArtifactResponse,
    StudyMilestoneResponse,
    StudyPlanResponse,
    TutorActivityResponse,
    TutorSessionResponse,
    TutorTurnResponse,
    TutorTurnResultResponse,
)

router = APIRouter(prefix="/agent/sessions", tags=["adaptive tutor"])


@router.get("/{session_id}/insights", response_model=MissionInsightsResponse)
def get_mission_insights(
    session_id: str,
    container: Annotated[Container, Depends(get_container)],
) -> MissionInsightsResponse:
    """Return deterministic, redacted Mission Intelligence signals."""
    try:
        insights = container.mission_insights_use_case().execute(session_id)
    except ValueError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
    return MissionInsightsResponse(
        progress_percent=insights.progress_percent,
        mastery_counts=insights.mastery_counts,
        assessment_count=insights.assessment_count,
        first_pass_proficiency_rate=insights.first_pass_proficiency_rate,
        remediation_cycles=insights.remediation_cycles,
        evidence_coverage=insights.evidence_coverage,
        action_budget_used=insights.action_budget_used,
        action_budget_remaining=insights.action_budget_remaining,
        ledger_verified=insights.ledger_verified,
        next_action=insights.next_action,
        signals=list(insights.signals),
    )


@router.get("/{session_id}/record", response_model=MissionRecordResponse)
def export_mission_record(
    session_id: str,
    container: Annotated[Container, Depends(get_container)],
) -> MissionRecordResponse:
    """Return a versioned export without private learner or source content."""
    try:
        record = container.export_mission_record_use_case().execute(session_id)
    except ValueError as error:
        code = (
            status.HTTP_404_NOT_FOUND
            if "was not found" in str(error)
            else status.HTTP_409_CONFLICT
        )
        raise HTTPException(code, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error
    return MissionRecordResponse.model_validate(record)


@router.post(
    "/{session_id}/record/verify",
    response_model=MissionLedgerVerificationResponse,
)
def verify_mission_record(
    session_id: str,
    container: Annotated[Container, Depends(get_container)],
) -> MissionLedgerVerificationResponse:
    """Verify the complete mission ledger and report its first broken link."""
    try:
        result = container.verify_mission_ledger_use_case().execute(session_id)
    except ValueError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
    return MissionLedgerVerificationResponse(
        valid=result.valid,
        sequence=result.sequence,
        reason=result.reason,
    )


@router.get("", response_model=list[TutorSessionResponse])
def list_sessions(
    container: Annotated[Container, Depends(get_container)],
    document_id: str | None = Query(default=None),
    session_status: str | None = Query(default=None, alias="status"),
) -> list[TutorSessionResponse]:
    """List missions ordered by most recently updated."""
    try:
        status_filter = (
            MissionStatus(session_status) if session_status is not None else None
        )
        sessions = container.list_study_sessions_use_case().execute(
            DocumentId(document_id) if document_id is not None else None,
            status_filter,
        )
    except ValueError as error:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(error)) from error
    return [
        _session_response(
            StudySessionResult(
                session=session,
                progress=tuple(
                    objective_progress(item.identifier, session.attempts)
                    for item in session.brief.objectives
                ),
                current_objective_id=(
                    session.plan.objective_ids[0]
                    if session.plan is not None and session.plan.objective_ids
                    else None
                ),
            )
        )
        for session in sessions
    ]


@router.post(
    "", response_model=TutorSessionResponse, status_code=status.HTTP_201_CREATED
)
def start_session(
    request: StartTutorSessionRequestModel,
    container: Annotated[Container, Depends(get_container)],
) -> TutorSessionResponse:
    """Build a cited document map and start a persistent session."""
    try:
        result = container.start_study_session_use_case().execute(
            StartStudySessionRequest(
                document_id=DocumentId(request.document_id),
                goal=request.goal,
                learner_level=LearnerLevel(request.learner_level),
                mode=StudyMode(request.mode),
                target_minutes=request.target_minutes,
            )
        )
    except DocumentNotFoundError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
    except ValueError as error:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from error
    return _session_response(result)


@router.post("/{session_id}/turns", response_model=TutorTurnResultResponse)
def continue_session(
    session_id: str,
    request: ContinueTutorSessionRequestModel,
    container: Annotated[Container, Depends(get_container)],
) -> TutorTurnResultResponse:
    """Run one bounded, grounded tutor turn."""
    try:
        result = container.continue_study_session_use_case().execute(
            ContinueStudySessionRequest(session_id, request.message)
        )
    except ValueError as error:
        code = (
            status.HTTP_404_NOT_FOUND
            if "was not found" in str(error)
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(code, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from error
    return _turn_result_response(result)


@router.post("/{session_id}/advance", response_model=TutorSessionResponse)
def advance_session(
    session_id: str,
    request: AdvanceStudySessionRequestModel,
    container: Annotated[Container, Depends(get_container)],
) -> TutorSessionResponse:
    """Advance a persistent mission by at most the configured action budget."""
    try:
        result = container.advance_study_session_use_case().execute(
            AdvanceStudyMissionRequest(session_id, request.message)
        )
    except ValueError as error:
        code = (
            status.HTTP_404_NOT_FOUND
            if "was not found" in str(error)
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(code, str(error)) from error
    return _session_response(result)


@router.post("/{session_id}/complete", response_model=TutorSessionResponse)
def complete_session(
    session_id: str,
    container: Annotated[Container, Depends(get_container)],
) -> TutorSessionResponse:
    """Manually complete a mission while preserving its evidence."""
    try:
        result = container.complete_study_session_use_case().execute(session_id)
    except ValueError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
    return _session_response(result)


@router.get("/{session_id}", response_model=TutorSessionResponse)
def get_session(
    session_id: str,
    container: Annotated[Container, Depends(get_container)],
) -> TutorSessionResponse:
    """Return complete state needed to resume a session."""
    try:
        result = container.get_study_session_use_case().execute(session_id)
    except ValueError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
    return _session_response(result)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    session_id: str,
    container: Annotated[Container, Depends(get_container)],
) -> None:
    """Delete one session while retaining its source document."""
    result = container.delete_study_session_use_case().execute(session_id)
    if not result.deleted:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Study session '{session_id}' was not found.",
        )


def _session_response(result: StudySessionResult) -> TutorSessionResponse:
    session = result.session
    return TutorSessionResponse(
        session_id=session.identifier,
        document_id=session.document_id.value,
        goal=session.goal,
        learner_level=session.learner_level.value,
        mode=session.mode.value,
        target_minutes=session.target_minutes,
        brief=_brief_response(session.brief),
        progress=[
            ObjectiveProgressResponse(
                objective_id=item.objective_id,
                percentage=item.percentage,
                label=item.label.value,
                attempt_count=item.attempt_count,
            )
            for item in result.progress
        ],
        current_objective_id=result.current_objective_id,
        activity=(
            _activity_response(result.activity) if result.activity is not None else None
        ),
        turns=[_turn_response(item) for item in session.turns],
        created_at=session.created_at,
        updated_at=session.updated_at,
        status=session.status.value,
        plan=(
            StudyPlanResponse(
                focus=session.plan.focus,
                objective_ids=list(session.plan.objective_ids),
                citations=[
                    _reference_response(item) for item in session.plan.citations
                ],
            )
            if session.plan is not None
            else None
        ),
        milestones=[
            StudyMilestoneResponse(
                id=item.identifier,
                kind=item.kind.value,
                title=item.title,
                objective_id=item.objective_id,
                capability=item.capability,
                status=item.status.value,
                citations=[_reference_response(ref) for ref in item.citations],
            )
            for item in session.milestones
        ],
        artifacts=[_artifact_response(item) for item in session.artifacts],
        pending_interaction=(
            PendingLearnerInteractionResponse(
                objective_id=session.pending_interaction.objective_id,
                question=session.pending_interaction.question,
                capability=session.pending_interaction.capability,
                citations=[
                    _reference_response(ref)
                    for ref in session.pending_interaction.citations
                ],
                attempts=session.pending_interaction.attempts,
            )
            if session.pending_interaction is not None
            else None
        ),
        trace=[
            MissionTraceEventResponse(
                event_type=item.event_type,
                summary=item.summary,
                capability=item.capability,
                state=item.state,
                created_at=item.created_at,
            )
            for item in session.trace
        ],
        can_advance=result.can_advance,
        completed_at=session.completed_at,
    )


def _brief_response(brief: DocumentBrief) -> DocumentBriefResponse:
    return DocumentBriefResponse(
        document_id=brief.document_id.value,
        synopsis=brief.synopsis,
        objectives=[
            LearningObjectiveResponse(
                id=item.identifier,
                title=item.title,
                description=item.description,
                prerequisite_ids=list(item.prerequisite_ids),
                citations=[_reference_response(ref) for ref in item.citations],
            )
            for item in brief.objectives
        ],
        concepts=[
            ConceptNodeResponse(
                id=item.identifier,
                label=item.label,
                explanation=item.explanation,
                prerequisite_ids=list(item.prerequisite_ids),
                citations=[_reference_response(ref) for ref in item.citations],
            )
            for item in brief.concepts
        ],
        glossary=[
            GlossaryTermResponse(
                term=item.term,
                definition=item.definition,
                citations=[_reference_response(ref) for ref in item.citations],
            )
            for item in brief.glossary
        ],
        misconceptions=list(brief.misconceptions),
    )


def _turn_result_response(result: TutorTurnResult) -> TutorTurnResultResponse:
    return TutorTurnResultResponse(
        intent=result.intent,
        activity=_activity_response(result.activity),
        assessment=(
            _attempt_response(result.assessment)
            if result.assessment is not None
            else None
        ),
        progress=[
            ObjectiveProgressResponse(
                objective_id=item.objective_id,
                percentage=item.percentage,
                label=item.label.value,
                attempt_count=item.attempt_count,
            )
            for item in result.progress
        ],
        current_objective_id=result.current_objective_id,
    )


def _activity_response(activity: TutorActivity) -> TutorActivityResponse:
    return TutorActivityResponse(
        kind=activity.kind.value,
        message=activity.message,
        objective_id=activity.objective_id,
        citations=[_reference_response(ref) for ref in activity.citations],
    )


def _turn_response(turn: TutorTurn) -> TutorTurnResponse:
    return TutorTurnResponse(
        kind=turn.kind.value,
        learner_message=turn.learner_message,
        tutor_message=turn.tutor_message,
        objective_id=turn.objective_id,
        citations=[_reference_response(ref) for ref in turn.citations],
        assessment=(
            _attempt_response(turn.assessment) if turn.assessment is not None else None
        ),
        created_at=turn.created_at,
    )


def _attempt_response(attempt: LearnerAttempt) -> LearnerAttemptResponse:
    return LearnerAttemptResponse(
        objective_id=attempt.objective_id,
        response=attempt.response,
        score=attempt.score,
        feedback=attempt.feedback,
        missing_concepts=list(attempt.missing_concepts),
        citations=[_reference_response(ref) for ref in attempt.citations],
        created_at=attempt.created_at,
    )


def _reference_response(reference: SourceReference) -> SourceReferenceResponse:
    return SourceReferenceResponse(
        document_id=reference.document_id.value,
        chunk_id=reference.chunk_id,
        page_number=reference.page_number,
        excerpt=reference.excerpt,
    )


def _artifact_response(artifact: StudyArtifact) -> StudyArtifactResponse:
    if isinstance(artifact, SummaryArtifact):
        return StudyArtifactResponse(
            kind="summary",
            summary=artifact.text,
            citations=[_reference_response(item) for item in artifact.citations],
        )
    if isinstance(artifact, QuizArtifact):
        return StudyArtifactResponse(
            kind="quiz",
            questions=[
                QuizQuestionResponse(
                    prompt=item.prompt,
                    answer=item.answer,
                    citations=[_reference_response(ref) for ref in item.citations],
                )
                for item in artifact.questions
            ],
            citations=[_reference_response(item) for item in artifact.citations],
        )
    if isinstance(artifact, FlashcardArtifact):
        return StudyArtifactResponse(
            kind="flashcards",
            cards=[
                FlashcardResponse(
                    front=item.front,
                    back=item.back,
                    citations=[_reference_response(ref) for ref in item.citations],
                )
                for item in artifact.cards
            ],
            citations=[_reference_response(item) for item in artifact.citations],
        )
    raise ValueError("Unsupported study artifact.")
