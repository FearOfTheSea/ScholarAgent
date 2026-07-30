"""Persistent adaptive single-document tutor endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from scholar_agent.application.dtos.tutor import (
    ContinueStudySessionRequest,
    StartStudySessionRequest,
    StudySessionResult,
    TutorActivity,
    TutorTurnResult,
)
from scholar_agent.domain.entities.study_session import (
    DocumentBrief,
    LearnerAttempt,
    LearnerLevel,
    SourceReference,
    StudyMode,
    TutorTurn,
)
from scholar_agent.domain.exceptions.document_not_found_error import (
    DocumentNotFoundError,
)
from scholar_agent.domain.value_objects.document_id import DocumentId
from scholar_agent.infrastructure.di.container import Container
from scholar_agent.presentation.api.dependencies import get_container
from scholar_agent.presentation.api.models import (
    ConceptNodeResponse,
    ContinueTutorSessionRequestModel,
    DocumentBriefResponse,
    GlossaryTermResponse,
    LearnerAttemptResponse,
    LearningObjectiveResponse,
    ObjectiveProgressResponse,
    SourceReferenceResponse,
    StartTutorSessionRequestModel,
    TutorActivityResponse,
    TutorSessionResponse,
    TutorTurnResponse,
    TutorTurnResultResponse,
)

router = APIRouter(prefix="/agent/sessions", tags=["adaptive tutor"])


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
