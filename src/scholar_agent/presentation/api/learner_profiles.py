"""Local learner profile and document-bound review endpoints."""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from scholar_agent.application.dtos.learner_profile import (
    CreateLearnerProfileRequest,
    RecordReviewOutcomeRequest,
)
from scholar_agent.application.use_cases.decide_equivalence import (
    DecideEquivalenceRequest,
)
from scholar_agent.application.use_cases.import_learner_profile import (
    ImportLearnerProfileRequest,
)
from scholar_agent.domain.entities.learner_profile import (
    ConceptEquivalenceCandidate,
    ConceptEquivalenceLink,
    ConceptFingerprint,
    EquivalenceDecision,
    ObservationModality,
)
from scholar_agent.domain.value_objects.citation_identity import CitationIdentity
from scholar_agent.domain.value_objects.document_id import DocumentId
from scholar_agent.infrastructure.di.container import Container
from scholar_agent.presentation.api.dependencies import get_container
from scholar_agent.presentation.api.models import (
    CitationIdentityRequestModel,
    ConceptFingerprintResponse,
    CreateLearnerProfileRequestModel,
    EquivalenceCandidateResponse,
    EquivalenceDecisionRequestModel,
    EquivalenceDecisionResponse,
    EvidenceObservationResponse,
    ImportLearnerProfileRequestModel,
    LearnerProfileResponse,
    RecordReviewOutcomeRequestModel,
    ReviewQueueEntryResponse,
    StartReviewMissionRequestModel,
    TutorSessionResponse,
)
from scholar_agent.presentation.api.tutor import _session_response

router = APIRouter(prefix="/learner-profiles", tags=["learner profiles"])


@router.post(
    "", response_model=LearnerProfileResponse, status_code=status.HTTP_201_CREATED
)
def create_profile(
    request: CreateLearnerProfileRequestModel,
    container: Annotated[Container, Depends(get_container)],
) -> LearnerProfileResponse:
    try:
        profile = container.create_learner_profile_use_case().execute(
            CreateLearnerProfileRequest(request.display_name, request.target_date)
        )
    except ValueError as error:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(error)) from error
    return _profile_response(profile)


@router.get("", response_model=list[LearnerProfileResponse])
def list_profiles(
    container: Annotated[Container, Depends(get_container)],
) -> list[LearnerProfileResponse]:
    return [
        _profile_response(profile)
        for profile in container.list_learner_profiles_use_case().execute()
    ]


@router.get("/{profile_id}", response_model=LearnerProfileResponse)
def get_profile(
    profile_id: str,
    container: Annotated[Container, Depends(get_container)],
) -> LearnerProfileResponse:
    try:
        profile = container.get_learner_profile_use_case().execute(profile_id)
    except ValueError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
    return _profile_response(profile)


@router.delete("/{profile_id}", response_model=dict[str, object])
def delete_profile(
    profile_id: str,
    container: Annotated[Container, Depends(get_container)],
) -> dict[str, object]:
    result = container.delete_learner_profile_use_case().execute(profile_id)
    if not result.deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Learner profile was not found.")
    return {
        "profile_id": result.profile_id,
        "deleted": result.deleted,
        "detached_session_count": result.detached_session_count,
    }


@router.get("/{profile_id}/export", response_model=dict[str, object])
def export_profile(
    profile_id: str,
    container: Annotated[Container, Depends(get_container)],
) -> dict[str, object]:
    try:
        return container.export_learner_profile_use_case().execute(profile_id)
    except ValueError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error


@router.post("/{profile_id}/import", response_model=LearnerProfileResponse)
def import_profile(
    profile_id: str,
    request: ImportLearnerProfileRequestModel,
    container: Annotated[Container, Depends(get_container)],
) -> LearnerProfileResponse:
    try:
        profile = container.import_learner_profile_use_case().execute(
            ImportLearnerProfileRequest(profile_id, request.payload, request.replace)
        )
    except ValueError as error:
        code = (
            status.HTTP_409_CONFLICT
            if "replace" in str(error) or "already exists" in str(error)
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(code, str(error)) from error
    return _profile_response(profile)


@router.get("/{profile_id}/review-queue", response_model=list[ReviewQueueEntryResponse])
def review_queue(
    profile_id: str,
    container: Annotated[Container, Depends(get_container)],
    as_of: datetime | None = None,
) -> list[ReviewQueueEntryResponse]:
    try:
        queue = container.get_review_queue_use_case().execute(profile_id, as_of)
    except ValueError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
    return [_queue_response(item) for item in queue]


@router.post(
    "/{profile_id}/review-outcomes",
    response_model=EvidenceObservationResponse,
    status_code=status.HTTP_201_CREATED,
)
def record_review_outcome(
    profile_id: str,
    request: RecordReviewOutcomeRequestModel,
    container: Annotated[Container, Depends(get_container)],
) -> EvidenceObservationResponse:
    try:
        fingerprint = _fingerprint(request.fingerprint)
        observation = container.record_review_outcome_use_case().execute(
            RecordReviewOutcomeRequest(
                profile_id,
                fingerprint,
                request.objective_id,
                ObservationModality(request.modality),
                request.score,
                request.difficulty,
                tuple(_citation(item) for item in request.citations),
                request.observed_at or datetime.now(UTC),
            )
        )
    except ValueError as error:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(error)) from error
    return _observation_response(observation)


@router.get(
    "/{profile_id}/equivalence-candidates",
    response_model=list[EquivalenceCandidateResponse],
)
def equivalence_candidates(
    profile_id: str,
    container: Annotated[Container, Depends(get_container)],
) -> list[EquivalenceCandidateResponse]:
    candidates = container.list_equivalence_candidates_use_case().execute(profile_id)
    return [_candidate_response(item) for item in candidates]


@router.post(
    "/{profile_id}/equivalence-decisions",
    response_model=EquivalenceDecisionResponse,
)
def decide_equivalence(
    profile_id: str,
    request: EquivalenceDecisionRequestModel,
    container: Annotated[Container, Depends(get_container)],
) -> EquivalenceDecisionResponse:
    try:
        link = container.decide_equivalence_use_case().execute(
            DecideEquivalenceRequest(
                profile_id,
                request.source_fingerprint,
                request.target_fingerprint,
                EquivalenceDecision(request.decision),
            )
        )
    except ValueError as error:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(error)) from error
    return _link_response(link)


@router.get(
    "/{profile_id}/equivalence-decisions",
    response_model=list[EquivalenceDecisionResponse],
)
def list_equivalence_decisions(
    profile_id: str,
    container: Annotated[Container, Depends(get_container)],
) -> list[EquivalenceDecisionResponse]:
    return [
        _link_response(item)
        for item in container.list_equivalence_decisions_use_case().execute(profile_id)
    ]


@router.post("/{profile_id}/review-missions", response_model=TutorSessionResponse)
def start_review_mission(
    profile_id: str,
    request: StartReviewMissionRequestModel,
    container: Annotated[Container, Depends(get_container)],
) -> TutorSessionResponse:
    try:
        result = container.start_review_mission_use_case().execute(
            profile_id, request.fingerprint, request.as_of
        )
    except ValueError as error:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(error)) from error
    return _session_response(result)


def _profile_response(profile: object) -> LearnerProfileResponse:
    from scholar_agent.domain.entities.learner_profile import LearnerProfile

    if not isinstance(profile, LearnerProfile):
        raise ValueError("Unsupported learner profile.")
    return LearnerProfileResponse(
        profile_id=profile.identifier,
        display_name=profile.display_name,
        target_date=profile.target_date,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


def _fingerprint_response(
    fingerprint: ConceptFingerprint,
) -> ConceptFingerprintResponse:
    return ConceptFingerprintResponse(
        algorithm_version=fingerprint.algorithm_version,
        fingerprint=fingerprint.value,
        document_id=fingerprint.document_id.value,
        normalized_title=fingerprint.normalized_title,
        normalized_description=fingerprint.normalized_description,
    )


def _fingerprint(model: ConceptFingerprintResponse) -> ConceptFingerprint:
    return ConceptFingerprint(
        model.algorithm_version,
        model.fingerprint,
        DocumentId(model.document_id),
        model.normalized_title,
        model.normalized_description,
    )


def _citation(model: CitationIdentityRequestModel) -> CitationIdentity:
    return CitationIdentity(
        DocumentId(model.document_id), model.chunk_id, model.page_number
    )


def _queue_response(item: object) -> ReviewQueueEntryResponse:
    from scholar_agent.application.dtos.learner_profile import ReviewQueueEntry

    if not isinstance(item, ReviewQueueEntry):
        raise ValueError("Unsupported review queue entry.")
    return ReviewQueueEntryResponse(
        fingerprint=_fingerprint_response(item.fingerprint),
        document_id=item.document_id,
        objective_id=item.objective_id,
        title=item.title,
        description=item.description,
        confidence=item.confidence,
        uncertainty=item.uncertainty,
        observation_count=item.observation_count,
        recall_count=item.recall_count,
        transfer_count=item.transfer_count,
        last_observed_at=item.last_observed_at,
        due_at=item.due_at,
        expected_minutes=item.expected_minutes,
        reason_codes=list(item.reason_codes),
        source_documents=list(item.source_documents),
    )


def _observation_response(observation: object) -> EvidenceObservationResponse:
    from scholar_agent.domain.entities.learner_profile import EvidenceObservation

    if not isinstance(observation, EvidenceObservation):
        raise ValueError("Unsupported evidence observation.")
    return EvidenceObservationResponse(
        observation_id=observation.identifier,
        profile_id=observation.profile_id,
        fingerprint=_fingerprint_response(observation.fingerprint),
        document_id=observation.document_id.value,
        objective_id=observation.objective_id,
        session_id=observation.session_id,
        source=observation.source.value,
        modality=observation.modality.value,
        score=observation.score,
        difficulty=observation.difficulty,
        citations=[
            CitationIdentityRequestModel(
                document_id=item.document_id.value,
                chunk_id=item.chunk_id,
                page_number=item.page_number,
            )
            for item in observation.citations
        ],
        observed_at=observation.observed_at,
    )


def _candidate_response(
    candidate: ConceptEquivalenceCandidate,
) -> EquivalenceCandidateResponse:
    return EquivalenceCandidateResponse(
        profile_id=candidate.profile_id,
        source=_fingerprint_response(candidate.source),
        target=_fingerprint_response(candidate.target),
        similarity=candidate.similarity,
        created_at=candidate.created_at,
    )


def _link_response(link: ConceptEquivalenceLink) -> EquivalenceDecisionResponse:
    return EquivalenceDecisionResponse(
        profile_id=link.profile_id,
        source=_fingerprint_response(link.source),
        target=_fingerprint_response(link.target),
        decision=link.decision.value,
        decided_at=link.decided_at,
    )
