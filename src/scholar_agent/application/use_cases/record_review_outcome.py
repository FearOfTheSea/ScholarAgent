"""Record one strict, redacted review outcome."""

from collections.abc import Callable
from datetime import UTC, datetime

from scholar_agent.application.dtos.learner_profile import RecordReviewOutcomeRequest
from scholar_agent.domain.entities.learner_profile import EvidenceObservation
from scholar_agent.domain.repositories.learner_profile_repository import (
    LearnerProfileRepository,
)


class RecordReviewOutcomeUseCase:
    """Validate and idempotently persist one review observation."""

    def __init__(
        self,
        profile_repository: LearnerProfileRepository,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._profiles = profile_repository
        self._clock = clock or (lambda: datetime.now(UTC))

    def execute(self, request: RecordReviewOutcomeRequest) -> EvidenceObservation:
        _validate_request(request)
        profile = self._profiles.get_profile(request.profile_id)
        if profile is None:
            raise ValueError(f"Learner profile '{request.profile_id}' was not found.")
        observation = EvidenceObservation.for_review(
            request.profile_id,
            request.fingerprint,
            request.objective_id,
            request.modality,
            request.score,
            request.difficulty,
            request.citations,
            request.observed_at or self._clock(),
            request.session_id,
        )
        self._profiles.append_observation(observation)
        return observation


def _validate_request(request: RecordReviewOutcomeRequest) -> None:
    if not request.profile_id.strip() or not request.objective_id.strip():
        raise ValueError("profile_id and objective_id must not be blank.")
    if request.score not in range(4):
        raise ValueError("score must be between 0 and 3.")
    if request.difficulty not in {1, 2, 3}:
        raise ValueError("difficulty must be between 1 and 3.")
    request.fingerprint.validate_canonical()
    if not request.citations:
        raise ValueError("Review outcomes require citations.")
    if any(
        citation.document_id != request.fingerprint.document_id
        for citation in request.citations
    ):
        raise ValueError("Review citations must use the fingerprint document.")
