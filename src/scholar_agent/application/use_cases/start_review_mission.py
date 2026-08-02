"""Start one deterministic document-bound mission from a review queue item."""

from datetime import datetime

from scholar_agent.application.dtos.learner_profile import ReviewQueueEntry
from scholar_agent.application.dtos.tutor import (
    StartStudySessionRequest,
    StudySessionResult,
)
from scholar_agent.application.use_cases.get_review_queue import GetReviewQueueUseCase
from scholar_agent.application.use_cases.start_study_session import (
    StartStudySessionUseCase,
)
from scholar_agent.domain.entities.study_session import LearnerLevel, StudyMode
from scholar_agent.domain.value_objects.document_id import DocumentId


class StartReviewMissionUseCase:
    """Resolve one queue fingerprint to exactly one source objective."""

    def __init__(
        self,
        queue_use_case: GetReviewQueueUseCase,
        start_session: StartStudySessionUseCase,
    ) -> None:
        self._queue = queue_use_case
        self._start = start_session

    def execute(
        self,
        profile_id: str,
        fingerprint_value: str,
        as_of: datetime | None = None,
    ) -> StudySessionResult:
        if not fingerprint_value.strip():
            raise ValueError("fingerprint must not be blank.")
        matches = tuple(
            item
            for item in self._queue.execute(profile_id, as_of)
            if item.fingerprint.value == fingerprint_value
        )
        if len(matches) != 1:
            raise ValueError("Review fingerprint did not resolve to one concept.")
        entry: ReviewQueueEntry = matches[0]
        return self._start.execute(
            StartStudySessionRequest(
                document_id=DocumentId(entry.document_id),
                goal=f"Review: {entry.title}",
                learner_level=LearnerLevel.INTERMEDIATE,
                mode=StudyMode.GUIDED,
                target_minutes=entry.expected_minutes,
                learner_profile_id=profile_id,
                focus_objective_id=entry.objective_id,
            )
        )
