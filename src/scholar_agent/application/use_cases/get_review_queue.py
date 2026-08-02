"""Return a synchronized deterministic review queue."""

from datetime import datetime

from scholar_agent.application.dtos.learner_profile import ReviewQueueEntry
from scholar_agent.application.services.mission_observations import (
    MissionObservationSyncService,
)
from scholar_agent.application.services.review_scheduler import ReviewScheduler
from scholar_agent.domain.repositories.learner_profile_repository import (
    LearnerProfileRepository,
)


class GetReviewQueueUseCase:
    """Resynchronize before calculating review recommendations."""

    def __init__(
        self,
        profile_repository: LearnerProfileRepository,
        scheduler: ReviewScheduler,
        sync_service: MissionObservationSyncService,
    ) -> None:
        self._profiles = profile_repository
        self._scheduler = scheduler
        self._sync = sync_service

    def execute(
        self, profile_id: str, as_of: datetime | None = None
    ) -> tuple[ReviewQueueEntry, ...]:
        if not profile_id.strip():
            raise ValueError("profile_id must not be blank.")
        profile = self._profiles.get_profile(profile_id)
        if profile is None:
            raise ValueError(f"Learner profile '{profile_id}' was not found.")
        self._sync.sync_profile(profile_id)
        return self._scheduler.queue(
            profile,
            self._profiles.list_observations(profile_id),
            self._profiles.list_equivalence_links(profile_id),
            as_of,
        )
