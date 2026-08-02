"""Repair missing profile evidence from already-persisted mission state."""

from scholar_agent.application.services.mission_observations import (
    MissionObservationSyncService,
)


class SyncMissionObservationsUseCase:
    """Explicitly resynchronize all assessments for one learner profile."""

    def __init__(self, sync_service: MissionObservationSyncService) -> None:
        self._sync = sync_service

    def execute(self, profile_id: str) -> int:
        if not profile_id.strip():
            raise ValueError("profile_id must not be blank.")
        return self._sync.sync_profile(profile_id)
