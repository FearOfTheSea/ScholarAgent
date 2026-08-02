"""Create a local learner profile."""

from datetime import UTC, datetime
from uuid import uuid4

from scholar_agent.application.dtos.learner_profile import CreateLearnerProfileRequest
from scholar_agent.domain.entities.learner_profile import LearnerProfile
from scholar_agent.domain.repositories.learner_profile_repository import (
    LearnerProfileRepository,
)


class CreateLearnerProfileUseCase:
    """Create profile metadata with no remote account or identity provider."""

    def __init__(self, repository: LearnerProfileRepository) -> None:
        self._repository = repository

    def execute(self, request: CreateLearnerProfileRequest) -> LearnerProfile:
        if not request.display_name.strip():
            raise ValueError("display_name must not be blank.")
        now = datetime.now(UTC)
        profile = LearnerProfile(
            str(uuid4()), request.display_name.strip(), request.target_date, now, now
        )
        self._repository.save_profile(profile)
        return profile
