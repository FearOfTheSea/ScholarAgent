"""Get one local learner profile."""

from scholar_agent.domain.entities.learner_profile import LearnerProfile
from scholar_agent.domain.repositories.learner_profile_repository import (
    LearnerProfileRepository,
)


class GetLearnerProfileUseCase:
    """Read profile metadata without exposing learner evidence by default."""

    def __init__(self, repository: LearnerProfileRepository) -> None:
        self._repository = repository

    def execute(self, profile_id: str) -> LearnerProfile:
        profile = self._repository.get_profile(profile_id)
        if profile is None:
            raise ValueError(f"Learner profile '{profile_id}' was not found.")
        return profile
