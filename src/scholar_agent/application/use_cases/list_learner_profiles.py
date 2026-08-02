"""List local learner profiles."""

from scholar_agent.domain.entities.learner_profile import LearnerProfile
from scholar_agent.domain.repositories.learner_profile_repository import (
    LearnerProfileRepository,
)


class ListLearnerProfilesUseCase:
    """Return local profiles in repository order."""

    def __init__(self, repository: LearnerProfileRepository) -> None:
        self._repository = repository

    def execute(self) -> tuple[LearnerProfile, ...]:
        return self._repository.list_profiles()
