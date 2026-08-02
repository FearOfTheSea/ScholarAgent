"""Delete one profile and detach its missions."""

from dataclasses import dataclass

from scholar_agent.domain.repositories.learner_profile_repository import (
    LearnerProfileRepository,
)


@dataclass(frozen=True, slots=True)
class DeleteLearnerProfileResult:
    profile_id: str
    deleted: bool
    detached_session_count: int


class DeleteLearnerProfileUseCase:
    """Cascade profile data while preserving PDFs and mission history."""

    def __init__(self, repository: LearnerProfileRepository) -> None:
        self._repository = repository

    def execute(self, profile_id: str) -> DeleteLearnerProfileResult:
        if not profile_id.strip():
            raise ValueError("profile_id must not be blank.")
        existed = self._repository.get_profile(profile_id) is not None
        detached = self._repository.delete_profile(profile_id) if existed else 0
        return DeleteLearnerProfileResult(profile_id, existed, detached)
