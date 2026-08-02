"""Export redacted learner profile state."""

from scholar_agent.application.services.learner_profile_serialization import (
    LearnerProfileSerializationService,
)
from scholar_agent.domain.repositories.learner_profile_repository import (
    LearnerProfileRepository,
)


class ExportLearnerProfileUseCase:
    """Export profile history without raw learner or model/source content."""

    def __init__(
        self,
        repository: LearnerProfileRepository,
        serialization: LearnerProfileSerializationService | None = None,
    ) -> None:
        self._repository = repository
        self._serialization = serialization or LearnerProfileSerializationService()

    def execute(self, profile_id: str) -> dict[str, object]:
        profile = self._repository.get_profile(profile_id)
        if profile is None:
            raise ValueError(f"Learner profile '{profile_id}' was not found.")
        return self._serialization.export(
            profile,
            self._repository.list_observations(profile_id),
            self._repository.list_candidates(profile_id),
            self._repository.list_equivalence_links(profile_id),
        )
