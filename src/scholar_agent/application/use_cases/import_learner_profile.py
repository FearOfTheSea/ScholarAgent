"""Validate and atomically import a learner profile."""

from dataclasses import dataclass

from scholar_agent.application.services.learner_profile_serialization import (
    LearnerProfileSerializationService,
)
from scholar_agent.domain.entities.learner_profile import LearnerProfile
from scholar_agent.domain.repositories.learner_profile_repository import (
    LearnerProfileRepository,
)


@dataclass(frozen=True, slots=True)
class ImportLearnerProfileRequest:
    profile_id: str
    payload: object
    replace: bool = False


class ImportLearnerProfileUseCase:
    """Fail closed before replacing any existing profile rows."""

    def __init__(
        self,
        repository: LearnerProfileRepository,
        serialization: LearnerProfileSerializationService | None = None,
    ) -> None:
        self._repository = repository
        self._serialization = serialization or LearnerProfileSerializationService()

    def execute(self, request: ImportLearnerProfileRequest) -> LearnerProfile:
        if not request.profile_id.strip():
            raise ValueError("profile_id must not be blank.")
        imported = self._serialization.import_payload(
            request.payload, request.profile_id
        )
        if (
            self._repository.get_profile(request.profile_id) is not None
            and not request.replace
        ):
            raise ValueError("Profile already exists; explicit replace is required.")
        self._repository.replace_profile_data(
            imported.profile,
            imported.observations,
            imported.candidates,
            imported.links,
        )
        return imported.profile
