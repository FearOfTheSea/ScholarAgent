"""List explicit equivalence decisions for one profile."""

from scholar_agent.domain.entities.learner_profile import ConceptEquivalenceLink
from scholar_agent.domain.repositories.learner_profile_repository import (
    LearnerProfileRepository,
)


class ListEquivalenceDecisionsUseCase:
    """Expose accepted and rejected links without changing their meaning."""

    def __init__(self, repository: LearnerProfileRepository) -> None:
        self._repository = repository

    def execute(self, profile_id: str) -> tuple[ConceptEquivalenceLink, ...]:
        return self._repository.list_equivalence_links(profile_id)
