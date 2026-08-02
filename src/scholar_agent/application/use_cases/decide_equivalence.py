"""Accept or reject one cross-document equivalence candidate."""

from dataclasses import dataclass
from datetime import UTC, datetime

from scholar_agent.domain.entities.learner_profile import (
    ConceptEquivalenceLink,
    EquivalenceDecision,
)
from scholar_agent.domain.repositories.learner_profile_repository import (
    LearnerProfileRepository,
)


@dataclass(frozen=True, slots=True)
class DecideEquivalenceRequest:
    profile_id: str
    source_fingerprint: str
    target_fingerprint: str
    decision: EquivalenceDecision


class DecideEquivalenceUseCase:
    """Persist explicit consent for one proposed cross-document link."""

    def __init__(self, repository: LearnerProfileRepository) -> None:
        self._repository = repository

    def execute(self, request: DecideEquivalenceRequest) -> ConceptEquivalenceLink:
        candidates = self._repository.list_candidates(request.profile_id)
        candidate = next(
            (
                item
                for item in candidates
                if {
                    item.source.value,
                    item.target.value,
                }
                == {request.source_fingerprint, request.target_fingerprint}
            ),
            None,
        )
        if candidate is None:
            raise ValueError("Equivalence candidate was not found.")
        link = ConceptEquivalenceLink(
            candidate.source,
            candidate.target,
            request.decision,
            datetime.now(UTC),
            request.profile_id,
        )
        self._repository.save_equivalence_link(link)
        return link
