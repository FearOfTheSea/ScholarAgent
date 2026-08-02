"""List consent-gated concept-equivalence candidates."""

from datetime import UTC, datetime

from scholar_agent.application.output_ports.equivalence_proposer import (
    ConceptEquivalenceProposer,
)
from scholar_agent.domain.entities.learner_profile import ConceptEquivalenceCandidate
from scholar_agent.domain.repositories.learner_profile_repository import (
    LearnerProfileRepository,
)


class ListEquivalenceCandidatesUseCase:
    """Expose proposals without applying them to knowledge estimates."""

    def __init__(
        self,
        repository: LearnerProfileRepository,
        proposer: ConceptEquivalenceProposer | None = None,
    ) -> None:
        self._repository = repository
        self._proposer = proposer

    def execute(self, profile_id: str) -> tuple[ConceptEquivalenceCandidate, ...]:
        candidates = self._repository.list_candidates(profile_id)
        if self._proposer is None:
            return candidates
        observations = self._repository.list_observations(profile_id)
        fingerprints = tuple(
            sorted(
                {item.fingerprint for item in observations},
                key=lambda item: item.value,
            )
        )
        existing = {
            frozenset((item.source.value, item.target.value)) for item in candidates
        }
        for index, left in enumerate(fingerprints):
            for right in fingerprints[index + 1 :]:
                if left.document_id == right.document_id:
                    continue
                pair = frozenset((left.value, right.value))
                if pair in existing:
                    continue
                similarity = self._proposer.similarity(left, right)
                if similarity < 0.5:
                    continue
                from scholar_agent.domain.entities.learner_profile import (
                    ConceptEquivalenceCandidate,
                )

                self._repository.save_candidate(
                    ConceptEquivalenceCandidate(
                        left,
                        right,
                        similarity,
                        datetime.now(UTC),
                        profile_id,
                    )
                )
        return self._repository.list_candidates(profile_id)
