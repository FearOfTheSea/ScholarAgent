"""Application port for deterministic concept-equivalence proposals."""

from abc import ABC, abstractmethod

from scholar_agent.domain.entities.learner_profile import ConceptFingerprint


class ConceptEquivalenceProposer(ABC):
    """Suggest similarities without granting consent or pooling evidence."""

    @abstractmethod
    def similarity(self, left: ConceptFingerprint, right: ConceptFingerprint) -> float:
        """Return a deterministic descriptor similarity in the range 0..1."""
