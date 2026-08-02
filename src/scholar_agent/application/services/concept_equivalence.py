"""Deterministic descriptor similarity for consent-gated concept links."""

from scholar_agent.application.output_ports.equivalence_proposer import (
    ConceptEquivalenceProposer,
)
from scholar_agent.domain.entities.learner_profile import ConceptFingerprint


class DeterministicConceptEquivalenceProposer(ConceptEquivalenceProposer):
    """Compare normalized descriptor tokens without a model or prompt."""

    def similarity(self, left: ConceptFingerprint, right: ConceptFingerprint) -> float:
        if left.document_id == right.document_id:
            return 0.0
        left_tokens = set(left.descriptor.split())
        right_tokens = set(right.descriptor.split())
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
