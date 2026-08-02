"""Domain value objects."""

from scholar_agent.domain.value_objects.citation_identity import CitationIdentity
from scholar_agent.domain.value_objects.document_id import DocumentId

__all__ = ["CitationIdentity", "ConceptFingerprint", "DocumentId"]


def __getattr__(name: str) -> object:
    if name == "ConceptFingerprint":
        from scholar_agent.domain.value_objects.concept_fingerprint import (
            ConceptFingerprint,
        )

        return ConceptFingerprint
    raise AttributeError(name)
