"""Private longitudinal learner evidence domain objects."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum

from scholar_agent.domain.value_objects.citation_identity import CitationIdentity
from scholar_agent.domain.value_objects.document_id import DocumentId

CONCEPT_FINGERPRINT_ALGORITHM = "nfkc-casefold-punct-v1"


class ObservationSource(StrEnum):
    """Where an evidence observation was produced."""

    MISSION = "mission"
    REVIEW = "review"


class ObservationModality(StrEnum):
    """Whether the learner was asked to recall or transfer knowledge."""

    RECALL = "recall"
    TRANSFER = "transfer"


class EquivalenceDecision(StrEnum):
    """Explicit learner consent state for cross-document pooling."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class LearnerProfile:
    """Local profile metadata; observations are held by its repository."""

    identifier: str
    display_name: str
    target_date: date | None = None
    created_at: datetime = datetime.min.replace(tzinfo=UTC)
    updated_at: datetime = datetime.min.replace(tzinfo=UTC)

    def __post_init__(self) -> None:
        if not self.identifier.strip():
            raise ValueError("Learner profile identifier must not be blank.")
        if not self.display_name.strip():
            raise ValueError("Learner profile display name must not be blank.")

    @classmethod
    def local_default(cls, now: datetime) -> LearnerProfile:
        """Return the stable local profile identity used by default."""
        return cls("local-default", "Local learner", None, now, now)


@dataclass(frozen=True, slots=True)
class ConceptFingerprint:
    """Stable, explainable identity for one document-local objective."""

    algorithm_version: str
    value: str
    document_id: DocumentId
    normalized_title: str
    normalized_description: str

    def __post_init__(self) -> None:
        if not self.algorithm_version.strip():
            raise ValueError("Concept fingerprint algorithm version is required.")
        if self.algorithm_version != CONCEPT_FINGERPRINT_ALGORITHM:
            raise ValueError(
                f"Unsupported concept fingerprint algorithm '{self.algorithm_version}'."
            )
        if len(self.value) != 64 or any(
            character not in "0123456789abcdef" for character in self.value
        ):
            raise ValueError("Concept fingerprint must be a SHA-256 hex digest.")
        self.validate_canonical()

    @classmethod
    def from_descriptor(
        cls,
        document_id: DocumentId,
        title: str,
        description: str,
        algorithm_version: str = CONCEPT_FINGERPRINT_ALGORITHM,
    ) -> ConceptFingerprint:
        normalized_title = normalize_concept_text(title)
        normalized_description = normalize_concept_text(description)
        digest = _concept_fingerprint_digest(
            algorithm_version,
            document_id,
            normalized_title,
            normalized_description,
        )
        return cls(
            algorithm_version,
            digest,
            document_id,
            normalized_title,
            normalized_description,
        )

    @property
    def descriptor(self) -> str:
        """Human-readable normalized descriptor for review explanations."""
        return f"{self.normalized_title}: {self.normalized_description}"

    def validate_canonical(self) -> None:
        """Reject a digest that does not match the normalized descriptor."""
        if self.normalized_title != normalize_concept_text(self.normalized_title):
            raise ValueError("Concept fingerprint title is not normalized.")
        if self.normalized_description != normalize_concept_text(
            self.normalized_description
        ):
            raise ValueError("Concept fingerprint description is not normalized.")
        expected = _concept_fingerprint_digest(
            self.algorithm_version,
            self.document_id,
            self.normalized_title,
            self.normalized_description,
        )
        if self.value != expected:
            raise ValueError("Concept fingerprint does not match its descriptor.")


def normalize_concept_text(value: str) -> str:
    """Apply the versioned NFKC, casefold, punctuation, and whitespace policy."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    without_punctuation = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    return re.sub(r"\s+", " ", without_punctuation).strip()


@dataclass(frozen=True, slots=True)
class EvidenceObservation:
    """Redacted evidence used for longitudinal estimates."""

    identifier: str
    profile_id: str
    fingerprint: ConceptFingerprint
    document_id: DocumentId
    objective_id: str
    session_id: str | None
    source: ObservationSource
    modality: ObservationModality
    score: int
    difficulty: int
    citations: tuple[CitationIdentity, ...]
    observed_at: datetime

    def __post_init__(self) -> None:
        if not self.identifier.strip() or not self.profile_id.strip():
            raise ValueError("Observation identifiers must not be blank.")
        if not self.objective_id.strip():
            raise ValueError("Observation objective_id must not be blank.")
        if self.fingerprint.document_id != self.document_id:
            raise ValueError("Observation fingerprint must use its document.")
        self.fingerprint.validate_canonical()
        if self.score not in range(4):
            raise ValueError("Observation score must be between 0 and 3.")
        if self.difficulty not in {1, 2, 3}:
            raise ValueError("Observation difficulty must be between 1 and 3.")
        if not self.citations:
            raise ValueError("Evidence observations require citations.")
        if any(item.document_id != self.document_id for item in self.citations):
            raise ValueError("Observation citations must use its document.")

    @classmethod
    def for_mission(
        cls,
        profile_id: str,
        fingerprint: ConceptFingerprint,
        objective_id: str,
        session_id: str,
        attempt_index: int,
        source: ObservationSource,
        modality: ObservationModality,
        score: int,
        difficulty: int,
        citations: tuple[CitationIdentity, ...],
        observed_at: datetime,
    ) -> EvidenceObservation:
        identifier = _observation_digest(
            {
                "attempt_index": attempt_index,
                "fingerprint": fingerprint.value,
                "session_id": session_id,
            }
        )
        return cls(
            identifier,
            profile_id,
            fingerprint,
            fingerprint.document_id,
            objective_id,
            session_id,
            source,
            modality,
            score,
            difficulty,
            citations,
            observed_at,
        )

    @classmethod
    def for_review(
        cls,
        profile_id: str,
        fingerprint: ConceptFingerprint,
        objective_id: str,
        modality: ObservationModality,
        score: int,
        difficulty: int,
        citations: tuple[CitationIdentity, ...],
        observed_at: datetime,
        session_id: str | None = None,
    ) -> EvidenceObservation:
        identifier = _observation_digest(
            {
                "fingerprint": fingerprint.value,
                "modality": modality.value,
                "observed_at": observed_at.isoformat(),
                "profile_id": profile_id,
                "session_id": session_id,
                "source": ObservationSource.REVIEW.value,
            }
        )
        return cls(
            identifier,
            profile_id,
            fingerprint,
            fingerprint.document_id,
            objective_id,
            session_id,
            ObservationSource.REVIEW,
            modality,
            score,
            difficulty,
            citations,
            observed_at,
        )


@dataclass(frozen=True, slots=True)
class ConceptEquivalenceCandidate:
    """A proposed cross-document link that has no effect until accepted."""

    source: ConceptFingerprint
    target: ConceptFingerprint
    similarity: float
    created_at: datetime
    profile_id: str = ""

    def __post_init__(self) -> None:
        self.source.validate_canonical()
        self.target.validate_canonical()
        if self.source.document_id == self.target.document_id:
            raise ValueError("Concepts from one document cannot be linked.")
        if not 0 <= self.similarity <= 1:
            raise ValueError("Concept similarity must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class ConceptEquivalenceLink:
    """A persisted explicit accepted or rejected equivalence decision."""

    source: ConceptFingerprint
    target: ConceptFingerprint
    decision: EquivalenceDecision
    decided_at: datetime
    profile_id: str = ""

    def __post_init__(self) -> None:
        self.source.validate_canonical()
        self.target.validate_canonical()
        if self.source.document_id == self.target.document_id:
            raise ValueError("Concepts from one document cannot be linked.")


def _observation_digest(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _concept_fingerprint_digest(
    algorithm_version: str,
    document_id: DocumentId,
    normalized_title: str,
    normalized_description: str,
) -> str:
    canonical = {
        "algorithm_version": algorithm_version,
        "document_id": document_id.value,
        "normalized_description": normalized_description,
        "normalized_title": normalized_title,
    }
    return hashlib.sha256(
        json.dumps(
            canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
