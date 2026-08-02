"""Framework-free DTOs for learner profiles, evidence, and review planning."""

from dataclasses import dataclass
from datetime import date, datetime

from scholar_agent.domain.entities.learner_profile import (
    ConceptEquivalenceCandidate,
    ConceptEquivalenceLink,
    ConceptFingerprint,
    EvidenceObservation,
    LearnerProfile,
    ObservationModality,
)
from scholar_agent.domain.value_objects.citation_identity import CitationIdentity


@dataclass(frozen=True, slots=True)
class KnowledgeEstimate:
    """A deterministic, explainable estimate for one concept group."""

    fingerprint: ConceptFingerprint
    observation_count: int
    recall_count: int
    transfer_count: int
    last_observed_at: datetime | None
    confidence: int
    uncertainty: int
    mastery_label: str
    total_weight: float


@dataclass(frozen=True, slots=True)
class ReviewQueueEntry:
    """A document-bound review recommendation."""

    fingerprint: ConceptFingerprint
    document_id: str
    objective_id: str
    title: str
    description: str
    confidence: int
    uncertainty: int
    observation_count: int
    recall_count: int
    transfer_count: int
    last_observed_at: datetime | None
    due_at: datetime
    expected_minutes: int
    reason_codes: tuple[str, ...]
    source_documents: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CreateLearnerProfileRequest:
    display_name: str
    target_date: date | None = None


@dataclass(frozen=True, slots=True)
class RecordReviewOutcomeRequest:
    profile_id: str
    fingerprint: ConceptFingerprint
    objective_id: str
    modality: ObservationModality
    score: int
    difficulty: int
    citations: tuple[CitationIdentity, ...]
    observed_at: datetime | None = None
    session_id: str | None = None


@dataclass(frozen=True, slots=True)
class LearnerProfileExport:
    """Validated redacted profile export payload."""

    profile: LearnerProfile
    observations: tuple[EvidenceObservation, ...]
    candidates: tuple[ConceptEquivalenceCandidate, ...]
    links: tuple[ConceptEquivalenceLink, ...]
