"""Port for private local learner profiles and redacted evidence."""

from abc import ABC, abstractmethod
from datetime import datetime

from scholar_agent.domain.entities.learner_profile import (
    ConceptEquivalenceCandidate,
    ConceptEquivalenceLink,
    EvidenceObservation,
    LearnerProfile,
)


class LearnerProfileRepository(ABC):
    """Persist profiles independently from mission history and source files."""

    @abstractmethod
    def save_profile(self, profile: LearnerProfile) -> None:
        """Create or replace profile metadata."""

    @abstractmethod
    def get_profile(self, profile_id: str) -> LearnerProfile | None:
        """Return one profile when it exists."""

    @abstractmethod
    def list_profiles(self) -> tuple[LearnerProfile, ...]:
        """Return profiles in deterministic display order."""

    @abstractmethod
    def get_or_create_default(self, now: datetime) -> LearnerProfile:
        """Lazily create and return the stable local-default profile."""

    @abstractmethod
    def delete_profile(self, profile_id: str) -> int:
        """Delete all profile data and return detached mission count."""

    @abstractmethod
    def append_observation(self, observation: EvidenceObservation) -> bool:
        """Append once by deterministic observation id; return whether inserted."""

    @abstractmethod
    def list_observations(self, profile_id: str) -> tuple[EvidenceObservation, ...]:
        """Return observations ordered by observed time and id."""

    @abstractmethod
    def save_candidate(self, candidate: ConceptEquivalenceCandidate) -> None:
        """Store or replace one proposed equivalence candidate."""

    @abstractmethod
    def list_candidates(
        self, profile_id: str
    ) -> tuple[ConceptEquivalenceCandidate, ...]:
        """Return candidates involving concepts observed by the profile."""

    @abstractmethod
    def save_equivalence_link(self, link: ConceptEquivalenceLink) -> None:
        """Store an explicit accepted or rejected link decision."""

    @abstractmethod
    def list_equivalence_links(
        self, profile_id: str
    ) -> tuple[ConceptEquivalenceLink, ...]:
        """Return decisions involving concepts observed by the profile."""

    @abstractmethod
    def replace_profile_data(
        self,
        profile: LearnerProfile,
        observations: tuple[EvidenceObservation, ...],
        candidates: tuple[ConceptEquivalenceCandidate, ...],
        links: tuple[ConceptEquivalenceLink, ...],
    ) -> None:
        """Atomically replace all rows belonging to one profile."""

    @abstractmethod
    def close(self) -> None:
        """Close local profile storage."""
