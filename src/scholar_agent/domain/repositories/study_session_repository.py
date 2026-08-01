"""Repository contract for local adaptive study state."""

from abc import ABC, abstractmethod

from scholar_agent.domain.entities.study_session import (
    DocumentBrief,
    MissionStatus,
    StudySession,
)
from scholar_agent.domain.value_objects.document_id import DocumentId


class StudySessionRepository(ABC):
    """Persists sessions and cached document briefs."""

    @abstractmethod
    def save(self, session: StudySession) -> None:
        """Create or replace a session."""

    @abstractmethod
    def get(self, session_id: str) -> StudySession | None:
        """Return a session when it exists."""

    @abstractmethod
    def list(
        self,
        document_id: DocumentId | None = None,
        status: MissionStatus | None = None,
    ) -> tuple[StudySession, ...]:
        """Return sessions ordered from most recently updated to oldest."""

    @abstractmethod
    def complete(self, session_id: str) -> StudySession | None:
        """Mark a session complete and return its updated state."""

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """Delete a session and report whether it existed."""

    @abstractmethod
    def delete_for_document(self, document_id: DocumentId) -> None:
        """Delete every session and brief for a document."""

    @abstractmethod
    def get_brief(self, document_id: DocumentId) -> DocumentBrief | None:
        """Return a cached document brief."""

    @abstractmethod
    def save_brief(self, brief: DocumentBrief) -> None:
        """Cache a document brief."""
