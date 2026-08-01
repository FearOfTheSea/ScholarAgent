"""List persistent study missions with additive filters."""

from scholar_agent.domain.entities.study_session import MissionStatus, StudySession
from scholar_agent.domain.repositories.study_session_repository import (
    StudySessionRepository,
)
from scholar_agent.domain.value_objects.document_id import DocumentId


class ListStudySessionsUseCase:
    """Return locally persisted missions ordered by most recent update."""

    def __init__(self, session_repository: StudySessionRepository) -> None:
        self._session_repository = session_repository

    def execute(
        self,
        document_id: DocumentId | None = None,
        status: MissionStatus | None = None,
    ) -> tuple[StudySession, ...]:
        return self._session_repository.list(document_id=document_id, status=status)
