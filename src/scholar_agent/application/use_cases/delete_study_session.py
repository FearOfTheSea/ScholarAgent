"""Delete one local adaptive study session."""

from scholar_agent.application.dtos.tutor import DeleteStudySessionResult
from scholar_agent.domain.repositories.study_session_repository import (
    StudySessionRepository,
)


class DeleteStudySessionUseCase:
    """Remove local tutoring state without deleting its document."""

    def __init__(self, session_repository: StudySessionRepository) -> None:
        self._session_repository = session_repository

    def execute(self, session_id: str) -> DeleteStudySessionResult:
        """Delete a session and report whether it existed."""
        return DeleteStudySessionResult(
            session_id=session_id,
            deleted=self._session_repository.delete(session_id),
        )
