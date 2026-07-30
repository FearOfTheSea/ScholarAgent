"""Retrieve one resumable study session."""

from scholar_agent.application.dtos.tutor import StudySessionResult
from scholar_agent.domain.entities.study_session import objective_progress
from scholar_agent.domain.repositories.study_session_repository import (
    StudySessionRepository,
)


class GetStudySessionUseCase:
    """Return persisted adaptive tutoring state."""

    def __init__(self, session_repository: StudySessionRepository) -> None:
        self._session_repository = session_repository

    def execute(self, session_id: str) -> StudySessionResult:
        """Return one session or fail with a stable validation error."""
        session = self._session_repository.get(session_id)
        if session is None:
            raise ValueError(f"Study session '{session_id}' was not found.")
        progress = tuple(
            objective_progress(objective.identifier, session.attempts)
            for objective in session.brief.objectives
        )
        current = min(progress, key=lambda item: (item.percentage, item.attempt_count))
        return StudySessionResult(session, progress, current.objective_id)
