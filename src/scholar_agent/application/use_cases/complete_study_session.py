"""Manually complete one persistent study mission."""

from scholar_agent.application.dtos.tutor import StudySessionResult
from scholar_agent.application.services.mission_state import MissionStateService
from scholar_agent.domain.entities.study_session import (
    objective_progress,
)
from scholar_agent.domain.repositories.study_session_repository import (
    StudySessionRepository,
)


class CompleteStudySessionUseCase:
    """Complete a session while preserving its artifacts and trace."""

    def __init__(
        self,
        session_repository: StudySessionRepository,
        state_service: MissionStateService | None = None,
    ) -> None:
        self._session_repository = session_repository
        self._state = state_service or MissionStateService(session_repository)

    def execute(self, session_id: str) -> StudySessionResult:
        session = self._session_repository.get(session_id)
        if session is None:
            raise ValueError(f"Study session '{session_id}' was not found.")
        completed = self._state.complete(
            session, "Learner manually completed the mission."
        )
        progress = tuple(
            objective_progress(item.identifier, completed.attempts)
            for item in completed.brief.objectives
        )
        current = progress[0].objective_id if progress else None
        return StudySessionResult(
            completed, progress, current, None, completed.trace[-1:]
        )
