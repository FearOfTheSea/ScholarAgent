"""Return deterministic intelligence for one persisted Study Mission."""

from scholar_agent.application.dtos.mission_intelligence import MissionInsights
from scholar_agent.application.services.mission_insights import MissionInsightsService
from scholar_agent.domain.repositories.study_session_repository import (
    StudySessionRepository,
)


class GetMissionInsightsUseCase:
    """Calculate local learning signals without model execution."""

    def __init__(
        self,
        session_repository: StudySessionRepository,
        insights_service: MissionInsightsService | None = None,
    ) -> None:
        self._session_repository = session_repository
        self._insights = insights_service or MissionInsightsService()

    def execute(self, session_id: str) -> MissionInsights:
        session = self._session_repository.get(session_id)
        if session is None:
            raise ValueError(f"Study session '{session_id}' was not found.")
        return self._insights.calculate(session)
