"""Port for bounded persistent mission advancement."""

from abc import ABC, abstractmethod

from scholar_agent.application.dtos.mission import AdvanceStudyMissionRequest
from scholar_agent.application.dtos.tutor import StudySessionResult


class IMissionRunner(ABC):
    """Advances one persistent single-document mission."""

    @abstractmethod
    def run(self, request: AdvanceStudyMissionRequest) -> StudySessionResult:
        """Advance or wait on a mission and persist every state change."""
