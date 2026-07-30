"""Port for the adaptive tutoring graph."""

from abc import ABC, abstractmethod

from scholar_agent.application.dtos.tutor import (
    ContinueStudySessionRequest,
    TutorTurnResult,
)


class ITutorRunner(ABC):
    """Runs one bounded adaptive tutor turn."""

    @abstractmethod
    def run(self, request: ContinueStudySessionRequest) -> TutorTurnResult:
        """Classify, execute, verify, and persist one turn."""
