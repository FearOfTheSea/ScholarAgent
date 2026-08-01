"""Advance a bounded persistent study mission."""

from scholar_agent.application.dtos.mission import AdvanceStudyMissionRequest
from scholar_agent.application.dtos.tutor import StudySessionResult
from scholar_agent.application.output_ports.mission_runner import IMissionRunner
from scholar_agent.application.services.request_validation_service import (
    RequestValidationService,
)


class AdvanceStudySessionUseCase:
    """Validate and delegate one mission advance request."""

    def __init__(
        self,
        mission_runner: IMissionRunner,
        validation_service: RequestValidationService,
    ) -> None:
        self._mission_runner = mission_runner
        self._validation_service = validation_service

    def execute(self, request: AdvanceStudyMissionRequest) -> StudySessionResult:
        """Advance with optional learner text; blank means continue automatically."""
        session_id = self._validation_service.validate_text(
            request.session_id, "session_id"
        )
        message = (
            None
            if request.message is None or not request.message.strip()
            else self._validation_service.validate_text(request.message, "message")
        )
        return self._mission_runner.run(
            AdvanceStudyMissionRequest(session_id=session_id, message=message)
        )
