"""Continue one persistent adaptive study session."""

from scholar_agent.application.dtos.tutor import (
    ContinueStudySessionRequest,
    TutorTurnResult,
)
from scholar_agent.application.output_ports.tutor_runner import ITutorRunner
from scholar_agent.application.services.request_validation_service import (
    RequestValidationService,
)


class ContinueStudySessionUseCase:
    """Validate a learner message and delegate one bounded graph turn."""

    def __init__(
        self,
        tutor_runner: ITutorRunner,
        validation_service: RequestValidationService,
    ) -> None:
        self._tutor_runner = tutor_runner
        self._validation_service = validation_service

    def execute(self, request: ContinueStudySessionRequest) -> TutorTurnResult:
        """Continue a session with one validated learner message."""
        session_id = self._validation_service.validate_text(
            request.session_id, "session_id"
        )
        message = self._validation_service.validate_text(request.message, "message")
        return self._tutor_runner.run(
            ContinueStudySessionRequest(session_id=session_id, message=message)
        )
