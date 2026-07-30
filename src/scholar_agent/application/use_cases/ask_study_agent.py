"""Unified ask-study-agent use case."""

from scholar_agent.application.dtos.agent import (
    AskStudyAgentRequest,
    AskStudyAgentResult,
)
from scholar_agent.application.output_ports.agent_runner import IAgentRunner
from scholar_agent.application.services.request_validation_service import (
    RequestValidationService,
)


class AskStudyAgentUseCase:
    """Validate and delegate one free-form study request to the agent port."""

    def __init__(
        self,
        agent_runner: IAgentRunner,
        validation_service: RequestValidationService,
    ) -> None:
        self._agent_runner = agent_runner
        self._validation_service = validation_service

    def execute(self, request: AskStudyAgentRequest) -> AskStudyAgentResult:
        """Run one validated, single-document study request."""
        prompt = self._validation_service.validate_text(request.prompt, "prompt")
        quiz_count_default = self._validation_service.validate_count(
            request.quiz_count_default,
            "quiz_count_default",
        )
        return self._agent_runner.run(
            AskStudyAgentRequest(
                prompt=prompt,
                document_id=request.document_id,
                quiz_count_default=quiz_count_default,
            )
        )
