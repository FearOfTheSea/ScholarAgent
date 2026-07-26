"""Prepare-study-session use case."""

from collections.abc import Mapping

from scholar_agent.application.dtos.agent import (
    PrepareStudySessionRequest,
    PrepareStudySessionResult,
    StudyAgentPlanStep,
)
from scholar_agent.application.dtos.retrieval import RetrievedChunk
from scholar_agent.application.dtos.study_results import QuizQuestion
from scholar_agent.application.output_ports.agent_runner import IAgentRunner
from scholar_agent.application.services.request_validation_service import (
    RequestValidationService,
)
from scholar_agent.domain.value_objects.document_id import DocumentId


class PrepareStudySessionUseCase:
    """Coordinates a goal-oriented study session through an agent port."""

    def __init__(
        self,
        agent_runner: IAgentRunner,
        validation_service: RequestValidationService,
    ) -> None:
        self._agent_runner = agent_runner
        self._validation_service = validation_service

    def execute(self, request: PrepareStudySessionRequest) -> PrepareStudySessionResult:
        """Run and convert the agent workflow into application DTOs."""
        goal = self._validation_service.validate_text(request.goal, "goal")
        question_count = self._validation_service.validate_count(
            request.question_count, "question_count"
        )
        if not request.document_ids:
            raise ValueError("At least one document is required.")
        state = self._agent_runner.run(
            {
                "goal": goal,
                "document_ids": [item.value for item in request.document_ids],
                "question_count": question_count,
                "session_id": request.session_id,
            },
        )
        return PrepareStudySessionResult(
            plan=_plan_steps(state.get("plan", ())),
            summary=_text_value(state.get("summary", "")),
            quiz=_quiz(state.get("quiz", ())),
            recommendations=_text_tuple(state.get("recommendations", ())),
            completed_tools=_text_tuple(state.get("completed_tools", ())),
            citations=_citations(state.get("citations", ())),
            errors=_text_tuple(state.get("errors", ())),
        )


def _plan_steps(value: object) -> tuple[StudyAgentPlanStep, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    steps: list[StudyAgentPlanStep] = []
    for item in value:
        if isinstance(item, Mapping):
            tool_name = item.get("tool_name")
            description = item.get("description")
            if isinstance(tool_name, str) and isinstance(description, str):
                steps.append(StudyAgentPlanStep(tool_name, description))
    return tuple(steps)


def _quiz(value: object) -> tuple[QuizQuestion, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    questions: list[QuizQuestion] = []
    for item in value:
        if isinstance(item, Mapping):
            prompt = item.get("prompt")
            answer = item.get("answer")
            if isinstance(prompt, str) and isinstance(answer, str):
                questions.append(QuizQuestion(prompt, answer))
    return tuple(questions)


def _citations(value: object) -> tuple[RetrievedChunk, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    citations: list[RetrievedChunk] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        document_id = item.get("document_id")
        chunk_id = item.get("chunk_id")
        page_number = item.get("page_number")
        section = item.get("section")
        score = item.get("similarity_score")
        if (
            isinstance(document_id, str)
            and isinstance(chunk_id, str)
            and (isinstance(page_number, int) or page_number is None)
            and (isinstance(section, str) or section is None)
            and isinstance(score, (int, float))
        ):
            citations.append(
                RetrievedChunk(
                    DocumentId(document_id),
                    "",
                    page_number,
                    section,
                    chunk_id,
                    float(score),
                ),
            )
    return tuple(citations)


def _text_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _text_value(value: object) -> str:
    return value if isinstance(value, str) else ""
