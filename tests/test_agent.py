"""Tests for the unified, constrained study agent."""

import json
from collections.abc import Mapping

import pytest

from scholar_agent.application.dtos.agent import (
    AskStudyAgentRequest,
    StudyAgentAnswerResult,
    StudyAgentFlashcardsResult,
    StudyAgentQuizResult,
    StudyAgentStatus,
    StudyAgentSummaryResult,
    StudyTask,
)
from scholar_agent.application.output_ports.llm_provider import ILLMProvider
from scholar_agent.application.output_ports.tool_executor import (
    IToolExecutor,
    StudyToolDefinition,
)
from scholar_agent.application.services.request_validation_service import (
    RequestValidationService,
)
from scholar_agent.application.use_cases.ask_study_agent import AskStudyAgentUseCase
from scholar_agent.domain.value_objects.document_id import DocumentId
from scholar_agent.infrastructure.adapters.langgraph_agent_runner import (
    LangGraphAgentRunner,
)
from scholar_agent.infrastructure.tools.capabilities import STUDY_CAPABILITIES


class FakePlannerLLM(ILLMProvider):
    """Return queued planner outputs without external inference."""

    def __init__(self, *outputs: str) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.outputs:
            raise AssertionError("The planner made an unexpected LLM call.")
        return self.outputs.pop(0)

    def is_available(self) -> bool:
        return True

    def has_model(self) -> bool:
        return True


class FakeAgentToolExecutor(IToolExecutor):
    """Return deterministic data for every registered study capability."""

    def __init__(self, failing_tool: str | None = None) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.failing_tool = failing_tool

    def capabilities(self) -> tuple[StudyToolDefinition, ...]:
        return STUDY_CAPABILITIES

    def execute(
        self,
        tool_name: str,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]:
        copied_arguments = dict(arguments)
        self.calls.append((tool_name, copied_arguments))
        if tool_name == self.failing_tool:
            raise RuntimeError("tool unavailable")
        if tool_name == StudyTask.ANSWER_QUESTION.value:
            return {
                "answer": "Grounded answer.",
                "citations": [
                    {
                        "document_id": "document-1",
                        "content": "Evidence.",
                        "chunk_id": "chunk-1",
                        "page_number": 2,
                        "section": None,
                        "similarity_score": 0.9,
                    }
                ],
            }
        if tool_name == StudyTask.SUMMARIZE_DOCUMENT.value:
            return {"summary": "Important concepts from the lecture."}
        if tool_name == StudyTask.GENERATE_QUIZ.value:
            requested = _integer_argument(arguments, "question_count")
            effective = min(requested, 10)
            return {
                "questions": [
                    {"prompt": "What is the key idea?", "answer": "Evidence."}
                ],
                "requested_count": requested,
                "effective_count": effective,
                "maximum_count": 10,
                "notice": (
                    f"You requested {requested} quiz questions; the current limit "
                    f"is 10, so {effective} were generated."
                    if requested > 10
                    else None
                ),
            }
        if tool_name == StudyTask.GENERATE_FLASHCARDS.value:
            requested = _integer_argument(arguments, "card_count")
            effective = min(requested, 20)
            return {
                "cards": [{"front": "Concept", "back": "Definition"}],
                "requested_count": requested,
                "effective_count": effective,
                "maximum_count": 20,
                "notice": (
                    f"You requested {requested} flashcards; the current limit is "
                    f"20, so {effective} were generated."
                    if requested > 20
                    else None
                ),
            }
        raise AssertionError(f"Unexpected tool: {tool_name}")


def _plan(*actions: dict[str, object], message: str | None = None) -> str:
    return json.dumps({"actions": list(actions), "message": message})


def test_agent_executes_only_answer_question_for_a_question() -> None:
    executor = FakeAgentToolExecutor()
    runner = _runner(
        executor,
        _plan(
            {
                "tool_name": "answer_question",
                "arguments": {"question": "What is gradient descent?"},
            }
        ),
    )

    result = runner.run(_request("What is gradient descent?"))

    assert [call[0] for call in executor.calls] == ["answer_question"]
    assert executor.calls[0][1]["document_id"] == "document-1"
    assert isinstance(result.results[0], StudyAgentAnswerResult)
    assert result.status is StudyAgentStatus.COMPLETED


def test_agent_executes_every_explicit_task_once_in_planner_order() -> None:
    executor = FakeAgentToolExecutor()
    runner = _runner(
        executor,
        _plan(
            {"tool_name": "summarize_document", "arguments": {}},
            {
                "tool_name": "generate_quiz",
                "arguments": {"question_count": 3},
            },
        ),
    )

    result = runner.run(_request("Summarize this and make a three-question quiz."))

    assert [call[0] for call in executor.calls] == [
        "summarize_document",
        "generate_quiz",
    ]
    assert isinstance(result.results[0], StudyAgentSummaryResult)
    assert isinstance(result.results[1], StudyAgentQuizResult)
    assert result.results[1].requested_count == 3


def test_agent_allows_an_inferred_study_bundle() -> None:
    executor = FakeAgentToolExecutor()
    runner = _runner(
        executor,
        _plan(
            {"tool_name": "summarize_document", "arguments": {}},
            {"tool_name": "generate_flashcards", "arguments": {}},
            {"tool_name": "generate_quiz", "arguments": {}},
        ),
    )

    result = runner.run(_request("Prepare me for my exam."))

    assert [step.task for step in result.plan] == [
        StudyTask.SUMMARIZE_DOCUMENT,
        StudyTask.GENERATE_FLASHCARDS,
        StudyTask.GENERATE_QUIZ,
    ]
    assert isinstance(result.results[1], StudyAgentFlashcardsResult)
    assert executor.calls[1][1]["card_count"] == 10
    assert executor.calls[2][1]["question_count"] == 5


def test_agent_returns_guidance_without_execution_for_comparison() -> None:
    executor = FakeAgentToolExecutor()
    runner = _runner(
        executor,
        _plan(message="Comparison is not currently supported."),
    )

    result = runner.run(_request("Compare this with my other PDF."))

    assert executor.calls == []
    assert result.status is StudyAgentStatus.NEEDS_CLARIFICATION
    assert result.message == "Comparison is not currently supported."


@pytest.mark.parametrize(
    "invalid_plan",
    [
        "not json",
        _plan(
            {"tool_name": "summarize_document", "arguments": {}},
            {"tool_name": "summarize_document", "arguments": {}},
        ),
        _plan(
            {"tool_name": "summarize_document", "arguments": {}},
            {
                "tool_name": "generate_quiz",
                "arguments": {"question_count": 0},
            },
        ),
        _plan({"tool_name": "compare_documents", "arguments": {}}),
    ],
)
def test_invalid_plan_is_retried_once_and_executes_nothing(
    invalid_plan: str,
) -> None:
    executor = FakeAgentToolExecutor()
    llm = FakePlannerLLM(invalid_plan, invalid_plan)
    result = LangGraphAgentRunner(executor, llm).run(_request("Do something."))

    assert len(llm.prompts) == 2
    assert executor.calls == []
    assert result.status is StudyAgentStatus.FAILED
    assert "invalid study plan" in (result.message or "")


def test_agent_continues_independent_tasks_after_runtime_failure() -> None:
    executor = FakeAgentToolExecutor(failing_tool="summarize_document")
    runner = _runner(
        executor,
        _plan(
            {"tool_name": "summarize_document", "arguments": {}},
            {"tool_name": "generate_quiz", "arguments": {}},
        ),
    )

    result = runner.run(_request("Summarize and quiz me."))

    assert [call[0] for call in executor.calls] == [
        "summarize_document",
        "generate_quiz",
    ]
    assert len(result.results) == 1
    assert isinstance(result.results[0], StudyAgentQuizResult)
    assert result.errors[0].task is StudyTask.SUMMARIZE_DOCUMENT
    assert result.status is StudyAgentStatus.PARTIAL


def test_agent_preserves_requested_counts_and_collects_cap_notices() -> None:
    executor = FakeAgentToolExecutor()
    runner = _runner(
        executor,
        _plan(
            {
                "tool_name": "generate_quiz",
                "arguments": {"question_count": 50},
            },
            {
                "tool_name": "generate_flashcards",
                "arguments": {"card_count": 50},
            },
        ),
    )

    result = runner.run(_request("Create 50 quiz questions and 50 flashcards."))

    quiz = result.results[0]
    flashcards = result.results[1]
    assert isinstance(quiz, StudyAgentQuizResult)
    assert isinstance(flashcards, StudyAgentFlashcardsResult)
    assert quiz.effective_count == 10
    assert flashcards.effective_count == 20
    assert len(result.notices) == 2


def test_use_case_rejects_a_blank_prompt_before_planning() -> None:
    llm = FakePlannerLLM(_plan(message="unused"))
    use_case = AskStudyAgentUseCase(
        LangGraphAgentRunner(FakeAgentToolExecutor(), llm),
        RequestValidationService(),
    )

    with pytest.raises(ValueError, match="prompt"):
        use_case.execute(_request("   "))

    assert llm.prompts == []


def _runner(
    executor: FakeAgentToolExecutor,
    *planner_outputs: str,
) -> LangGraphAgentRunner:
    return LangGraphAgentRunner(executor, FakePlannerLLM(*planner_outputs))


def _request(prompt: str) -> AskStudyAgentRequest:
    return AskStudyAgentRequest(prompt, DocumentId("document-1"))


def _integer_argument(arguments: Mapping[str, object], key: str) -> int:
    value = arguments[key]
    assert isinstance(value, int)
    return value
