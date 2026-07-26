"""Tests for the goal-oriented study agent."""

import pytest

from scholar_agent.application.dtos.agent import PrepareStudySessionRequest
from scholar_agent.application.output_ports.tool_executor import IToolExecutor
from scholar_agent.application.services.request_validation_service import (
    RequestValidationService,
)
from scholar_agent.application.use_cases.prepare_study_session import (
    PrepareStudySessionUseCase,
)
from scholar_agent.domain.value_objects.document_id import DocumentId
from scholar_agent.infrastructure.adapters.langgraph_agent_runner import (
    LangGraphAgentRunner,
)


class FakeAgentToolExecutor(IToolExecutor):
    """Returns deterministic data for every approved study tool."""

    def __init__(self, failing_tool: str | None = None) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.failing_tool = failing_tool

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, object],
    ) -> dict[str, object]:
        self.calls.append((tool_name, arguments))
        if tool_name == self.failing_tool:
            raise RuntimeError("optional tool unavailable")
        if tool_name == "semantic_search":
            return {
                "chunks": [
                    {
                        "document_id": "document-1",
                        "chunk_id": "chunk-1",
                        "page_number": 2,
                        "similarity_score": 0.9,
                    },
                ],
            }
        if tool_name == "summarize_document":
            return {"summary": "Important concepts from the lecture."}
        if tool_name == "generate_quiz":
            return {
                "questions": [
                    {"prompt": "What is the key idea?", "answer": "Evidence."}
                ],
            }
        return {}


def test_agent_plans_and_executes_multiple_tools_for_one_document() -> None:
    executor = FakeAgentToolExecutor()
    runner = LangGraphAgentRunner(executor)

    result = runner.run(
        {
            "goal": "Prepare me for an exam",
            "document_ids": ["document-1"],
            "question_count": 3,
        },
    )

    assert [call[0] for call in executor.calls] == [
        "semantic_search",
        "summarize_document",
        "generate_quiz",
    ]
    assert result["completed_tools"] == [
        "semantic_search",
        "summarize_document",
        "generate_quiz",
    ]
    assert result["summary"] == "Important concepts from the lecture."
    assert result["quiz"] == [
        {"prompt": "What is the key idea?", "answer": "Evidence."}
    ]


def test_agent_adds_comparison_for_multiple_documents() -> None:
    executor = FakeAgentToolExecutor()
    result = LangGraphAgentRunner(executor).run(
        {
            "goal": "Compare these materials for my exam",
            "document_ids": ["document-1", "document-2"],
            "question_count": 2,
        },
    )

    assert "compare_documents" in result["completed_tools"]
    comparison_call = next(
        arguments for tool, arguments in executor.calls if tool == "compare_documents"
    )
    assert comparison_call == {
        "first_document_id": "document-1",
        "second_document_id": "document-2",
    }


def test_agent_returns_partial_results_when_optional_tool_fails() -> None:
    executor = FakeAgentToolExecutor(failing_tool="compare_documents")
    result = LangGraphAgentRunner(executor).run(
        {
            "goal": "Prepare me for an exam",
            "document_ids": ["document-1", "document-2"],
            "question_count": 2,
        },
    )

    assert (
        result["summary"]
        == (
            "Important concepts from the lecture.\n\n"
            "Important concepts from the lecture."
        )
    )
    assert result["quiz"]
    assert result["errors"] == ["compare_documents: optional tool unavailable"]


def test_use_case_rejects_empty_document_selection() -> None:
    runner = LangGraphAgentRunner(FakeAgentToolExecutor())
    use_case = PrepareStudySessionUseCase(runner, RequestValidationService())

    with pytest.raises(ValueError, match="At least one document"):
        use_case.execute(PrepareStudySessionRequest("Study", ()))


def test_use_case_converts_agent_state_to_typed_result() -> None:
    runner = LangGraphAgentRunner(FakeAgentToolExecutor())
    use_case = PrepareStudySessionUseCase(runner, RequestValidationService())

    result = use_case.execute(
        PrepareStudySessionRequest("Study", (DocumentId("document-1"),), 2)
    )

    assert result.plan[0].tool_name == "semantic_search"
    assert result.citations[0].document_id == DocumentId("document-1")
