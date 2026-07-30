"""API tests for the unified study-agent contract."""

import asyncio
from pathlib import Path

from dependency_injector import providers
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response

from scholar_agent.application.dtos.agent import (
    AskStudyAgentRequest,
    AskStudyAgentResult,
    StudyAgentAnswerResult,
    StudyAgentFlashcardsResult,
    StudyAgentPlanStep,
    StudyAgentQuizResult,
    StudyAgentStatus,
    StudyAgentSummaryResult,
    StudyTask,
)
from scholar_agent.application.dtos.retrieval import RetrievedChunk
from scholar_agent.application.dtos.study_results import Flashcard, QuizQuestion
from scholar_agent.config.settings import Settings
from scholar_agent.domain.value_objects.document_id import DocumentId
from scholar_agent.presentation.api.main import create_app


class FakeAgentRunner:
    """Return every typed result variant through the agent port."""

    def __init__(self) -> None:
        self.requests: list[AskStudyAgentRequest] = []

    def run(self, request: AskStudyAgentRequest) -> AskStudyAgentResult:
        self.requests.append(request)
        citation = RetrievedChunk(
            document_id=request.document_id,
            content="Evidence.",
            page_number=1,
            section=None,
            chunk_id="chunk-1",
            similarity_score=0.9,
        )
        return AskStudyAgentResult(
            status=StudyAgentStatus.COMPLETED,
            plan=(
                StudyAgentPlanStep(StudyTask.ANSWER_QUESTION, "Answer."),
                StudyAgentPlanStep(StudyTask.SUMMARIZE_DOCUMENT, "Summarize."),
                StudyAgentPlanStep(StudyTask.GENERATE_QUIZ, "Quiz."),
                StudyAgentPlanStep(StudyTask.GENERATE_FLASHCARDS, "Flashcards."),
            ),
            results=(
                StudyAgentAnswerResult("Grounded answer.", (citation,)),
                StudyAgentSummaryResult("Summary."),
                StudyAgentQuizResult(
                    (QuizQuestion("Question?", "Answer."),),
                    requested_count=50,
                    effective_count=10,
                    maximum_count=10,
                ),
                StudyAgentFlashcardsResult(
                    (Flashcard("Front", "Back"),),
                    requested_count=10,
                    effective_count=10,
                    maximum_count=20,
                ),
            ),
            notices=(
                "You requested 50 quiz questions; the current limit is 10, so "
                "10 were generated.",
            ),
        )


def test_unified_agent_endpoint_serializes_every_result_variant(
    tmp_path: Path,
) -> None:
    app, runner = _app_with_agent(tmp_path)

    response = asyncio.run(
        _post(
            app,
            "/agent/requests",
            {"prompt": "Prepare me.", "document_id": "document-1"},
        )
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert [item["task"] for item in payload["results"]] == [
        "answer_question",
        "summarize_document",
        "generate_quiz",
        "generate_flashcards",
    ]
    assert payload["results"][0]["citations"][0]["chunk_id"] == "chunk-1"
    assert payload["results"][2]["effective_count"] == 10
    assert runner.requests[0].document_id == DocumentId("document-1")


def test_legacy_agent_endpoint_delegates_and_sets_deprecation_headers(
    tmp_path: Path,
) -> None:
    app, runner = _app_with_agent(tmp_path)

    response = asyncio.run(
        _post(
            app,
            "/agent/study",
            {
                "goal": "Prepare me.",
                "document_ids": ["document-1"],
                "question_count": 7,
                "session_id": "legacy-session",
            },
        )
    )

    assert response.status_code == 200
    assert response.headers["deprecation"] == "true"
    assert response.headers["link"] == '</agent/requests>; rel="successor-version"'
    assert response.json()["results"][3]["task"] == "generate_flashcards"
    assert runner.requests[0].quiz_count_default == 7


def test_legacy_agent_endpoint_rejects_multiple_documents(tmp_path: Path) -> None:
    app, runner = _app_with_agent(tmp_path)

    response = asyncio.run(
        _post(
            app,
            "/agent/study",
            {
                "goal": "Prepare me.",
                "document_ids": ["document-1", "document-2"],
            },
        )
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Exactly one document is required."
    assert runner.requests == []


def test_question_endpoint_requires_one_document_id(tmp_path: Path) -> None:
    app, _ = _app_with_agent(tmp_path)

    response = asyncio.run(
        _post(app, "/questions", {"question": "What does this mean?"})
    )

    assert response.status_code == 422


def test_comparison_endpoint_and_openapi_operation_are_removed(
    tmp_path: Path,
) -> None:
    app, _ = _app_with_agent(tmp_path)

    response = asyncio.run(_post(app, "/comparisons", {}))

    assert response.status_code == 404
    assert "/comparisons" not in app.openapi()["paths"]


def _app_with_agent(tmp_path: Path) -> tuple[FastAPI, FakeAgentRunner]:
    settings = Settings(
        catalog_db_path=tmp_path / "catalog.sqlite3",
        document_library_path=tmp_path / "documents",
        vector_db_path=tmp_path / "vectors",
    )
    app = create_app(settings)
    runner = FakeAgentRunner()
    app.state.container.agent_runner.override(providers.Object(runner))
    return app, runner


async def _post(app: FastAPI, path: str, payload: dict[str, object]) -> Response:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.post(path, json=payload)
