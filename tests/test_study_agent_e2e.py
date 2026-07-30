"""Opt-in real-model E2E test covering every unified Study Agent capability."""

import asyncio
import os
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from scholar_agent.config.settings import Settings
from scholar_agent.infrastructure.tools.capabilities import STUDY_CAPABILITIES
from scholar_agent.presentation.api.main import create_app

RUN_LOCAL_E2E = os.environ.get("RUN_LOCAL_E2E") == "1"
E2E_LLM_PROVIDER = os.environ.get("E2E_LLM_PROVIDER", "ollama")
EXPECTED_CAPABILITIES = {
    "answer_question",
    "summarize_document",
    "generate_quiz",
    "generate_flashcards",
}


@pytest.mark.local_runtime
@pytest.mark.skipif(not RUN_LOCAL_E2E, reason="Set RUN_LOCAL_E2E=1 to run local E2E.")
def test_real_pdf_study_agent_all_capabilities(tmp_path: Path) -> None:
    """Ingest one PDF and execute every registered agent capability."""
    assert {
        definition.task.value for definition in STUDY_CAPABILITIES
    } == EXPECTED_CAPABILITIES
    fixture = (
        Path(__file__).parent
        / "fixtures"
        / "evaluation"
        / "lecture_03_linear_regression.pdf"
    )
    settings = Settings(
        llm_provider_type=E2E_LLM_PROVIDER,
        model_name="qwen3:1.7b",
        scratch_gpt_checkpoint_path=Path("data/scholar_gpt.pt"),
        catalog_db_path=tmp_path / "catalog.sqlite3",
        document_library_path=tmp_path / "documents",
        vector_db_path=tmp_path / "vectors",
    )
    app = create_app(settings)

    asyncio.run(_exercise_journey(app, fixture))


async def _exercise_journey(app: object, fixture: Path) -> None:
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
        timeout=600,
    ) as client:
        with fixture.open("rb") as source:
            upload = await client.post(
                "/documents",
                files={"file": (fixture.name, source, "application/pdf")},
            )
        assert upload.status_code == 201, upload.text
        document_id = upload.json()["id"]

        response = await client.post(
            "/agent/requests",
            json={
                "document_id": document_id,
                "prompt": (
                    "Use every requested capability. Answer this question: What does "
                    "the cost function measure? Then summarize the document, generate "
                    "exactly 2 quiz questions, and generate exactly 3 flashcards."
                ),
            },
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["status"] == "completed", payload
        assert payload["errors"] == []
        assert {step["task"] for step in payload["plan"]} == EXPECTED_CAPABILITIES
        results = {item["task"]: item for item in payload["results"]}
        assert set(results) == EXPECTED_CAPABILITIES

        answer = results["answer_question"]
        assert answer["answer"].strip()
        assert answer["citations"]
        assert all(
            citation["document_id"] == document_id for citation in answer["citations"]
        )

        assert results["summarize_document"]["summary"].strip()

        quiz = results["generate_quiz"]
        assert quiz["requested_count"] == 2
        assert quiz["generated_count"] == 2
        assert all(item["prompt"] and item["answer"] for item in quiz["questions"])

        flashcards = results["generate_flashcards"]
        assert flashcards["requested_count"] == 3
        assert flashcards["generated_count"] == 3
        assert all(item["front"] and item["back"] for item in flashcards["cards"])

        deleted = await client.delete(f"/documents/{document_id}")
        assert deleted.status_code == 204, deleted.text
