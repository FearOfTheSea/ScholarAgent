"""Opt-in real-model E2E test for the adaptive single-document tutor."""

import asyncio
import os
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from scholar_agent.config.settings import Settings
from scholar_agent.presentation.api.main import create_app

RUN_LOCAL_E2E = os.environ.get("RUN_LOCAL_E2E") == "1"
E2E_LLM_PROVIDER = os.environ.get("E2E_LLM_PROVIDER", "ollama")


@pytest.mark.local_runtime
@pytest.mark.skipif(not RUN_LOCAL_E2E, reason="Set RUN_LOCAL_E2E=1 to run local E2E.")
def test_real_pdf_adaptive_tutor_journey(tmp_path: Path) -> None:
    """Ingest, tutor, resume, and delete using real local adapters."""
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
        timeout=300,
    ) as client:
        with fixture.open("rb") as source:
            upload = await client.post(
                "/documents",
                files={"file": (fixture.name, source, "application/pdf")},
            )
        assert upload.status_code == 201, upload.text
        document_id = upload.json()["id"]

        started = await client.post(
            "/agent/sessions",
            json={
                "document_id": document_id,
                "goal": "Understand linear regression well enough for an exam.",
                "learner_level": "intermediate",
                "mode": "exam",
                "target_minutes": 30,
            },
        )
        assert started.status_code == 201, started.text
        start_payload = started.json()
        session_id = start_payload["session_id"]
        assert start_payload["brief"]["objectives"]
        assert start_payload["brief"]["concepts"]
        _assert_single_document_sources(start_payload, document_id)

        explained = await client.post(
            f"/agent/sessions/{session_id}/turns",
            json={"message": "Explain the purpose of the loss function."},
        )
        assert explained.status_code == 200, explained.text
        explanation = explained.json()
        assert explanation["intent"] == "explain"
        assert explanation["activity"]["kind"] == "explanation"
        assert explanation["activity"]["citations"]
        _assert_single_document_sources(explanation, document_id)

        assessed = await client.post(
            f"/agent/sessions/{session_id}/turns",
            json={
                "message": (
                    "The loss function measures prediction error, and training "
                    "chooses parameters that minimize that error."
                )
            },
        )
        assert assessed.status_code == 200, assessed.text
        assessment = assessed.json()
        assert assessment["intent"] == "answer"
        assert assessment["assessment"]["score"] in {0, 1, 2, 3}
        assert assessment["progress"][0]["attempt_count"] >= 1
        _assert_single_document_sources(assessment, document_id)

        resumed = await client.get(f"/agent/sessions/{session_id}")
        assert resumed.status_code == 200, resumed.text
        assert len(resumed.json()["turns"]) == 2

        deleted = await client.delete(f"/documents/{document_id}")
        assert deleted.status_code == 204, deleted.text
        missing_session = await client.get(f"/agent/sessions/{session_id}")
        assert missing_session.status_code == 404


def _assert_single_document_sources(payload: object, document_id: str) -> None:
    """Recursively reject evidence from any other document."""
    if isinstance(payload, dict):
        cited_document = payload.get("document_id")
        if "chunk_id" in payload and cited_document is not None:
            assert cited_document == document_id
        for value in payload.values():
            _assert_single_document_sources(value, document_id)
    elif isinstance(payload, list):
        for value in payload:
            _assert_single_document_sources(value, document_id)
