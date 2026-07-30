"""Tests for the FastAPI health endpoint."""

import asyncio
from pathlib import Path

import pytest
from dependency_injector import providers
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response

from scholar_agent.application.dtos.retrieval import RetrievedChunk
from scholar_agent.application.output_ports.retriever import IRetriever
from scholar_agent.config.settings import Settings
from scholar_agent.domain.value_objects.document_id import DocumentId
from scholar_agent.presentation.api.main import create_app


class UnavailableLocalLLM:
    """Deterministic local-runtime substitute for readiness testing."""

    def generate(self, prompt: str) -> str:
        raise AssertionError("Readiness must not generate text.")

    def is_available(self) -> bool:
        return False

    def has_model(self) -> bool:
        return False


class EvidenceRetriever(IRetriever):
    """Supplies evidence so a question request reaches the LLM port."""

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        document_ids: tuple[DocumentId, ...] = (),
    ) -> tuple[RetrievedChunk, ...]:
        return (
            RetrievedChunk(
                document_id=DocumentId("document-1"),
                content="Local evidence.",
                page_number=1,
                section=None,
                chunk_id="chunk-1",
                similarity_score=1.0,
            ),
        )


class GenerationUnavailableLLM(UnavailableLocalLLM):
    """Models an Ollama process that cannot complete a generation request."""

    def generate(self, prompt: str) -> str:
        raise RuntimeError("The local Ollama service is unavailable.")


def test_health_endpoint_returns_service_status() -> None:
    """The API can start without any AI capability configured."""
    app = create_app(Settings(debug=False))
    response = asyncio.run(_get_response(app, "/health"))

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "scholar-agent",
        "version": "0.1.0",
    }


def test_readiness_reports_unavailable_without_a_local_runtime(tmp_path: Path) -> None:
    """The readiness route is informative when Ollama is not available."""
    settings = Settings(
        catalog_db_path=tmp_path / "catalog.sqlite3",
        document_library_path=tmp_path / "documents",
        vector_db_path=tmp_path / "vectors",
    )
    app = create_app(settings)
    app.state.container.llm_provider.override(providers.Object(UnavailableLocalLLM()))

    response = asyncio.run(_get_response(app, "/ready"))

    assert response.status_code == 200
    assert response.json() == {
        "status": "unavailable",
        "ollama_available": False,
        "model_available": False,
    }


@pytest.mark.parametrize(
    ("filename", "content", "expected_detail"),
    (
        ("empty.pdf", b"", "The uploaded PDF is empty."),
        ("notes.txt", b"%PDF local", "Only PDF files are supported."),
        ("invalid.pdf", b"not a pdf", "The uploaded file is not a valid PDF."),
    ),
)
def test_document_upload_rejects_invalid_content_before_model_work(
    tmp_path: Path,
    filename: str,
    content: bytes,
    expected_detail: str,
) -> None:
    """Invalid input never reaches lazy embedding or model providers."""
    settings = Settings(
        catalog_db_path=tmp_path / "catalog.sqlite3",
        document_library_path=tmp_path / "documents",
        vector_db_path=tmp_path / "vectors",
    )
    app = create_app(settings)
    response = asyncio.run(_post_document(app, filename, content))

    assert response.status_code == 400
    assert response.json()["detail"] == expected_detail


def test_document_upload_rejects_a_malformed_pdf_before_embedding_work(
    tmp_path: Path,
) -> None:
    """A PDF header alone is insufficient for PyMuPDF extraction."""
    settings = Settings(
        catalog_db_path=tmp_path / "catalog.sqlite3",
        document_library_path=tmp_path / "documents",
        vector_db_path=tmp_path / "vectors",
    )
    app = create_app(settings)

    response = asyncio.run(_post_document(app, "broken.pdf", b"%PDF-1.7\nbroken"))

    assert response.status_code == 400
    assert response.json()["detail"].startswith("Unable to extract text from")


def test_question_returns_unavailable_when_the_local_model_cannot_generate(
    tmp_path: Path,
) -> None:
    """Generation requests expose local-model failures as service unavailability."""
    settings = Settings(
        catalog_db_path=tmp_path / "catalog.sqlite3",
        document_library_path=tmp_path / "documents",
        vector_db_path=tmp_path / "vectors",
    )
    app = create_app(settings)
    app.state.container.llm_provider.override(
        providers.Object(GenerationUnavailableLLM())
    )
    app.state.container.retriever.override(providers.Object(EvidenceRetriever()))

    response = asyncio.run(_post_question(app))

    assert response.status_code == 503
    assert response.json()["detail"] == "The local Ollama service is unavailable."


async def _get_response(app: FastAPI, path: str) -> Response:
    """Call the ASGI application without starting a network server."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.get(path)


async def _post_document(app: FastAPI, filename: str, content: bytes) -> Response:
    """Send one file payload to the local document endpoint."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.post(
            "/documents",
            files={"file": (filename, content, "application/pdf")},
        )


async def _post_question(app: FastAPI) -> Response:
    """Send a question that has local evidence but an unavailable model."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.post(
            "/questions",
            json={
                "question": "What does the material say?",
                "document_id": "document-1",
            },
        )
