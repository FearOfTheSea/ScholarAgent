"""Integration tests for local infrastructure adapters."""

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from scholar_agent.application.dtos.retrieval import DocumentChunk, LoadedPage
from scholar_agent.application.output_ports.tool_executor import IToolExecutor
from scholar_agent.domain.entities.document import Document
from scholar_agent.domain.value_objects.document_id import DocumentId
from scholar_agent.infrastructure.adapters.faiss_repository import FAISSRepository
from scholar_agent.infrastructure.adapters.langchain_text_chunker import (
    LangChainTextChunker,
)
from scholar_agent.infrastructure.adapters.langgraph_runner import LangGraphRunner
from scholar_agent.infrastructure.adapters.ollama_adapter import OllamaAdapter
from scholar_agent.infrastructure.adapters.sqlite_document_repository import (
    SQLiteDocumentRepository,
)


class RecordingToolExecutor(IToolExecutor):
    """Records the data passed through the thin graph node."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, object],
    ) -> dict[str, object]:
        self.calls.append((tool_name, arguments))
        return {"tool": tool_name, "arguments": arguments}


def test_ollama_adapter_uses_only_the_local_http_api() -> None:
    """Readiness and generation use the supplied localhost-compatible client."""

    generation_requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:1.7b"}]})
        if request.url.path == "/api/generate":
            payload = json.loads(request.content)
            assert isinstance(payload, dict)
            generation_requests.append(payload)
            return httpx.Response(200, json={"response": "Local answer"})
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = OllamaAdapter(
        model_name="qwen3:1.7b",
        base_url="http://localhost:11434",
        context_length=2048,
        maximum_tokens=400,
        client=client,
    )

    assert adapter.is_available() is True
    assert adapter.has_model() is True
    assert adapter.generate("Test prompt") == "Local answer"
    assert generation_requests[0]["think"] is False
    assert generation_requests[0]["options"] == {
        "num_ctx": 2048,
        "num_predict": 400,
        "temperature": 0.2,
    }


def test_faiss_repository_persists_and_deletes_document_chunks(tmp_path: Path) -> None:
    """FAISS search stays aligned with persistent local chunk metadata."""
    document_id = DocumentId("document-1")
    repository = FAISSRepository(tmp_path / "vectors")
    chunks = (
        DocumentChunk(document_id, "first concept", 1, None, "chunk-1", 0),
        DocumentChunk(document_id, "second concept", 2, None, "chunk-2", 1),
    )
    repository.add(chunks, ((1.0, 0.0), (0.0, 1.0)))

    results = repository.search((1.0, 0.0))

    assert results[0].chunk_id == "chunk-1"
    assert results[0].page_number == 1
    repository.delete_document(document_id)
    assert repository.search((1.0, 0.0)) == ()


def test_sqlite_document_repository_persists_catalog_records(tmp_path: Path) -> None:
    """The catalog database starts empty and accepts local document records."""
    repository = SQLiteDocumentRepository(tmp_path / "catalog.sqlite3")
    document = Document(
        identifier=DocumentId("document-1"),
        title="notes",
        source="notes.pdf",
        page_count=2,
        created_at=datetime.now(UTC),
    )

    assert repository.list_all() == ()
    repository.save(document)
    assert repository.get_by_id(document.identifier) == document
    assert repository.delete(document.identifier) is True
    assert repository.list_all() == ()


def test_langchain_chunker_retains_page_and_document_metadata() -> None:
    """The adapter uses LangChain only to create application-owned chunks."""
    document_id = DocumentId("document-1")
    chunker = LangChainTextChunker(chunk_size=24, chunk_overlap=4)

    chunks = chunker.chunk(
        (
            LoadedPage(
                document_id=document_id,
                page_number=3,
                content="One sentence. Two sentence. Three sentence.",
            ),
        ),
    )

    assert len(chunks) > 1
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.document_id == document_id for chunk in chunks)
    assert all(chunk.page_number == 3 for chunk in chunks)


def test_langgraph_runner_delegates_selected_tool_execution() -> None:
    """The graph remains a thin route to the output-port implementation."""
    executor = RecordingToolExecutor()
    runner = LangGraphRunner(executor)

    result = runner.run(
        {
            "tool_name": "citation_lookup",
            "arguments": {"document_id": "document-1", "chunk_id": "chunk-1"},
        },
    )

    assert executor.calls == [
        (
            "citation_lookup",
            {"document_id": "document-1", "chunk_id": "chunk-1"},
        ),
    ]
    assert result["result"] == {
        "tool": "citation_lookup",
        "arguments": {"document_id": "document-1", "chunk_id": "chunk-1"},
    }
