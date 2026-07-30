"""Tests for dependency-injection composition."""

from pathlib import Path

from scholar_agent.application.dtos.agent import StudyTask
from scholar_agent.application.use_cases.ask_study_agent import AskStudyAgentUseCase
from scholar_agent.config.settings import Settings
from scholar_agent.infrastructure.adapters import (
    FAISSRepository,
    InMemoryStore,
    LangChainRetriever,
    LangChainTextChunker,
    LangGraphRunner,
    LocalDocumentLibrary,
    OllamaAdapter,
    PyMuPDFLoader,
    SentenceTransformerEmbedding,
    SQLiteDocumentRepository,
)
from scholar_agent.infrastructure.di import build_container
from scholar_agent.infrastructure.tools import StudyToolExecutor
from scholar_agent.infrastructure.tools.answer_question_tool import AnswerQuestionTool


def test_container_registers_every_output_port_implementation(tmp_path: Path) -> None:
    """Every output port has a concrete local implementation."""
    settings = Settings(
        debug=True,
        vector_db_path=tmp_path / "vectors",
        catalog_db_path=tmp_path / "catalog.sqlite3",
        document_library_path=tmp_path / "documents",
    )
    container = build_container(settings)

    assert isinstance(container.llm_provider(), OllamaAdapter)
    assert isinstance(container.retriever(), LangChainRetriever)
    assert isinstance(container.vector_store(), FAISSRepository)
    assert isinstance(container.embedding_provider(), SentenceTransformerEmbedding)
    assert isinstance(container.pdf_loader(), PyMuPDFLoader)
    assert isinstance(container.text_chunker(), LangChainTextChunker)
    assert isinstance(container.document_library(), LocalDocumentLibrary)
    assert isinstance(container.document_repository(), SQLiteDocumentRepository)
    assert isinstance(container.memory_store(), InMemoryStore)
    assert isinstance(container.graph_runner(), LangGraphRunner)
    assert isinstance(container.tool_executor(), StudyToolExecutor)
    assert isinstance(container.answer_question_tool(), AnswerQuestionTool)
    assert isinstance(container.ask_study_agent_use_case(), AskStudyAgentUseCase)
    assert [item.task for item in container.tool_executor().capabilities()] == [
        StudyTask.ANSWER_QUESTION,
        StudyTask.SUMMARIZE_DOCUMENT,
        StudyTask.GENERATE_QUIZ,
        StudyTask.GENERATE_FLASHCARDS,
    ]
