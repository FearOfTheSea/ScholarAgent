"""Tests for local study use cases."""

from scholar_agent.application.dtos.retrieval import RetrievedChunk
from scholar_agent.application.dtos.study_requests import AnswerQuestionRequest
from scholar_agent.application.output_ports.llm_provider import ILLMProvider
from scholar_agent.application.output_ports.retriever import IRetriever
from scholar_agent.application.services.request_validation_service import (
    RequestValidationService,
)
from scholar_agent.application.use_cases.answer_question import AnswerQuestionUseCase
from scholar_agent.domain.value_objects.document_id import DocumentId


class FakeLLMProvider(ILLMProvider):
    """Minimal deterministic local-model substitute for unit tests."""

    def generate(self, prompt: str) -> str:
        return "Grounded response [document-1:p1:chunk-1]"

    def is_available(self) -> bool:
        return True

    def has_model(self) -> bool:
        return True


class FakeRetriever(IRetriever):
    """Returns a fixed evidence chunk."""

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        document_ids: tuple[DocumentId, ...] = (),
    ) -> tuple[RetrievedChunk, ...]:
        return (
            RetrievedChunk(
                document_id=DocumentId("document-1"),
                content="ScholarAgent uses local source excerpts.",
                page_number=1,
                section=None,
                chunk_id="chunk-1",
                similarity_score=0.91,
            ),
        )


def test_answer_question_returns_grounded_answer_and_citations() -> None:
    """Question answering returns the exact evidence selected by retrieval."""
    use_case = AnswerQuestionUseCase(
        llm_provider=FakeLLMProvider(),
        retriever=FakeRetriever(),
        validation_service=RequestValidationService(),
    )

    result = use_case.execute(AnswerQuestionRequest(question="How does it work?"))

    assert result.answer == "Grounded response [document-1:p1:chunk-1]"
    assert result.citations[0].chunk_id == "chunk-1"
