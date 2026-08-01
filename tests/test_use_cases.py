"""Tests for local study use cases."""

import json

from scholar_agent.application.dtos.retrieval import DocumentChunk, RetrievedChunk
from scholar_agent.application.dtos.study_requests import (
    AnswerQuestionRequest,
    GenerateFlashcardsRequest,
    GenerateQuizRequest,
)
from scholar_agent.application.output_ports.llm_provider import ILLMProvider
from scholar_agent.application.output_ports.retriever import IRetriever
from scholar_agent.application.services.generation_count_policy import (
    GenerationCountPolicy,
)
from scholar_agent.application.services.request_validation_service import (
    RequestValidationService,
)
from scholar_agent.application.use_cases.answer_question import AnswerQuestionUseCase
from scholar_agent.application.use_cases.generate_flashcards import (
    GenerateFlashcardsUseCase,
)
from scholar_agent.application.use_cases.generate_quiz import GenerateQuizUseCase
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

    def __init__(self) -> None:
        self.document_ids: tuple[DocumentId, ...] = ()

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        document_ids: tuple[DocumentId, ...] = (),
    ) -> tuple[RetrievedChunk, ...]:
        self.document_ids = document_ids
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
    retriever = FakeRetriever()
    use_case = AnswerQuestionUseCase(
        llm_provider=FakeLLMProvider(),
        retriever=retriever,
        validation_service=RequestValidationService(),
    )

    document_id = DocumentId("document-1")
    result = use_case.execute(
        AnswerQuestionRequest(question="How does it work?", document_id=document_id)
    )

    assert result.answer == "Grounded response [document-1:p1:chunk-1]"
    assert result.citations[0].chunk_id == "chunk-1"
    assert retriever.document_ids == (document_id,)


class StructuredFakeLLM(ILLMProvider):
    """Return one deterministic structured-generation response."""

    def __init__(self, output: str) -> None:
        self.output = output
        self.prompt = ""

    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return self.output

    def is_available(self) -> bool:
        return True

    def has_model(self) -> bool:
        return True


class FakeVectorStore:
    """Provide one source chunk to generation use cases."""

    def list_document_chunks(
        self,
        document_id: DocumentId,
    ) -> tuple[DocumentChunk, ...]:
        return (
            DocumentChunk(
                document_id=document_id,
                content="A useful concept.",
                page_number=1,
                section=None,
                chunk_id="chunk-1",
                ordinal=0,
            ),
        )


def test_quiz_use_case_caps_a_large_request_at_ten() -> None:
    llm = StructuredFakeLLM(
        json.dumps(
            [
                {
                    "prompt": f"Question {index}",
                    "answer": "Answer",
                    "citations": ["chunk-1"],
                }
                for index in range(12)
            ]
        )
    )
    use_case = GenerateQuizUseCase(
        llm_provider=llm,
        vector_store=FakeVectorStore(),  # type: ignore[arg-type]
        count_policy=GenerationCountPolicy(),
    )

    result = use_case.execute(
        GenerateQuizRequest(DocumentId("document-1"), question_count=50)
    )

    assert len(result.questions) == 10
    assert result.requested_count == 50
    assert result.effective_count == 10
    assert result.maximum_count == 10
    assert result.notice == (
        "You requested 50 quiz questions; the current limit is 10, so 10 "
        "were generated."
    )
    assert "Create exactly 10 study questions" in llm.prompt


def test_flashcard_use_case_caps_a_large_request_at_twenty() -> None:
    llm = StructuredFakeLLM(
        json.dumps(
            [
                {
                    "front": f"Concept {index}",
                    "back": "Definition",
                    "citations": ["chunk-1"],
                }
                for index in range(25)
            ]
        )
    )
    use_case = GenerateFlashcardsUseCase(
        llm_provider=llm,
        vector_store=FakeVectorStore(),  # type: ignore[arg-type]
        count_policy=GenerationCountPolicy(),
    )

    result = use_case.execute(
        GenerateFlashcardsRequest(DocumentId("document-1"), card_count=50)
    )

    assert len(result.cards) == 20
    assert result.requested_count == 50
    assert result.effective_count == 20
    assert result.maximum_count == 20
    assert result.notice == (
        "You requested 50 flashcards; the current limit is 20, so 20 were generated."
    )
    assert "Create exactly 20 study flashcards" in llm.prompt
