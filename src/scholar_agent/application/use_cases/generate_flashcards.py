"""Generate-flashcards use case placeholder."""

from scholar_agent.application.dtos.study_requests import GenerateFlashcardsRequest
from scholar_agent.application.dtos.study_results import (
    Flashcard,
    GenerateFlashcardsResult,
)
from scholar_agent.application.input_ports.study_assistant import GenerateFlashcards
from scholar_agent.application.output_ports.llm_provider import ILLMProvider
from scholar_agent.application.output_ports.vector_store import IVectorStore
from scholar_agent.application.services.generation_count_policy import (
    GenerationCountPolicy,
    generation_limit_notice,
)
from scholar_agent.application.services.structured_output import parse_cited_items
from scholar_agent.application.services.study_prompts import (
    chunks_to_source_text,
    flashcards_prompt,
)
from scholar_agent.domain.exceptions.document_not_found_error import (
    DocumentNotFoundError,
)


class GenerateFlashcardsUseCase(GenerateFlashcards):
    """Coordinates flashcard generation when the capability is implemented."""

    def __init__(
        self,
        llm_provider: ILLMProvider,
        vector_store: IVectorStore,
        count_policy: GenerationCountPolicy,
    ) -> None:
        self._llm_provider = llm_provider
        self._vector_store = vector_store
        self._count_policy = count_policy

    def execute(self, request: GenerateFlashcardsRequest) -> GenerateFlashcardsResult:
        """Generate typed flashcards from one document."""
        count = self._count_policy.flashcards(request.card_count)
        chunks = self._vector_store.list_document_chunks(request.document_id)
        if not chunks:
            raise DocumentNotFoundError(request.document_id.value)
        prompt = flashcards_prompt(chunks_to_source_text(chunks), count.effective)
        raw_output = self._llm_provider.generate(prompt)
        try:
            parsed = parse_cited_items(
                raw_output,
                "front",
                "back",
                chunks,
            )
        except ValueError as first_error:
            repaired = self._llm_provider.generate(
                f"{prompt}\nVALIDATION ERROR: {first_error}\n"
                "Return only the corrected JSON array."
            )
            parsed = parse_cited_items(repaired, "front", "back", chunks)
        cards = tuple(
            Flashcard(front=front, back=back, citations=citations)
            for front, back, citations in parsed
        )
        citations = tuple(reference for card in cards for reference in card.citations)
        citations = tuple(dict.fromkeys(citations))
        return GenerateFlashcardsResult(
            cards=cards[: count.effective],
            requested_count=count.requested,
            effective_count=count.effective,
            maximum_count=count.maximum,
            notice=generation_limit_notice("flashcards", count),
            citations=citations,
        )
