"""Generate-flashcards use case placeholder."""

from scholar_agent.application.dtos.study_requests import GenerateFlashcardsRequest
from scholar_agent.application.dtos.study_results import (
    Flashcard,
    GenerateFlashcardsResult,
)
from scholar_agent.application.input_ports.study_assistant import GenerateFlashcards
from scholar_agent.application.output_ports.llm_provider import ILLMProvider
from scholar_agent.application.output_ports.vector_store import IVectorStore
from scholar_agent.application.services.request_validation_service import (
    RequestValidationService,
)
from scholar_agent.application.services.structured_output import parse_items
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
        validation_service: RequestValidationService,
    ) -> None:
        self._llm_provider = llm_provider
        self._vector_store = vector_store
        self._validation_service = validation_service

    def execute(self, request: GenerateFlashcardsRequest) -> GenerateFlashcardsResult:
        """Generate typed flashcards from one document."""
        card_count = self._validation_service.validate_count(
            request.card_count, "card_count"
        )
        chunks = self._vector_store.list_document_chunks(request.document_id)
        if not chunks:
            raise DocumentNotFoundError(request.document_id.value)
        raw_output = self._llm_provider.generate(
            flashcards_prompt(chunks_to_source_text(chunks), card_count),
        )
        cards = tuple(
            Flashcard(front=front, back=back)
            for front, back in parse_items(raw_output, "front", "back")
        )
        return GenerateFlashcardsResult(cards=cards[:card_count])
