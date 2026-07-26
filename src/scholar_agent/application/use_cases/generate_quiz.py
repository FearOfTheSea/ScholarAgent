"""Generate-quiz use case placeholder."""

from scholar_agent.application.dtos.study_requests import GenerateQuizRequest
from scholar_agent.application.dtos.study_results import (
    GenerateQuizResult,
    QuizQuestion,
)
from scholar_agent.application.input_ports.study_assistant import GenerateQuiz
from scholar_agent.application.output_ports.llm_provider import ILLMProvider
from scholar_agent.application.output_ports.vector_store import IVectorStore
from scholar_agent.application.services.request_validation_service import (
    RequestValidationService,
)
from scholar_agent.application.services.structured_output import parse_items
from scholar_agent.application.services.study_prompts import (
    chunks_to_source_text,
    quiz_prompt,
)
from scholar_agent.domain.exceptions.document_not_found_error import (
    DocumentNotFoundError,
)


class GenerateQuizUseCase(GenerateQuiz):
    """Coordinates quiz generation when the capability is implemented."""

    def __init__(
        self,
        llm_provider: ILLMProvider,
        vector_store: IVectorStore,
        validation_service: RequestValidationService,
    ) -> None:
        self._llm_provider = llm_provider
        self._vector_store = vector_store
        self._validation_service = validation_service

    def execute(self, request: GenerateQuizRequest) -> GenerateQuizResult:
        """Generate typed study questions from one document."""
        question_count = self._validation_service.validate_count(
            request.question_count,
            "question_count",
        )
        chunks = self._vector_store.list_document_chunks(request.document_id)
        if not chunks:
            raise DocumentNotFoundError(request.document_id.value)
        raw_output = self._llm_provider.generate(
            quiz_prompt(chunks_to_source_text(chunks), question_count),
        )
        questions = tuple(
            QuizQuestion(prompt=prompt, answer=answer)
            for prompt, answer in parse_items(raw_output, "prompt", "answer")
        )
        return GenerateQuizResult(questions=questions[:question_count])
