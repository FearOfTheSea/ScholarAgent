"""Generate-quiz use case placeholder."""

from scholar_agent.application.dtos.study_requests import GenerateQuizRequest
from scholar_agent.application.dtos.study_results import (
    GenerateQuizResult,
    QuizQuestion,
)
from scholar_agent.application.input_ports.study_assistant import GenerateQuiz
from scholar_agent.application.output_ports.llm_provider import ILLMProvider
from scholar_agent.application.output_ports.vector_store import IVectorStore
from scholar_agent.application.services.generation_count_policy import (
    GenerationCountPolicy,
    generation_limit_notice,
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
        count_policy: GenerationCountPolicy,
    ) -> None:
        self._llm_provider = llm_provider
        self._vector_store = vector_store
        self._count_policy = count_policy

    def execute(self, request: GenerateQuizRequest) -> GenerateQuizResult:
        """Generate typed study questions from one document."""
        count = self._count_policy.quiz(request.question_count)
        chunks = self._vector_store.list_document_chunks(request.document_id)
        if not chunks:
            raise DocumentNotFoundError(request.document_id.value)
        raw_output = self._llm_provider.generate(
            quiz_prompt(chunks_to_source_text(chunks), count.effective),
        )
        questions = tuple(
            QuizQuestion(prompt=prompt, answer=answer)
            for prompt, answer in parse_items(raw_output, "prompt", "answer")
        )
        return GenerateQuizResult(
            questions=questions[: count.effective],
            requested_count=count.requested,
            effective_count=count.effective,
            maximum_count=count.maximum,
            notice=generation_limit_notice("quiz questions", count),
        )
