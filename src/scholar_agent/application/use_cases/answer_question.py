"""Answer-question use case placeholder."""

from scholar_agent.application.dtos.study_requests import AnswerQuestionRequest
from scholar_agent.application.dtos.study_results import AnswerQuestionResult
from scholar_agent.application.input_ports.study_assistant import AnswerQuestion
from scholar_agent.application.output_ports.llm_provider import ILLMProvider
from scholar_agent.application.output_ports.retriever import IRetriever
from scholar_agent.application.services.request_validation_service import (
    RequestValidationService,
)
from scholar_agent.application.services.study_prompts import answer_question_prompt


class AnswerQuestionUseCase(AnswerQuestion):
    """Coordinates question answering when the capability is implemented."""

    def __init__(
        self,
        llm_provider: ILLMProvider,
        retriever: IRetriever,
        validation_service: RequestValidationService,
    ) -> None:
        self._llm_provider = llm_provider
        self._retriever = retriever
        self._validation_service = validation_service

    def execute(self, request: AnswerQuestionRequest) -> AnswerQuestionResult:
        """Answer a study question from retrieved document excerpts."""
        question = self._validation_service.validate_text(request.question, "question")
        citations = self._retriever.retrieve(
            query=question,
            document_ids=(request.document_id,),
        )
        if not citations:
            return AnswerQuestionResult(
                answer=(
                    "The selected document does not provide enough information to "
                    "answer this question."
                ),
                citations=(),
            )
        answer = self._llm_provider.generate(
            answer_question_prompt(question, citations)
        )
        return AnswerQuestionResult(answer=answer, citations=citations)
