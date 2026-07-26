"""Compare-documents use case placeholder."""

from scholar_agent.application.dtos.study_requests import CompareDocumentsRequest
from scholar_agent.application.dtos.study_results import CompareDocumentsResult
from scholar_agent.application.input_ports.study_assistant import CompareDocuments
from scholar_agent.application.output_ports.llm_provider import ILLMProvider
from scholar_agent.application.output_ports.retriever import IRetriever
from scholar_agent.application.services.study_prompts import compare_documents_prompt


class CompareDocumentsUseCase(CompareDocuments):
    """Coordinates document comparison when the capability is implemented."""

    def __init__(self, llm_provider: ILLMProvider, retriever: IRetriever) -> None:
        self._llm_provider = llm_provider
        self._retriever = retriever

    def execute(self, request: CompareDocumentsRequest) -> CompareDocumentsResult:
        """Compare independently retrieved evidence from two documents."""
        first_chunks = self._retriever.retrieve(
            query="key concepts, claims, and evidence",
            document_ids=(request.first_document_id,),
        )
        second_chunks = self._retriever.retrieve(
            query="key concepts, claims, and evidence",
            document_ids=(request.second_document_id,),
        )
        citations = first_chunks + second_chunks
        if not citations:
            return CompareDocumentsResult(
                comparison=(
                    "The selected documents do not provide enough extractable "
                    "information to compare."
                ),
                citations=(),
            )
        comparison = self._llm_provider.generate(
            compare_documents_prompt(first_chunks, second_chunks),
        )
        return CompareDocumentsResult(comparison=comparison, citations=citations)
