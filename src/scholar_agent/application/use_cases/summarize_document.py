"""Summarize-document use case placeholder."""

from scholar_agent.application.dtos.study_requests import SummarizeDocumentRequest
from scholar_agent.application.dtos.study_results import SummarizeDocumentResult
from scholar_agent.application.input_ports.study_assistant import SummarizeDocument
from scholar_agent.application.output_ports.llm_provider import ILLMProvider
from scholar_agent.application.output_ports.vector_store import IVectorStore
from scholar_agent.application.services.study_prompts import (
    chunks_to_source_text,
    combine_summaries_prompt,
    split_source_text,
    summarize_prompt,
)
from scholar_agent.domain.exceptions.document_not_found_error import (
    DocumentNotFoundError,
)


class SummarizeDocumentUseCase(SummarizeDocument):
    """Coordinates document summarization when the capability is implemented."""

    def __init__(self, llm_provider: ILLMProvider, vector_store: IVectorStore) -> None:
        self._llm_provider = llm_provider
        self._vector_store = vector_store

    def execute(self, request: SummarizeDocumentRequest) -> SummarizeDocumentResult:
        """Summarize a document within the local model's context budget."""
        chunks = self._vector_store.list_document_chunks(request.document_id)
        if not chunks:
            raise DocumentNotFoundError(request.document_id.value)
        source_text = chunks_to_source_text(chunks, maximum_length=None)
        partial_summaries = tuple(
            self._llm_provider.generate(summarize_prompt(segment))
            for segment in split_source_text(source_text)
        )
        if len(partial_summaries) == 1:
            return SummarizeDocumentResult(summary=partial_summaries[0])
        summary = self._llm_provider.generate(
            combine_summaries_prompt(partial_summaries)
        )
        return SummarizeDocumentResult(summary=summary)
