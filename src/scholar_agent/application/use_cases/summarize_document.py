"""Summarize-document use case placeholder."""

from scholar_agent.application.dtos.study_requests import SummarizeDocumentRequest
from scholar_agent.application.dtos.study_results import SummarizeDocumentResult
from scholar_agent.application.input_ports.study_assistant import SummarizeDocument
from scholar_agent.application.output_ports.llm_provider import ILLMProvider
from scholar_agent.application.output_ports.vector_store import IVectorStore
from scholar_agent.application.services.structured_output import parse_cited_summary
from scholar_agent.application.services.study_prompts import (
    chunks_to_source_text,
    combine_summaries_prompt,
    split_source_chunks,
    summarize_prompt,
)
from scholar_agent.domain.entities.study_session import SourceReference
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
        segments = split_source_chunks(chunks)
        partial_summaries: list[str] = []
        partial_citations: list[SourceReference] = []
        for segment_chunks in segments:
            prompt = summarize_prompt(
                chunks_to_source_text(segment_chunks, maximum_length=None)
            )
            raw_output = self._llm_provider.generate(prompt)
            try:
                summary, citations = parse_cited_summary(raw_output, segment_chunks)
            except ValueError as first_error:
                repaired = self._llm_provider.generate(
                    prompt + f"\nVALIDATION ERROR: {first_error}\n"
                    "Return the required JSON now."
                )
                summary, citations = parse_cited_summary(repaired, segment_chunks)
            partial_summaries.append(summary)
            partial_citations.extend(citations)
        if len(partial_summaries) == 1:
            return SummarizeDocumentResult(
                summary=partial_summaries[0],
                citations=tuple(partial_citations),
            )
        unique_citations = tuple(dict.fromkeys(partial_citations))
        raw_combined = self._llm_provider.generate(
            combine_summaries_prompt(tuple(partial_summaries), unique_citations)
        )
        try:
            summary, citations = parse_cited_summary(raw_combined, unique_citations)
        except ValueError as first_error:
            repaired = self._llm_provider.generate(
                combine_summaries_prompt(tuple(partial_summaries), unique_citations)
                + f"\nVALIDATION ERROR: {first_error}\nReturn the required JSON now."
            )
            summary, citations = parse_cited_summary(repaired, unique_citations)
        return SummarizeDocumentResult(summary=summary, citations=citations)
