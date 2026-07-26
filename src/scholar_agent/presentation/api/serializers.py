"""Presentation serializers for application and domain results."""

from scholar_agent.application.dtos.retrieval import RetrievedChunk
from scholar_agent.domain.entities.document import Document
from scholar_agent.presentation.api.models import CitationResponse, DocumentResponse


def document_response(document: Document) -> DocumentResponse:
    """Convert a domain document into an API response model."""
    return DocumentResponse(
        id=document.identifier.value,
        title=document.title,
        source=document.source,
        page_count=document.page_count,
        created_at=document.created_at,
    )


def citation_response(chunk: RetrievedChunk) -> CitationResponse:
    """Convert a retrieved chunk into public citation metadata."""
    return CitationResponse(
        document_id=chunk.document_id.value,
        chunk_id=chunk.chunk_id,
        page_number=chunk.page_number,
        section=chunk.section,
        similarity_score=chunk.similarity_score,
    )
