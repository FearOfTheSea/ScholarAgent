"""Build and cache a cited document learning map."""

from scholar_agent.application.dtos.tutor import BuildDocumentBriefResult
from scholar_agent.application.output_ports.llm_provider import ILLMProvider
from scholar_agent.application.output_ports.vector_store import IVectorStore
from scholar_agent.application.services.document_brief_parser import (
    document_brief_prompt,
    document_brief_repair_prompt,
    parse_document_brief,
)
from scholar_agent.domain.exceptions.document_not_found_error import (
    DocumentNotFoundError,
)
from scholar_agent.domain.repositories.study_session_repository import (
    StudySessionRepository,
)
from scholar_agent.domain.value_objects.document_id import DocumentId


class BuildDocumentBriefUseCase:
    """Create one reusable cited brief for a locally indexed document."""

    def __init__(
        self,
        llm_provider: ILLMProvider,
        vector_store: IVectorStore,
        session_repository: StudySessionRepository,
    ) -> None:
        self._llm_provider = llm_provider
        self._vector_store = vector_store
        self._session_repository = session_repository

    def execute(self, document_id: DocumentId) -> BuildDocumentBriefResult:
        """Return a cached brief or generate and validate it once."""
        cached = self._session_repository.get_brief(document_id)
        if cached is not None:
            return BuildDocumentBriefResult(cached, cached=True)
        chunks = self._vector_store.list_document_chunks(document_id)
        if not chunks:
            raise DocumentNotFoundError(document_id.value)
        prompt = document_brief_prompt(document_id, chunks)
        raw_output = self._llm_provider.generate(prompt)
        try:
            brief = parse_document_brief(raw_output, document_id, chunks)
        except ValueError as first_error:
            repaired = self._llm_provider.generate(
                document_brief_repair_prompt(prompt, raw_output, str(first_error))
            )
            brief = parse_document_brief(repaired, document_id, chunks)
        self._session_repository.save_brief(brief)
        return BuildDocumentBriefResult(brief, cached=False)
