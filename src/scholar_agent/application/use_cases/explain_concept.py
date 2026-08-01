"""Explain one cited objective for a single-document mission."""

from scholar_agent.application.dtos.mission import (
    ExplainConceptRequest,
    ExplainConceptResult,
)
from scholar_agent.application.dtos.retrieval import DocumentChunk
from scholar_agent.application.output_ports.llm_provider import ILLMProvider
from scholar_agent.application.output_ports.vector_store import IVectorStore
from scholar_agent.application.services.mission_prompts import (
    chunks_to_mission_source_text,
    explain_concept_prompt,
)
from scholar_agent.application.services.structured_output import parse_explanation
from scholar_agent.domain.entities.study_session import LearningObjective
from scholar_agent.domain.repositories.study_session_repository import (
    StudySessionRepository,
)
from scholar_agent.domain.value_objects.document_id import DocumentId


class ExplainConceptUseCase:
    """Generate and validate one cited explanation."""

    def __init__(
        self,
        llm_provider: ILLMProvider,
        vector_store: IVectorStore,
        session_repository: StudySessionRepository,
    ) -> None:
        self._llm_provider = llm_provider
        self._vector_store = vector_store
        self._session_repository = session_repository

    def execute(self, request: ExplainConceptRequest) -> ExplainConceptResult:
        """Explain the requested objective from the requested evidence only."""
        self._validate_request(request)
        objective = self._objective(request.document_id, request.objective_id)
        chunks = self._chunks(request.document_id, request.source_chunk_ids)
        prompt = explain_concept_prompt(
            objective.identifier,
            request.learner_question,
            request.style,
            chunks_to_mission_source_text(chunks),
        )
        raw_output = self._llm_provider.generate(prompt)
        try:
            explanation, check_question, citations = parse_explanation(
                raw_output, chunks
            )
        except ValueError as first_error:
            repaired = self._llm_provider.generate(
                f"{prompt}\nVALIDATION ERROR: {first_error}\n"
                "Return corrected JSON only."
            )
            explanation, check_question, citations = parse_explanation(repaired, chunks)
        return ExplainConceptResult(
            objective_id=objective.identifier,
            explanation=explanation,
            check_question=check_question,
            citations=citations,
        )

    @staticmethod
    def _validate_request(request: ExplainConceptRequest) -> None:
        if not request.objective_id.strip():
            raise ValueError("objective_id must be non-blank text.")
        if not request.source_chunk_ids or any(
            not item.strip() for item in request.source_chunk_ids
        ):
            raise ValueError("source_chunk_ids must contain non-blank IDs.")
        if (
            request.learner_question is not None
            and not request.learner_question.strip()
        ):
            raise ValueError("learner_question must be non-blank when supplied.")
        if not request.style.strip():
            raise ValueError("style must be non-blank text.")

    def _objective(
        self, document_id: DocumentId, objective_id: str
    ) -> LearningObjective:
        brief = self._session_repository.get_brief(document_id)
        if brief is None:
            raise ValueError("A cited document map is required before explaining.")
        for objective in brief.objectives:
            if objective.identifier == objective_id:
                return objective
        raise ValueError(f"Unknown objective '{objective_id}'.")

    def _chunks(
        self, document_id: DocumentId, chunk_ids: tuple[str, ...]
    ) -> tuple[DocumentChunk, ...]:
        if not chunk_ids:
            raise ValueError("At least one source chunk ID is required.")
        if len(chunk_ids) > 20 or len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("source_chunk_ids must contain 1 to 20 unique IDs.")
        chunks = []
        for chunk_id in chunk_ids:
            chunk = self._vector_store.get_chunk(document_id, chunk_id)
            if chunk is None or chunk.document_id != document_id:
                raise ValueError(f"Source chunk '{chunk_id}' is not in this document.")
            chunks.append(chunk)
        return tuple(chunks)
