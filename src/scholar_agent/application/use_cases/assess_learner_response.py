"""Assess one learner response for a single-document mission."""

from scholar_agent.application.dtos.mission import (
    AssessLearnerResponseRequest,
    AssessLearnerResponseResult,
)
from scholar_agent.application.dtos.retrieval import DocumentChunk
from scholar_agent.application.output_ports.llm_provider import ILLMProvider
from scholar_agent.application.output_ports.vector_store import IVectorStore
from scholar_agent.application.services.mission_prompts import (
    assess_response_prompt,
    chunks_to_mission_source_text,
)
from scholar_agent.application.services.structured_output import parse_assessment
from scholar_agent.domain.repositories.study_session_repository import (
    StudySessionRepository,
)
from scholar_agent.domain.value_objects.document_id import DocumentId


class AssessLearnerResponseUseCase:
    """Generate and validate one bounded learner assessment."""

    def __init__(
        self,
        llm_provider: ILLMProvider,
        vector_store: IVectorStore,
        session_repository: StudySessionRepository,
    ) -> None:
        self._llm_provider = llm_provider
        self._vector_store = vector_store
        self._session_repository = session_repository

    def execute(
        self, request: AssessLearnerResponseRequest
    ) -> AssessLearnerResponseResult:
        """Assess the response using only the pending question's evidence."""
        self._validate_request(request)
        self._ensure_objective(request.document_id, request.objective_id)
        chunks = self._chunks(request.document_id, request.source_chunk_ids)
        prompt = assess_response_prompt(
            request.objective_id,
            request.pending_question,
            request.learner_response,
            chunks_to_mission_source_text(chunks),
        )
        raw_output = self._llm_provider.generate(prompt)
        try:
            parsed = parse_assessment(raw_output, chunks)
        except ValueError as first_error:
            repaired = self._llm_provider.generate(
                f"{prompt}\nVALIDATION ERROR: {first_error}\n"
                "Return corrected JSON only."
            )
            parsed = parse_assessment(repaired, chunks)
        score, feedback, missing, next_question, citations = parsed
        return AssessLearnerResponseResult(
            objective_id=request.objective_id,
            score=score,
            feedback=feedback,
            missing_concepts=missing,
            next_question=next_question,
            citations=citations,
        )

    @staticmethod
    def _validate_request(request: AssessLearnerResponseRequest) -> None:
        fields = (
            ("objective_id", request.objective_id),
            ("pending_question", request.pending_question),
            ("learner_response", request.learner_response),
        )
        for field, value in fields:
            if not value.strip():
                raise ValueError(f"{field} must be non-blank text.")
        if not request.source_chunk_ids or any(
            not item.strip() for item in request.source_chunk_ids
        ):
            raise ValueError("source_chunk_ids must contain non-blank IDs.")

    def _ensure_objective(self, document_id: DocumentId, objective_id: str) -> None:
        brief = self._session_repository.get_brief(document_id)
        if brief is None or not any(
            item.identifier == objective_id for item in brief.objectives
        ):
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
