"""Tests for the persistent, grounded single-document tutor."""

import json
from datetime import UTC, datetime
from pathlib import Path

from scholar_agent.application.dtos.retrieval import DocumentChunk
from scholar_agent.application.dtos.tutor import ContinueStudySessionRequest
from scholar_agent.application.output_ports.llm_provider import ILLMProvider
from scholar_agent.application.output_ports.retriever import IRetriever
from scholar_agent.application.services.document_brief_parser import (
    parse_document_brief,
)
from scholar_agent.application.services.tutor_turn_service import TutorTurnService
from scholar_agent.config.settings import Settings
from scholar_agent.domain.entities.study_session import (
    ConceptNode,
    DocumentBrief,
    LearnerAttempt,
    LearnerLevel,
    LearningObjective,
    MasteryLabel,
    SourceReference,
    StudyMode,
    StudySession,
    objective_progress,
)
from scholar_agent.domain.value_objects.document_id import DocumentId
from scholar_agent.infrastructure.adapters.sqlite_study_session_repository import (
    SQLiteStudySessionRepository,
)
from scholar_agent.presentation.api.main import create_app


class QueueLLM(ILLMProvider):
    """Return deterministic structured tutor outputs."""

    def __init__(self, *outputs: str) -> None:
        self.outputs = list(outputs)

    def generate(self, prompt: str) -> str:
        del prompt
        return self.outputs.pop(0)

    def is_available(self) -> bool:
        return True

    def has_model(self) -> bool:
        return True


class EmptyRetriever(IRetriever):
    """Retriever unused by answer-assessment tests."""

    def retrieve(
        self,
        query: str,
        limit: int = 4,
        document_ids: tuple[DocumentId, ...] = (),
    ):
        del query, limit, document_ids
        return ()


def test_document_brief_requires_real_single_document_citations() -> None:
    document_id = DocumentId("document-1")
    chunks = (DocumentChunk(document_id, "Evidence", 2, None, "chunk-1", 0),)
    raw = json.dumps(
        {
            "synopsis": "A focused synopsis.",
            "objectives": [
                {
                    "id": "objective-1",
                    "title": "Understand evidence",
                    "description": "Explain the key evidence.",
                    "prerequisites": [],
                    "citations": ["chunk-1"],
                }
            ],
            "concepts": [
                {
                    "id": "concept-1",
                    "label": "Evidence",
                    "explanation": "The document's central evidence.",
                    "prerequisites": [],
                    "citations": ["chunk-1"],
                }
            ],
            "glossary": [
                {
                    "term": "Evidence",
                    "definition": "Support for a claim.",
                    "citations": ["chunk-1"],
                }
            ],
            "misconceptions": [
                {
                    "term": "Evidence equals opinion",
                    "definition": "Evidence supports a claim; opinion alone does not.",
                    "citations": ["chunk-1"],
                }
            ],
        }
    )

    brief = parse_document_brief(raw, document_id, chunks)

    assert brief.objectives[0].citations[0].page_number == 2
    assert brief.concepts[0].citations[0].document_id == document_id
    assert brief.misconceptions == (
        "Evidence equals opinion: Evidence supports a claim; opinion alone does not.",
    )


def test_mastery_uses_latest_three_attempts_and_requires_two_for_mastered() -> None:
    reference = _reference()
    attempts = tuple(
        LearnerAttempt(
            "objective-1",
            f"answer-{index}",
            score,
            "feedback",
            (),
            (reference,),
            datetime.now(UTC),
        )
        for index, score in enumerate((0, 3, 3, 3))
    )

    one_attempt = objective_progress("objective-1", attempts[:1])
    mastered = objective_progress("objective-1", attempts)

    assert one_attempt.label is MasteryLabel.DEVELOPING
    assert mastered.percentage == 100
    assert mastered.label is MasteryLabel.MASTERED
    assert mastered.attempt_count == 4


def test_sqlite_repository_round_trips_and_cascades_document_state(
    tmp_path: Path,
) -> None:
    repository = SQLiteStudySessionRepository(tmp_path / "catalog.sqlite3")
    session = _session()

    repository.save_brief(session.brief)
    repository.save(session)

    restored = repository.get(session.identifier)
    assert restored == session
    assert repository.get_brief(session.document_id) == session.brief

    repository.delete_for_document(session.document_id)

    assert repository.get(session.identifier) is None
    assert repository.get_brief(session.document_id) is None


def test_tutor_scores_answer_persists_turn_and_updates_mastery(
    tmp_path: Path,
) -> None:
    repository = SQLiteStudySessionRepository(tmp_path / "catalog.sqlite3")
    session = _session()
    repository.save(session)
    service = TutorTurnService(
        QueueLLM(
            json.dumps(
                {
                    "score": 3,
                    "feedback": "The answer identifies the documented concept.",
                    "missing_concepts": [],
                    "next_question": "How is the concept applied?",
                }
            ),
            json.dumps(
                {
                    "supported": True,
                    "response": "The answer identifies the documented concept.",
                }
            ),
        ),
        EmptyRetriever(),
        repository,
    )
    request = ContinueStudySessionRequest(session.identifier, "It is documented.")

    prepared = service.prepare(request, service.classify(request))
    result = service.persist(prepared)

    assert result.intent == "answer"
    assert result.assessment is not None
    assert result.assessment.score == 3
    assert result.progress[0].label is MasteryLabel.PROFICIENT
    restored = repository.get(session.identifier)
    assert restored is not None
    assert len(restored.turns) == 1


def test_tutor_rejects_cross_document_and_web_requests_without_llm(
    tmp_path: Path,
) -> None:
    repository = SQLiteStudySessionRepository(tmp_path / "catalog.sqlite3")
    session = _session()
    repository.save(session)
    service = TutorTurnService(QueueLLM(), EmptyRetriever(), repository)
    request = ContinueStudySessionRequest(
        session.identifier,
        "Compare this with another document from the web.",
    )

    prepared = service.prepare(request, service.classify(request))
    result = service.persist(prepared)

    assert result.intent == "unsupported"
    assert result.activity.citations == ()
    assert "one selected document" in result.activity.message


def test_adaptive_tutor_api_contract_is_exposed(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            catalog_db_path=tmp_path / "catalog.sqlite3",
            document_library_path=tmp_path / "documents",
            vector_db_path=tmp_path / "vectors",
        )
    )

    paths = app.openapi()["paths"]

    assert "post" in paths["/agent/sessions"]
    assert {"get", "delete"} <= set(paths["/agent/sessions/{session_id}"])
    assert "post" in paths["/agent/sessions/{session_id}/turns"]


def _reference() -> SourceReference:
    return SourceReference(
        DocumentId("document-1"),
        "chunk-1",
        1,
        "Documented concept evidence.",
    )


def _session() -> StudySession:
    now = datetime.now(UTC)
    reference = _reference()
    brief = DocumentBrief(
        document_id=reference.document_id,
        synopsis="Synopsis.",
        objectives=(
            LearningObjective(
                "objective-1",
                "Concept",
                "Explain the documented concept.",
                (),
                (reference,),
            ),
        ),
        concepts=(
            ConceptNode(
                "concept-1",
                "Concept",
                "Documented concept.",
                (),
                (reference,),
            ),
        ),
        glossary=(),
        misconceptions=(),
    )
    return StudySession(
        identifier="session-1",
        document_id=reference.document_id,
        goal="Learn.",
        learner_level=LearnerLevel.INTERMEDIATE,
        mode=StudyMode.GUIDED,
        target_minutes=30,
        brief=brief,
        created_at=now,
        updated_at=now,
    )
