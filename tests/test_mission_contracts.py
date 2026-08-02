"""Focused citation and persistence contracts for the study mission roadmap."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scholar_agent.application.dtos.mission import (
    AdvanceStudyMissionRequest,
    AssessLearnerResponseRequest,
    ExplainConceptRequest,
)
from scholar_agent.application.dtos.retrieval import DocumentChunk, RetrievedChunk
from scholar_agent.application.dtos.study_requests import SummarizeDocumentRequest
from scholar_agent.application.output_ports.llm_provider import ILLMProvider
from scholar_agent.application.output_ports.tool_executor import IToolExecutor
from scholar_agent.application.services.mission_planning import MissionPlanner
from scholar_agent.application.use_cases.assess_learner_response import (
    AssessLearnerResponseUseCase,
)
from scholar_agent.application.use_cases.explain_concept import ExplainConceptUseCase
from scholar_agent.application.use_cases.summarize_document import (
    SummarizeDocumentUseCase,
)
from scholar_agent.domain.entities.study_session import (
    DocumentBrief,
    LearnerLevel,
    LearningObjective,
    MilestoneKind,
    MilestoneStatus,
    MissionStatus,
    SourceReference,
    StudyMilestone,
    StudyMode,
    StudyPlan,
    StudySession,
)
from scholar_agent.domain.value_objects.document_id import DocumentId
from scholar_agent.infrastructure.adapters.langchain_retriever import (
    LangChainRetriever,
)
from scholar_agent.infrastructure.adapters.langgraph_mission_runner import (
    LangGraphMissionRunner,
)
from scholar_agent.infrastructure.adapters.sqlite_study_session_repository import (
    SQLiteStudySessionRepository,
    _session_payload,
)
from scholar_agent.infrastructure.tools.capabilities import MISSION_CAPABILITIES
from scholar_agent.infrastructure.tools.semantic_search_tool import SemanticSearchTool


class QueuedLLM(ILLMProvider):
    """Deterministic model for strict structured-output tests."""

    def __init__(self, *outputs: str) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.outputs.pop(0)

    def is_available(self) -> bool:
        return True

    def has_model(self) -> bool:
        return True


class ChunkStore:
    """Minimal vector-store substitute for summary tests."""

    def __init__(self, chunks: tuple[DocumentChunk, ...]) -> None:
        self.chunks = chunks

    def list_document_chunks(
        self, document_id: DocumentId
    ) -> tuple[DocumentChunk, ...]:
        return self.chunks

    def get_chunk(self, document_id: DocumentId, chunk_id: str):
        return next(
            (
                chunk
                for chunk in self.chunks
                if chunk.document_id == document_id and chunk.chunk_id == chunk_id
            ),
            None,
        )


def _chunks() -> tuple[DocumentChunk, ...]:
    return tuple(
        DocumentChunk(
            document_id=DocumentId("document-1"),
            content=("source " + str(index) + " ") * 900,
            page_number=index,
            section=None,
            chunk_id=f"chunk-{index}",
            ordinal=index,
        )
        for index in (1, 2)
    )


def test_summary_repairs_strictly_and_preserves_segment_citations() -> None:
    chunks = _chunks()
    llm = QueuedLLM(
        "plain text is not accepted",
        json.dumps({"summary": "First", "citations": ["chunk-1"]}),
        json.dumps({"summary": "Second", "citations": ["chunk-2"]}),
        json.dumps({"summary": "Combined", "citations": ["chunk-1", "chunk-2"]}),
    )
    result = SummarizeDocumentUseCase(llm, ChunkStore(chunks)).execute(
        SummarizeDocumentRequest(DocumentId("document-1"))
    )

    assert result.summary == "Combined"
    assert [item.chunk_id for item in result.citations] == ["chunk-1", "chunk-2"]
    assert len(llm.prompts) == 4
    assert "[chunk-1|page=1]" in llm.prompts[0]
    assert "[chunk-2|page=2]" in llm.prompts[2]


def test_omitted_and_explicit_retrieval_limits_are_distinct() -> None:
    class Embeddings:
        def embed(self, text: str) -> tuple[float, ...]:
            return (1.0,)

    class Store:
        def __init__(self) -> None:
            self.limits: list[int] = []

        def search(self, embedding, limit=5, document_ids=()):
            self.limits.append(limit)
            return ()

    store = Store()
    retriever = LangChainRetriever(Embeddings(), store, default_limit=9)  # type: ignore[arg-type]
    retriever.retrieve("default")
    retriever.retrieve("explicit", limit=5)

    assert store.limits == [9, 5]


def test_semantic_search_is_bound_to_one_selected_document() -> None:
    class Retriever:
        def __init__(self) -> None:
            self.document_ids = ()

        def retrieve(self, query, limit=None, document_ids=()):
            self.document_ids = document_ids
            chunks = (
                RetrievedChunk(
                    DocumentId("document-1"), "one", 1, None, "chunk-1", 0.9
                ),
                RetrievedChunk(
                    DocumentId("document-2"), "two", 1, None, "chunk-2", 0.8
                ),
            )
            return tuple(chunk for chunk in chunks if chunk.document_id in document_ids)

    retriever = Retriever()
    result = SemanticSearchTool(retriever).execute(
        {"document_id": "document-1", "query": "bounded"}
    )

    assert retriever.document_ids == (DocumentId("document-1"),)
    assert [item["document_id"] for item in result["chunks"]] == ["document-1"]


def test_mission_capability_catalog_is_exact_and_excludes_answer_question() -> None:
    assert [item.task.value for item in MISSION_CAPABILITIES] == [
        "semantic_search",
        "summarize_document",
        "generate_quiz",
        "generate_flashcards",
        "citation_lookup",
        "build_document_map",
        "explain_concept",
        "assess_learner_response",
    ]


def test_explain_and_assess_are_strict_repaired_and_document_bound(
    tmp_path: Path,
) -> None:
    chunks = _chunks()[:1]
    repository = SQLiteStudySessionRepository(tmp_path / "catalog.sqlite3")
    source = SourceReference(DocumentId("document-1"), "chunk-1", 1, "Evidence")
    brief = DocumentBrief(
        DocumentId("document-1"),
        "brief",
        (LearningObjective("objective-1", "One", "One", (), (source,)),),
        (),
        (),
        (),
    )
    repository.save_brief(brief)
    store = ChunkStore(chunks)
    llm = QueuedLLM(
        "not json",
        json.dumps(
            {
                "explanation": "Cited explanation.",
                "check_question": "Check?",
                "citations": ["chunk-1"],
            }
        ),
        json.dumps(
            {
                "score": 2,
                "feedback": "Good.",
                "missing_concepts": [],
                "next_question": "Transfer?",
                "citations": ["chunk-1"],
            }
        ),
    )
    explain = ExplainConceptUseCase(llm, store, repository)
    assessment = AssessLearnerResponseUseCase(llm, store, repository)

    explanation = explain.execute(
        ExplainConceptRequest(DocumentId("document-1"), "objective-1", ("chunk-1",))
    )
    scored = assessment.execute(
        AssessLearnerResponseRequest(
            DocumentId("document-1"),
            "objective-1",
            "Check?",
            "My answer",
            ("chunk-1",),
        )
    )

    assert explanation.citations[0].chunk_id == "chunk-1"
    assert scored.score == 2
    assert len(llm.prompts) == 3

    with pytest.raises(ValueError, match="source_chunk_ids"):
        explain.execute(
            ExplainConceptRequest(DocumentId("document-1"), "objective-1", ())
        )
    with pytest.raises(ValueError, match="not in this document"):
        assessment.execute(
            AssessLearnerResponseRequest(
                DocumentId("document-1"),
                "objective-1",
                "Check?",
                "Answer",
                ("missing",),
            )
        )


def _session(identifier: str, document_id: str = "document-1") -> StudySession:
    doc = DocumentId(document_id)
    objective = LearningObjective("objective-1", "One", "One idea", (), ())
    brief = DocumentBrief(doc, "brief", (objective,), (), (), ())
    now = datetime.now(UTC)
    return StudySession(
        identifier=identifier,
        document_id=doc,
        goal="Learn",
        learner_level=LearnerLevel.INTERMEDIATE,
        mode=StudyMode.GUIDED,
        target_minutes=30,
        brief=brief,
        created_at=now,
        updated_at=now,
    )


def test_v1_reads_current_shape_and_save_writes_schema_version_four(
    tmp_path: Path,
) -> None:
    repository = SQLiteStudySessionRepository(tmp_path / "catalog.sqlite3")
    session = _session("legacy")
    payload = _session_payload(session)
    for key in (
        "schema_version",
        "status",
        "plan",
        "milestones",
        "artifacts",
        "pending_interaction",
        "trace",
        "action_count",
        "completed_at",
    ):
        payload.pop(key, None)
    with repository._connection:  # type: ignore[attr-defined]
        repository._connection.execute(  # type: ignore[attr-defined]
            "INSERT INTO study_sessions(session_id, document_id, payload, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (
                session.identifier,
                session.document_id.value,
                json.dumps(payload),
                session.updated_at.isoformat(),
            ),
        )

    restored = repository.get("legacy")
    assert restored is not None
    assert restored.status is MissionStatus.ACTIVE
    assert restored.milestones
    assert restored.artifacts == ()

    repository.save(restored)
    row = repository._connection.execute(  # type: ignore[attr-defined]
        "SELECT payload FROM study_sessions WHERE session_id = ?", ("legacy",)
    ).fetchone()
    assert json.loads(row[0])["schema_version"] == 4


def test_repository_lists_filters_completes_and_cascades(tmp_path: Path) -> None:
    repository = SQLiteStudySessionRepository(tmp_path / "catalog.sqlite3")
    first = _session("first")
    second = replace(
        _session("second"),
        updated_at=first.updated_at + timedelta(seconds=1),
    )
    other = _session("other", "document-2")
    repository.save(first)
    repository.save(second)
    repository.save(other)

    assert [item.identifier for item in repository.list(DocumentId("document-1"))] == [
        "second",
        "first",
    ]
    completed = repository.complete("first")
    assert completed is not None
    assert completed.status is MissionStatus.COMPLETED
    assert [
        item.identifier for item in repository.list(status=MissionStatus.COMPLETED)
    ] == ["first"]

    repository.delete_for_document(DocumentId("document-1"))
    assert repository.get("first") is None
    assert repository.get("second") is None
    assert repository.get("other") is not None


def test_planner_falls_back_to_earliest_prerequisite_valid_capacity() -> None:
    class PlannerLLM(QueuedLLM):
        pass

    references = ()
    objectives = (
        LearningObjective("objective-1", "One", "One", (), references),
        LearningObjective("objective-2", "Two", "Two", ("objective-1",), references),
        LearningObjective(
            "objective-3", "Three", "Three", ("objective-2",), references
        ),
    )
    brief = DocumentBrief(DocumentId("document-1"), "brief", objectives, (), (), ())
    planner = MissionPlanner(
        PlannerLLM(json.dumps({"focus": "Third", "objective_ids": ["objective-3"]}))
    )

    plan = planner.plan("Learn", LearnerLevel.INTERMEDIATE, StudyMode.GUIDED, 20, brief)

    assert plan.objective_ids == ("objective-1", "objective-2")


def test_mission_waits_remediates_persists_and_completes(tmp_path: Path) -> None:
    reference = {
        "document_id": "document-1",
        "chunk_id": "chunk-1",
        "page_number": 1,
        "excerpt": "Evidence",
    }

    class MissionTools(IToolExecutor):
        def __init__(self) -> None:
            self.assessment_scores = [0, 3, 3, 3]
            self.calls: list[str] = []

        def execute(self, tool_name, arguments):
            self.calls.append(tool_name)
            if tool_name == "build_document_map":
                return {"document_id": "document-1", "objectives": []}
            if tool_name == "explain_concept":
                return {
                    "objective_id": "objective-1",
                    "explanation": "Cited explanation.",
                    "check_question": "What is the key idea?",
                    "citations": [reference],
                }
            if tool_name == "semantic_search":
                return {
                    "chunks": [
                        {
                            "document_id": "document-1",
                            "chunk_id": "chunk-1",
                        }
                    ]
                }
            if tool_name == "assess_learner_response":
                return {
                    "objective_id": "objective-1",
                    "score": self.assessment_scores.pop(0),
                    "feedback": "Keep connecting the terms.",
                    "missing_concepts": ["the relation"],
                    "next_question": "How are the terms related?",
                    "citations": [reference],
                }
            if tool_name == "generate_quiz":
                return {
                    "questions": [
                        {
                            "prompt": "Final question?",
                            "answer": "The relation.",
                            "citations": [reference],
                        }
                    ]
                }
            raise AssertionError(tool_name)

        def capabilities(self):
            return ()

    repository = SQLiteStudySessionRepository(tmp_path / "catalog.sqlite3")
    document_id = DocumentId("document-1")
    source = SourceReference(document_id, "chunk-1", 1, "Evidence")
    objective = LearningObjective("objective-1", "One", "One", (), (source,))
    brief = DocumentBrief(document_id, "brief", (objective,), (), (), ())
    plan = StudyPlan("Learn", ("objective-1",), (source,))
    session = StudySession(
        "mission-1",
        document_id,
        "Learn",
        LearnerLevel.INTERMEDIATE,
        StudyMode.GUIDED,
        30,
        brief,
        plan=plan,
        milestones=(
            StudyMilestone(
                identifier="orient",
                kind=MilestoneKind.ORIENT,
                title="Orient",
                objective_id=None,
                capability="build_document_map",
                status=MilestoneStatus.ACTIVE,
                citations=(source,),
            ),
            StudyMilestone(
                identifier="learn",
                kind=MilestoneKind.LEARN,
                title="Learn",
                objective_id="objective-1",
                capability="explain_concept",
                status=MilestoneStatus.PENDING,
                citations=(source,),
            ),
            StudyMilestone(
                identifier="practice",
                kind=MilestoneKind.PRACTICE,
                title="Practice",
                objective_id="objective-1",
                capability="assess_learner_response",
                status=MilestoneStatus.PENDING,
                citations=(source,),
            ),
            StudyMilestone(
                identifier="review",
                kind=MilestoneKind.REVIEW,
                title="Review",
                objective_id=None,
                capability="generate_quiz",
                status=MilestoneStatus.PENDING,
                citations=(source,),
            ),
        ),
    )
    repository.save(session)
    runner = LangGraphMissionRunner(MissionTools(), repository)

    first = runner.run(AdvanceStudyMissionRequest("mission-1"))
    assert first.session.status is MissionStatus.AWAITING_LEARNER
    assert first.session.pending_interaction is not None
    assert first.session.action_count == 2
    assert repository.get("mission-1") == first.session

    second = runner.run(AdvanceStudyMissionRequest("mission-1", "An incomplete answer"))
    assert second.session.pending_interaction is not None
    assert second.session.action_count == 5
    assert second.session.status is MissionStatus.AWAITING_LEARNER

    third = runner.run(AdvanceStudyMissionRequest("mission-1", "The relation"))
    assert third.session.pending_interaction is not None
    fourth = runner.run(AdvanceStudyMissionRequest("mission-1", "The relation"))
    assert fourth.session.status is MissionStatus.AWAITING_LEARNER
    fifth = runner.run(AdvanceStudyMissionRequest("mission-1", "The relation"))
    assert fifth.session.status is MissionStatus.COMPLETED
    assert fifth.session.action_count == 9
    assert all(event.event_type != "raw_output" for event in fifth.session.trace)
