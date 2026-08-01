"""Acceptance tests for the real LangGraph mission boundary."""

import asyncio
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from dependency_injector import providers
from httpx import ASGITransport, AsyncClient

from scholar_agent.application.dtos.mission import AdvanceStudyMissionRequest
from scholar_agent.application.dtos.tutor import (
    StartStudySessionRequest,
    StudySessionResult,
)
from scholar_agent.application.output_ports.tool_executor import IToolExecutor
from scholar_agent.config.settings import Settings
from scholar_agent.domain.entities.study_session import (
    DocumentBrief,
    LearnerLevel,
    LearningObjective,
    MilestoneKind,
    MilestoneStatus,
    MissionStatus,
    PendingLearnerInteraction,
    SourceReference,
    StudyMilestone,
    StudyMode,
    StudyPlan,
    StudySession,
    objective_progress,
)
from scholar_agent.domain.value_objects.document_id import DocumentId
from scholar_agent.infrastructure.adapters.langgraph_mission_runner import (
    LangGraphMissionRunner,
)
from scholar_agent.infrastructure.adapters.sqlite_study_session_repository import (
    SQLiteStudySessionRepository,
)
from scholar_agent.presentation.api.main import create_app


class MapTools(IToolExecutor):
    """Deterministic map capability used to inspect graph loops and bounds."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, tool_name, arguments):
        self.calls.append(tool_name)
        if tool_name == "build_document_map":
            return {"document_id": arguments["document_id"], "objectives": []}
        raise AssertionError(tool_name)

    def capabilities(self):
        return ()


class BrokenSummaryTools(MapTools):
    """Return an invalid optional artifact after capability checkpointing."""

    def execute(self, tool_name, arguments):
        if tool_name == "summarize_document":
            self.calls.append(tool_name)
            return {}
        return super().execute(tool_name, arguments)


def test_mission_adapter_compiles_and_invokes_inspectable_state_graph(
    tmp_path: Path,
) -> None:
    repository = SQLiteStudySessionRepository(tmp_path / "catalog.sqlite3")
    tools = MapTools()
    session = _mission_session(
        "graph",
        tuple(_milestone(index) for index in range(2)),
    )
    repository.save(session)
    runner = LangGraphMissionRunner(tools, repository)

    compiled = runner.build_graph()
    assert {
        "load",
        "classify",
        "select",
        "execute",
        "verify",
        "reflect",
        "checkpoint",
        "finalize",
    } <= set(compiled.get_graph().nodes)

    result = runner.run(AdvanceStudyMissionRequest(session.identifier))

    assert tools.calls == ["build_document_map", "build_document_map"]
    assert result.session.status is MissionStatus.COMPLETED
    assert result.session.action_count == 2
    assert [entry.sequence for entry in result.session.ledger] == list(
        range(1, len(result.session.ledger) + 1)
    )
    assert {
        "capability_completed",
        "mastery_changed",
        "mission_completed",
    } <= {str(entry.event_type) for entry in result.session.ledger}


def test_mission_enforces_four_capability_executions_per_advance(
    tmp_path: Path,
) -> None:
    repository = SQLiteStudySessionRepository(tmp_path / "catalog.sqlite3")
    tools = MapTools()
    session = _mission_session(
        "turn-bound",
        tuple(_milestone(index) for index in range(8)),
    )
    repository.save(session)

    result = LangGraphMissionRunner(tools, repository).run(
        AdvanceStudyMissionRequest(session.identifier)
    )

    assert len(tools.calls) == 4
    assert result.session.action_count == 4
    assert result.session.status is MissionStatus.ACTIVE


def test_mission_enforces_sixty_four_capability_executions_per_session(
    tmp_path: Path,
) -> None:
    repository = SQLiteStudySessionRepository(tmp_path / "catalog.sqlite3")
    tools = MapTools()
    session = replace(
        _mission_session(
            "session-bound",
            tuple(_milestone(index) for index in range(2)),
        ),
        action_count=64,
    )
    repository.save(session)

    result = LangGraphMissionRunner(tools, repository).run(
        AdvanceStudyMissionRequest(session.identifier)
    )

    assert tools.calls == []
    assert result.session.action_count == 64
    assert result.session.status is MissionStatus.FAILED


def test_optional_artifact_failure_keeps_its_persisted_action_count(
    tmp_path: Path,
) -> None:
    repository = SQLiteStudySessionRepository(tmp_path / "catalog.sqlite3")
    tools = BrokenSummaryTools()
    milestone = StudyMilestone(
        identifier="summary",
        kind=MilestoneKind.LEARN,
        title="Summary",
        objective_id=None,
        capability="summarize_document",
        status=MilestoneStatus.ACTIVE,
    )
    session = _mission_session("optional-failure", (milestone,))
    repository.save(session)

    result = LangGraphMissionRunner(tools, repository).run(
        AdvanceStudyMissionRequest(session.identifier)
    )

    assert result.session.status is MissionStatus.COMPLETED
    assert result.session.action_count == 1
    assert any(event.event_type == "failure" for event in result.session.trace)


def test_mission_api_exposes_routes_and_covers_resume_completion_journey(
    tmp_path: Path,
) -> None:
    settings = Settings(
        catalog_db_path=tmp_path / "catalog.sqlite3",
        document_library_path=tmp_path / "documents",
        vector_db_path=tmp_path / "vectors",
    )
    app = create_app(settings)
    repository = app.state.container.study_session_repository()
    app.state.container.start_study_session_use_case.override(
        providers.Object(_FakeStart(repository))
    )
    app.state.container.advance_study_session_use_case.override(
        providers.Object(_FakeAdvance(repository))
    )

    paths = app.openapi()["paths"]
    assert "get" in paths["/agent/sessions"]
    assert "post" in paths["/agent/sessions/{session_id}/advance"]
    assert "post" in paths["/agent/sessions/{session_id}/complete"]

    async def journey() -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            started = await client.post(
                "/agent/sessions",
                json={
                    "document_id": "document-1",
                    "goal": "Learn the idea",
                    "learner_level": "intermediate",
                    "mode": "guided",
                    "target_minutes": 30,
                },
            )
            assert started.status_code == 201
            session_id = started.json()["session_id"]

            advanced = await client.post(
                f"/agent/sessions/{session_id}/advance", json={}
            )
            assert advanced.status_code == 200
            assert advanced.json()["status"] == "awaiting_learner"

            resumed = await client.get(f"/agent/sessions/{session_id}")
            assert resumed.status_code == 200
            assert resumed.json()["pending_interaction"]["question"]

            completed = await client.post(f"/agent/sessions/{session_id}/complete")
            assert completed.status_code == 200
            assert completed.json()["status"] == "completed"

    asyncio.run(journey())


def test_domain_and_application_have_no_framework_imports() -> None:
    root = Path(__file__).parents[1] / "src" / "scholar_agent"
    forbidden = re.compile(
        r"^\s*(?:from|import)\s+(?:langchain|langgraph|fastapi|streamlit|"
        r"openai|ollama|faiss|pymupdf)(?:\.|\s|$)",
        re.MULTILINE | re.IGNORECASE,
    )
    for layer in ("domain", "application"):
        for path in (root / layer).rglob("*.py"):
            assert forbidden.search(path.read_text(encoding="utf-8")) is None, path


class _FakeStart:
    def __init__(self, repository) -> None:
        self._repository = repository

    def execute(self, request: StartStudySessionRequest) -> StudySessionResult:
        session = _mission_session("api-session", (_milestone(0),))
        self._repository.save(session)
        return _result(session)


class _FakeAdvance:
    def __init__(self, repository) -> None:
        self._repository = repository

    def execute(self, request: AdvanceStudyMissionRequest) -> StudySessionResult:
        session = self._repository.get(request.session_id)
        assert session is not None
        pending = PendingLearnerInteraction(
            "objective-1",
            "What is the key idea?",
            citations=session.brief.objectives[0].citations,
        )
        session = replace(
            session,
            status=MissionStatus.AWAITING_LEARNER,
            pending_interaction=pending,
        )
        self._repository.save(session)
        return _result(session)


def _result(session: StudySession) -> StudySessionResult:
    return StudySessionResult(
        session=session,
        progress=tuple(
            objective_progress(item.identifier, session.attempts)
            for item in session.brief.objectives
        ),
        current_objective_id="objective-1",
        activity=None,
    )


def _mission_session(
    identifier: str, milestones: tuple[StudyMilestone, ...]
) -> StudySession:
    document_id = DocumentId("document-1")
    reference = SourceReference(document_id, "chunk-1", 1, "Evidence")
    objective = LearningObjective("objective-1", "One", "One idea", (), (reference,))
    brief = DocumentBrief(document_id, "Brief", (objective,), (), (), ())
    now = datetime.now(UTC)
    return StudySession(
        identifier=identifier,
        document_id=document_id,
        goal="Learn",
        learner_level=LearnerLevel.INTERMEDIATE,
        mode=StudyMode.GUIDED,
        target_minutes=30,
        brief=brief,
        plan=StudyPlan("Learn", ("objective-1",), (reference,)),
        milestones=milestones,
        created_at=now,
        updated_at=now,
    )


def _milestone(index: int) -> StudyMilestone:
    return StudyMilestone(
        identifier=f"map-{index}",
        kind=MilestoneKind.ORIENT,
        title="Map",
        objective_id=None,
        capability="build_document_map",
        status=MilestoneStatus.ACTIVE if index == 0 else MilestoneStatus.PENDING,
        citations=(),
    )
