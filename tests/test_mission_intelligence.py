"""Focused Phase 1 ledger, insight, export, API, and UI contracts."""

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from httpx import ASGITransport, AsyncClient

from scholar_agent.application.services.mission_insights import MissionInsightsService
from scholar_agent.application.services.mission_ledger import MissionLedgerService
from scholar_agent.application.services.mission_state import MissionStateService
from scholar_agent.application.use_cases.export_mission_record import (
    ExportMissionRecordUseCase,
)
from scholar_agent.application.use_cases.get_mission_insights import (
    GetMissionInsightsUseCase,
)
from scholar_agent.application.use_cases.verify_mission_ledger import (
    VerifyMissionLedgerUseCase,
)
from scholar_agent.config.settings import Settings
from scholar_agent.domain.entities.mission_ledger import (
    MissionLedgerEventType,
    MissionStateProjection,
    compute_ledger_digest,
)
from scholar_agent.domain.entities.study_session import (
    DocumentBrief,
    LearnerAttempt,
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
)
from scholar_agent.domain.value_objects.document_id import DocumentId
from scholar_agent.infrastructure.adapters.sqlite_study_session_repository import (
    SQLiteStudySessionRepository,
    _session_payload,
)
from scholar_agent.presentation.api.main import create_app


def test_digest_excludes_human_text_and_timestamps() -> None:
    projection = MissionStateProjection(
        "active", "milestone-1", None, 1, 0, 0, 0, (), "milestone-1"
    )
    arguments = {
        "previous_digest": "",
        "sequence": 1,
        "event_type": MissionLedgerEventType.MISSION_STARTED,
        "projection": projection,
        "objective_id": None,
        "capability": None,
        "citations": (),
    }
    first = compute_ledger_digest(**arguments)
    second = compute_ledger_digest(**arguments)
    assert first == second
    assert len(first) == 64


def test_repeated_checkpoint_with_transition_key_is_idempotent(tmp_path: Path) -> None:
    repository = SQLiteStudySessionRepository(tmp_path / "catalog.sqlite3")
    session = _session()
    state = MissionStateService(repository)
    first = state.checkpoint(
        session,
        "start",
        "Mission started.",
        transition_key="start-once",
    )
    second = state.checkpoint(
        first,
        "start",
        "Mission started again.",
        transition_key="start-once",
    )
    assert len(second.ledger) == 1
    assert second.ledger[0].summary == "Mission started."
    assert len(second.trace) == 1
    repository.close()


def test_ledger_capacity_fails_recoverably_without_truncating() -> None:
    session = _session()
    ledger = MissionLedgerService()
    for index in range(512):
        session = replace(
            session,
            action_count=index,
            ledger=session.ledger,
        )
        session = ledger.append(
            session,
            MissionLedgerEventType.WAITING_FOR_LEARNER,
            "Waiting.",
            transition_key=f"wait-{index}",
        )
    failed = ledger.append(
        session,
        MissionLedgerEventType.WAITING_FOR_LEARNER,
        "Waiting.",
        transition_key="wait-over-capacity",
    )
    assert failed.status is MissionStatus.FAILED
    assert len(failed.ledger) == 512


def test_v2_reads_empty_ledger_and_next_save_emits_v3(tmp_path: Path) -> None:
    repository = SQLiteStudySessionRepository(tmp_path / "catalog.sqlite3")
    session = _session()
    payload = _session_payload(session)
    payload["schema_version"] = 2
    payload.pop("ledger", None)
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
    restored = repository.get(session.identifier)
    assert restored is not None
    assert restored.ledger == ()
    repository.save(restored)
    row = repository._connection.execute(  # type: ignore[attr-defined]
        "SELECT payload FROM study_sessions WHERE session_id = ?",
        (session.identifier,),
    ).fetchone()
    assert json.loads(row[0])["schema_version"] == 3
    repository.close()


def test_insights_calculate_formulas_and_signals(tmp_path: Path) -> None:
    repository = SQLiteStudySessionRepository(tmp_path / "catalog.sqlite3")
    session = _session()
    reference = session.brief.objectives[0].citations[0]
    second_objective = LearningObjective(
        "objective-2", "Two", "Another idea", (), (reference,)
    )
    session = replace(
        session,
        brief=replace(
            session.brief,
            objectives=session.brief.objectives + (second_objective,),
        ),
        plan=StudyPlan("Learn", ("objective-1", "objective-2"), (reference,)),
    )
    now = datetime.now(UTC)
    attempts = (
        LearnerAttempt("objective-1", "first", 2, "Good", (), (reference,), now),
        LearnerAttempt("objective-1", "second", 3, "Good", (), (reference,), now),
    )
    session = replace(session, attempts=attempts, action_count=3)
    state = MissionStateService(repository)
    session = state.checkpoint(
        session,
        "capability",
        "Completed map.",
        capability="build_document_map",
        citations=(reference,),
        transition_key="map",
    )
    session = state.checkpoint(
        session,
        "remediation",
        "Remediation started.",
        objective_id="objective-1",
        transition_key="remediation",
    )
    insights = MissionInsightsService().calculate(session)
    assert insights.progress_percent == 50
    assert insights.mastery_counts["mastered"] == 1
    assert insights.mastery_counts["unseen"] == 1
    assert insights.assessment_count == 2
    assert insights.first_pass_proficiency_rate == 1
    assert insights.remediation_cycles == 1
    assert insights.evidence_coverage == 1
    assert insights.action_budget_used == 3
    assert insights.action_budget_remaining == 61
    assert "unassessed_objectives" in insights.signals
    repository.close()


def test_insights_use_null_for_undefined_denominators() -> None:
    session = replace(_session(), milestones=(), plan=None)
    insights = MissionInsightsService().calculate(session)
    assert insights.progress_percent is None
    assert insights.first_pass_proficiency_rate is None
    assert insights.evidence_coverage is None


def test_tampered_current_ledger_is_detected_on_load_and_verify(
    tmp_path: Path,
) -> None:
    repository = SQLiteStudySessionRepository(tmp_path / "catalog.sqlite3")
    state = MissionStateService(repository)
    session = state.checkpoint(
        _session(),
        "start",
        "Mission started.",
        transition_key="start",
    )
    payload = _session_payload(session)
    payload["ledger"][0]["current_digest"] = "0" * 64  # type: ignore[index]
    with repository._connection:  # type: ignore[attr-defined]
        repository._connection.execute(  # type: ignore[attr-defined]
            "UPDATE study_sessions SET payload = ? WHERE session_id = ?",
            (json.dumps(payload), session.identifier),
        )
    verification = VerifyMissionLedgerUseCase(repository).execute(session.identifier)
    assert not verification.valid
    assert verification.reason
    try:
        repository.get(session.identifier)
    except RuntimeError as error:
        assert "ledger" in str(error).lower()
    else:
        raise AssertionError("A tampered ledger must fail current-schema loading.")
    repository.close()


def test_redacted_export_contains_no_forbidden_raw_fields(tmp_path: Path) -> None:
    repository = SQLiteStudySessionRepository(tmp_path / "catalog.sqlite3")
    session = _session()
    reference = session.brief.objectives[0].citations[0]
    session = replace(
        session,
        pending_interaction=PendingLearnerInteraction(
            "objective-1",
            "Private question",
            reference_answer="Private answer",
            citations=(reference,),
        ),
    )
    state = MissionStateService(repository)
    session = state.checkpoint(
        session,
        "wait",
        "Waiting for the learner response.",
        objective_id="objective-1",
        citations=(reference,),
        transition_key="wait",
    )
    repository.save(session)
    insights = GetMissionInsightsUseCase(repository)
    record = ExportMissionRecordUseCase(repository, insights).execute(
        session.identifier
    )
    keys = set(_all_keys(record))
    assert not keys.intersection(
        {"response", "reference_answer", "prompt", "answer", "model_output", "excerpt"}
    )
    assert record["ledger"][0]["citations"][0] == {
        "document_id": "document-1",
        "chunk_id": "chunk-1",
        "page_number": 1,
    }
    repository.close()


def test_intelligence_endpoints_are_additive_and_return_redacted_record(
    tmp_path: Path,
) -> None:
    settings = Settings(
        catalog_db_path=tmp_path / "catalog.sqlite3",
        document_library_path=tmp_path / "documents",
        vector_db_path=tmp_path / "vectors",
    )
    app = create_app(settings)
    repository = app.state.container.study_session_repository()
    repository.save(_session())
    paths = app.openapi()["paths"]
    assert "/agent/sessions/{session_id}/insights" in paths
    assert "/agent/sessions/{session_id}/record" in paths
    assert "/agent/sessions/{session_id}/record/verify" in paths

    async def journey() -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            insights = await client.get("/agent/sessions/session-1/insights")
            record = await client.get("/agent/sessions/session-1/record")
            verified = await client.post("/agent/sessions/session-1/record/verify")
            assert insights.status_code == 200
            assert record.status_code == 200
            assert verified.status_code == 200
            assert verified.json()["valid"] is True
            assert "excerpt" not in json.dumps(record.json())

    asyncio.run(journey())
    repository.close()


def test_mission_intelligence_panel_is_present_in_fresh_process_ui() -> None:
    source = (
        Path(__file__).parents[1]
        / "src"
        / "scholar_agent"
        / "presentation"
        / "web"
        / "app.py"
    ).read_text(encoding="utf-8")
    assert "Mission Intelligence" in source
    assert "Decision timeline" in source


def _session() -> StudySession:
    document_id = DocumentId("document-1")
    reference = SourceReference(document_id, "chunk-1", 1, "Evidence")
    objective = LearningObjective("objective-1", "One", "One idea", (), (reference,))
    brief = DocumentBrief(document_id, "Brief", (objective,), (), (), ())
    now = datetime.now(UTC)
    return StudySession(
        identifier="session-1",
        document_id=document_id,
        goal="Learn",
        learner_level=LearnerLevel.INTERMEDIATE,
        mode=StudyMode.GUIDED,
        target_minutes=30,
        brief=brief,
        plan=StudyPlan("Learn", ("objective-1",), (reference,)),
        milestones=(
            StudyMilestone(
                "milestone-1",
                MilestoneKind.ORIENT,
                "Map",
                None,
                "build_document_map",
                MilestoneStatus.COMPLETED,
                (reference,),
            ),
            StudyMilestone(
                "milestone-2",
                MilestoneKind.PRACTICE,
                "Practice",
                "objective-1",
                "assess_learner_response",
                MilestoneStatus.PENDING,
                (reference,),
            ),
        ),
        status=MissionStatus.ACTIVE,
        created_at=now,
        updated_at=now,
    )


def _all_keys(value: object):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)
