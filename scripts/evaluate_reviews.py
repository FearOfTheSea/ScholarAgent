"""Run deterministic Phase 2 learner-model and review-memory scenarios."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

from scholar_agent.application.dtos.mission import AdvanceStudyMissionRequest
from scholar_agent.application.dtos.tutor import (
    BuildDocumentBriefResult,
    StartStudySessionRequest,
)
from scholar_agent.application.services.knowledge_tracing import KnowledgeTracingService
from scholar_agent.application.services.mission_observations import (
    MissionObservationSyncService,
)
from scholar_agent.application.services.mission_state import MissionStateService
from scholar_agent.application.services.request_validation_service import (
    RequestValidationService,
)
from scholar_agent.application.services.review_scheduler import ReviewScheduler
from scholar_agent.application.use_cases.export_learner_profile import (
    ExportLearnerProfileUseCase,
)
from scholar_agent.application.use_cases.get_review_queue import GetReviewQueueUseCase
from scholar_agent.application.use_cases.import_learner_profile import (
    ImportLearnerProfileRequest,
    ImportLearnerProfileUseCase,
)
from scholar_agent.application.use_cases.start_review_mission import (
    StartReviewMissionUseCase,
)
from scholar_agent.domain.entities.learner_profile import (
    ConceptEquivalenceLink,
    ConceptFingerprint,
    EquivalenceDecision,
    EvidenceObservation,
    LearnerProfile,
    ObservationModality,
)
from scholar_agent.domain.entities.study_session import (
    DocumentBrief,
    LearnerLevel,
    LearningObjective,
    StudyMode,
    StudySession,
)
from scholar_agent.domain.value_objects.citation_identity import CitationIdentity
from scholar_agent.domain.value_objects.document_id import DocumentId
from scholar_agent.infrastructure.adapters.langgraph_mission_runner import (
    LangGraphMissionRunner,
)
from scholar_agent.infrastructure.adapters.sqlite_learner_profile_repository import (
    SQLiteLearnerProfileRepository,
)
from scholar_agent.infrastructure.adapters.sqlite_study_session_repository import (
    SQLiteStudySessionRepository,
)

NOW = datetime(2026, 8, 1, 12, tzinfo=UTC)


class EvaluationTools:
    """Deterministic cited capability fake for the review mission scenario."""

    def __init__(self) -> None:
        self.scores = [3, 3, 3]
        self.calls: list[tuple[str, str]] = []
        self.scope_violations = 0

    def execute(
        self, tool_name: str, arguments: Mapping[str, object]
    ) -> Mapping[str, object]:
        document_id = str(arguments.get("document_id", ""))
        self.calls.append((tool_name, document_id))
        if document_id != "document-1":
            self.scope_violations += 1
        citation = _reference_payload()
        if tool_name == "build_document_map":
            return {"document_id": str(arguments.get("document_id", ""))}
        if tool_name == "explain_concept":
            return {
                "objective_id": str(arguments.get("objective_id", "objective-1")),
                "explanation": "The cited concept connects the key terms.",
                "check_question": "How do the key terms connect?",
                "citations": [citation],
            }
        if tool_name == "semantic_search":
            return {"chunks": [{**citation, "content": "Evidence."}]}
        if tool_name == "assess_learner_response":
            return {
                "objective_id": str(arguments.get("objective_id", "objective-1")),
                "score": self.scores.pop(0) if self.scores else 3,
                "feedback": "Connect the cited terms.",
                "missing_concepts": [],
                "next_question": "How do the key terms connect?",
                "citations": [citation],
            }
        if tool_name == "summarize_document":
            return {"summary": "The cited idea.", "citations": [citation]}
        if tool_name == "generate_quiz":
            return {
                "questions": [
                    {
                        "prompt": "What is the idea?",
                        "answer": "The cited idea.",
                        "citations": [citation],
                    }
                ]
            }
        if tool_name == "generate_flashcards":
            return {
                "cards": [
                    {
                        "front": "Idea",
                        "back": "The cited idea.",
                        "citations": [citation],
                    }
                ]
            }
        if tool_name == "citation_lookup":
            return {"citations": [citation]}
        raise AssertionError(f"Unexpected capability: {tool_name}")


class BriefProvider:
    """Application boundary substitute for one cited document brief."""

    def __init__(self, brief: DocumentBrief) -> None:
        self.brief = brief

    def execute(self, document_id: DocumentId) -> BuildDocumentBriefResult:
        return BuildDocumentBriefResult(self.brief, cached=False)


def _brief() -> DocumentBrief:
    document_id = DocumentId("document-1")
    citation = CitationIdentity(document_id, "chunk-1", 1)
    from scholar_agent.domain.value_objects.source_reference import SourceReference

    reference = SourceReference(
        document_id, citation.chunk_id, citation.page_number, "Evidence"
    )
    objective = LearningObjective(
        "objective-1", "The central idea", "One cited idea.", (), (reference,)
    )
    return DocumentBrief(document_id, "A cited brief.", (objective,), (), (), ())


def _reference_payload() -> dict[str, object]:
    return {
        "document_id": "document-1",
        "chunk_id": "chunk-1",
        "page_number": 1,
        "excerpt": "Evidence",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/evaluation/review_phase2_report.json")
    )
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="scholar-agent-phase2-") as directory:
        results = _run_scenarios(Path(directory))
    report = _report(results)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    passed = sum(item["passed"] for item in results.values())
    print(f"Phase 2 review evaluator: {passed}/{len(results)} scenarios passed")
    print(f"Report: {arguments.output}")
    return 0 if passed == len(results) and all(report["release_gates"].values()) else 1


def _run_scenarios(root: Path) -> dict[str, dict[str, Any]]:
    return {
        "mission_observation_recorded": _mission_observation(root / "mission"),
        "idempotent_resync": _idempotent_resync(root / "resync"),
        "queue_order_and_target_date": _queue_order(root / "queue"),
        "stale_confidence_decay": _stale_decay(),
        "transfer_weighting": _transfer_weighting(),
        "profile_round_trip": _round_trip(root / "round-trip"),
        "deletion_cascade_and_detachment": _deletion(root / "deletion"),
        "equivalence_consent": _equivalence(),
        "start_review_dispatch": _start_review(root / "review"),
    }


def _mission_observation(root: Path) -> dict[str, Any]:
    fixture = _mission_fixture(root)
    fixture.runner.run(AdvanceStudyMissionRequest(fixture.session_id))
    fixture.runner.run(AdvanceStudyMissionRequest(fixture.session_id, "answer"))
    observations = fixture.profiles.list_observations("local-default")
    result = _scenario(
        bool(observations),
        privacy=_check(
            bool(observations) and _redacted(observations[0]),
            "observation contains only redacted fields",
        ),
        document_isolation=_check(
            bool(observations)
            and _tool_calls_isolated(fixture.tools)
            and {item.document_id.value for item in observations} == {"document-1"},
            "mission tool calls and evidence stay on the selected document",
        ),
    )
    fixture.close()
    return result


def _idempotent_resync(root: Path) -> dict[str, Any]:
    fixture = _mission_fixture(root)
    fixture.runner.run(AdvanceStudyMissionRequest(fixture.session_id))
    with patch.object(
        fixture.profiles,
        "append_observation",
        side_effect=RuntimeError("simulated profile write failure"),
    ):
        fixture.runner.run(AdvanceStudyMissionRequest(fixture.session_id, "answer"))
    missing = not fixture.profiles.list_observations("local-default")
    session = fixture.sessions.get(fixture.session_id)
    assert session is not None
    repaired = fixture.sync.sync_profile("local-default")
    repeated = fixture.sync.sync_profile("local-default")
    result = _scenario(
        missing and repaired == 1 and repeated == 0,
        privacy=_check(
            len(fixture.profiles.list_observations("local-default")) == 1,
            "resync repairs one redacted observation without duplication",
        ),
    )
    fixture.close()
    return result


def _queue_order(root: Path) -> dict[str, Any]:
    profile = LearnerProfile(
        "local-default", "Local learner", date(2026, 8, 5), NOW, NOW
    )
    first = _fingerprint("document-1", "Older concept")
    second = _fingerprint("document-1", "Recent concept")
    observations = (
        _observation(profile.identifier, first, observed_at=NOW - timedelta(days=90)),
        *tuple(
            _observation(profile.identifier, second, observed_at=NOW, suffix=str(index))
            for index in range(4)
        ),
    )
    queue = ReviewScheduler(clock=lambda: NOW).queue(profile, observations, as_of=NOW)
    repeat = ReviewScheduler(clock=lambda: NOW).queue(profile, observations, as_of=NOW)
    result = _scenario(
        len(queue) == 2 and queue == repeat,
        scheduling=_check(
            queue[0].fingerprint == first
            and any("target_date" in item.reason_codes for item in queue),
            "due ordering and target-date cap are deterministic",
        ),
    )
    return result


def _stale_decay() -> dict[str, Any]:
    fingerprint = _fingerprint("document-1", "Stale concept")
    fresh = _observation("local-default", fingerprint, observed_at=NOW)
    stale = _observation(
        "local-default",
        fingerprint,
        observed_at=NOW - timedelta(days=90),
        suffix="stale",
    )
    tracing = KnowledgeTracingService(lambda: NOW)
    fresh_estimate = tracing.estimate((fresh,), as_of=NOW)
    stale_estimate = tracing.estimate((stale,), as_of=NOW)
    return _scenario(
        stale_estimate.confidence < fresh_estimate.confidence,
        decay=_check(
            stale_estimate.confidence < fresh_estimate.confidence,
            "confidence decays with age",
        ),
    )


def _transfer_weighting() -> dict[str, Any]:
    fingerprint = _fingerprint("document-1", "Transfer concept")
    tracing = KnowledgeTracingService(lambda: NOW)
    recall = tracing.estimate(
        (_observation("local-default", fingerprint, suffix="recall"),), as_of=NOW
    )
    transfer = tracing.estimate(
        (
            _observation(
                "local-default",
                fingerprint,
                modality=ObservationModality.TRANSFER,
                suffix="transfer",
            ),
        ),
        as_of=NOW,
    )
    return _scenario(
        transfer.confidence > recall.confidence,
        transfer_weighting=_check(
            transfer.confidence > recall.confidence,
            "identical transfer evidence outweighs recall",
        ),
    )


def _round_trip(root: Path) -> dict[str, Any]:
    repository = SQLiteLearnerProfileRepository(root / "profiles.sqlite3")
    profile = repository.get_or_create_default(NOW)
    observation = _observation(profile.identifier, _fingerprint("document-1"))
    repository.append_observation(observation)
    export = ExportLearnerProfileUseCase(repository).execute(profile.identifier)
    repository.delete_profile(profile.identifier)
    ImportLearnerProfileUseCase(repository).execute(
        ImportLearnerProfileRequest(profile.identifier, export)
    )
    restored = ExportLearnerProfileUseCase(repository).execute(profile.identifier)
    passed = restored == export and _redacted_payload(export)
    repository.close()
    return _scenario(
        passed,
        privacy=_check(passed, "export excludes raw content"),
        round_trip=_check(passed, "export/import is exact"),
    )


def _deletion(root: Path) -> dict[str, Any]:
    sessions = SQLiteStudySessionRepository(root / "sessions.sqlite3")
    profiles = SQLiteLearnerProfileRepository(root / "profiles.sqlite3", sessions)
    profile = profiles.get_or_create_default(NOW)
    session = replace(_session_for_profile(), learner_profile_id=profile.identifier)
    sessions.save(session)
    detached = profiles.delete_profile(profile.identifier)
    profiles.close()
    sessions.close()
    reopened_sessions = SQLiteStudySessionRepository(root / "sessions.sqlite3")
    reopened_profiles = SQLiteLearnerProfileRepository(
        root / "profiles.sqlite3", reopened_sessions
    )
    restored = reopened_sessions.get(session.identifier)
    passed = (
        detached == 1
        and reopened_profiles.get_profile(profile.identifier) is None
        and not reopened_profiles.list_observations(profile.identifier)
        and restored is not None
        and restored.learner_profile_id is None
    )
    reopened_profiles.close()
    reopened_sessions.close()
    return _scenario(
        passed, deletion=_check(passed, "profile rows cascade and missions detach")
    )


def _equivalence() -> dict[str, Any]:
    profile = LearnerProfile("local-default", "Local learner", None, NOW, NOW)
    left = _fingerprint("document-1", "Shared concept")
    right = _fingerprint("document-2", "Shared concept")
    observations = (
        _observation(profile.identifier, left),
        _observation(profile.identifier, right, suffix="right"),
    )
    rejected = ConceptEquivalenceLink(
        left, right, EquivalenceDecision.REJECTED, NOW, profile.identifier
    )
    accepted = replace(rejected, decision=EquivalenceDecision.ACCEPTED)
    scheduler = ReviewScheduler(clock=lambda: NOW)
    rejected_queue = scheduler.queue(profile, observations, (rejected,), NOW)
    accepted_queue = scheduler.queue(profile, observations, (accepted,), NOW)
    passed = {item.observation_count for item in rejected_queue} == {1} and {
        item.observation_count for item in accepted_queue
    } == {2}
    return _scenario(
        passed, equivalence_consent=_check(passed, "only accepted links pool history")
    )


def _start_review(root: Path) -> dict[str, Any]:
    fixture = _mission_fixture(root)
    profile = fixture.profiles.get_profile("local-default")
    assert profile is not None
    objective = _brief().objectives[0]
    fingerprint = ConceptFingerprint.from_descriptor(
        objective.citations[0].document_id, objective.title, objective.description
    )
    fixture.profiles.append_observation(
        _observation("local-default", fingerprint, objective_id=objective.identifier)
    )
    queue = GetReviewQueueUseCase(
        fixture.profiles, ReviewScheduler(clock=lambda: NOW), fixture.sync
    )
    start = StartReviewMissionUseCase(
        queue,
        fixture.start_use_case,
    )
    result = start.execute("local-default", fingerprint.value, NOW)
    passed = (
        result.session.document_id == DocumentId("document-1")
        and result.session.plan is not None
        and result.session.plan.objective_ids == (objective.identifier,)
        and result.session.learner_profile_id == "local-default"
    )
    fixture.close()
    return _scenario(
        passed,
        document_isolation=_check(
            passed, "review dispatch resolves one document and objective"
        ),
    )


class _MissionFixture:
    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.sessions = SQLiteStudySessionRepository(root / "sessions.sqlite3")
        self.profiles = SQLiteLearnerProfileRepository(
            root / "profiles.sqlite3", self.sessions
        )
        self.profiles.get_or_create_default(NOW)
        self.sync = MissionObservationSyncService(
            self.profiles, self.sessions, lambda: NOW
        )
        self.state = MissionStateService(self.sessions, observation_sync=self.sync)
        from scholar_agent.application.use_cases.start_study_session import (
            StartStudySessionUseCase,
        )

        self.start_use_case = StartStudySessionUseCase(
            BriefProvider(_brief()),
            self.sessions,
            RequestValidationService(),
            state_service=self.state,
            profile_repository=self.profiles,
        )
        started = self.start_use_case.execute(
            StartStudySessionRequest(
                DocumentId("document-1"), learner_profile_id="local-default"
            )
        )
        self.session_id = started.session.identifier
        self.tools = EvaluationTools()
        self.runner = LangGraphMissionRunner(
            self.tools, self.sessions, state_service=self.state
        )

    def close(self) -> None:
        self.profiles.close()
        self.sessions.close()


def _mission_fixture(root: Path) -> _MissionFixture:
    return _MissionFixture(root)


def _scenario(passed: bool, **checks: dict[str, object]) -> dict[str, Any]:
    return {"passed": passed, "checks": checks}


def _check(passed: bool, reason: str) -> dict[str, object]:
    return {"applicable": True, "passed": passed, "reason": reason}


def _not_applicable(reason: str) -> dict[str, object]:
    return {"applicable": False, "passed": None, "reason": reason}


def _report(results: dict[str, dict[str, Any]]) -> dict[str, object]:
    gate_names = {
        "privacy_redaction": "privacy",
        "deterministic_scheduling": "scheduling",
        "decay": "decay",
        "transfer_weighting": "transfer_weighting",
        "round_trip_fidelity": "round_trip",
        "deletion_completeness": "deletion",
        "equivalence_consent": "equivalence_consent",
        "document_isolation": "document_isolation",
    }
    details: dict[str, object] = {}
    release: dict[str, bool] = {}
    for gate, key in gate_names.items():
        release[gate], details[gate] = _aggregate(results, key)
    return {
        "report_version": 1,
        "phase": "Phase 2 — Durable Learner Model and Review Memory",
        "generated_at": datetime.now(UTC).isoformat(),
        "release_gates": release,
        "metrics": {"gate_details": details},
        "scenarios": results,
    }


def _aggregate(
    results: dict[str, dict[str, Any]], key: str
) -> tuple[bool, dict[str, object]]:
    applicable: list[str] = []
    failures: dict[str, object] = {}
    for name, result in results.items():
        value = result.get("checks", {}).get(key)
        if not isinstance(value, dict) or value.get("applicable") is not True:
            continue
        applicable.append(name)
        if value.get("passed") is not True:
            failures[name] = value.get("reason", "check failed")
    return bool(applicable) and not failures, {
        "applicable_scenarios": applicable,
        "applicable_count": len(applicable),
        "failures": failures,
    }


def _fingerprint(document_id: str, title: str = "Concept") -> ConceptFingerprint:
    return ConceptFingerprint.from_descriptor(
        DocumentId(document_id), title, "A durable learning concept."
    )


def _observation(
    profile_id: str,
    fingerprint: ConceptFingerprint,
    *,
    objective_id: str = "objective-1",
    score: int = 3,
    modality: ObservationModality = ObservationModality.RECALL,
    observed_at: datetime = NOW,
    suffix: str = "",
) -> EvidenceObservation:
    return EvidenceObservation.for_review(
        profile_id,
        fingerprint,
        objective_id,
        modality,
        score,
        2,
        (CitationIdentity(fingerprint.document_id, "chunk-1", 1),),
        observed_at,
        session_id=suffix or None,
    )


def _session_for_profile() -> StudySession:
    brief = _brief()
    return StudySession(
        identifier="detached-session",
        document_id=brief.document_id,
        goal="Learn",
        learner_level=LearnerLevel.INTERMEDIATE,
        mode=StudyMode.GUIDED,
        target_minutes=30,
        brief=brief,
        created_at=NOW,
        updated_at=NOW,
    )


def _redacted(observation: EvidenceObservation) -> bool:
    return _redacted_payload(asdict(observation))


def _tool_calls_isolated(tools: EvaluationTools) -> bool:
    return (
        bool(tools.calls)
        and tools.scope_violations == 0
        and all(document_id == "document-1" for _, document_id in tools.calls)
    )


def _redacted_payload(payload: object) -> bool:
    forbidden = {
        "response",
        "feedback",
        "prompt",
        "model_output",
        "reference_answer",
        "excerpt",
        "source_text",
        "turns",
        "pdf_content",
    }
    if isinstance(payload, dict):
        return not forbidden.intersection(payload) and all(
            _redacted_payload(value) for value in payload.values()
        )
    if isinstance(payload, list):
        return all(_redacted_payload(value) for value in payload)
    return True


if __name__ == "__main__":
    raise SystemExit(main())
