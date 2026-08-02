"""Run the deterministic Phase 1 Study Mission release scenarios."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scholar_agent.application.dtos.mission import AdvanceStudyMissionRequest
from scholar_agent.application.dtos.tutor import (
    BuildDocumentBriefResult,
    StartStudySessionRequest,
)
from scholar_agent.application.output_ports.tool_executor import IToolExecutor
from scholar_agent.application.services.mission_ledger import MissionLedgerService
from scholar_agent.application.services.request_validation_service import (
    RequestValidationService,
)
from scholar_agent.application.use_cases.start_study_session import (
    StartStudySessionUseCase,
)
from scholar_agent.domain.entities.study_material import (
    FlashcardArtifact,
    QuizArtifact,
    SummaryArtifact,
)
from scholar_agent.domain.entities.study_session import (
    DocumentBrief,
    LearnerLevel,
    LearningObjective,
    MissionStatus,
    SourceReference,
    StudyMode,
    StudySession,
    TutorTurnKind,
)
from scholar_agent.domain.value_objects.document_id import DocumentId
from scholar_agent.infrastructure.adapters.langgraph_mission_runner import (
    LangGraphMissionRunner,
)
from scholar_agent.infrastructure.adapters.sqlite_study_session_repository import (
    SQLiteStudySessionRepository,
    _session_payload,
)


class EvaluationTools(IToolExecutor):
    """Deterministic cited capability provider used by all eight scenarios."""

    def __init__(self, scores: list[int] | None = None) -> None:
        self.scores = list(scores or [3, 3, 3, 3])
        self.calls: list[tuple[str, str]] = []
        self.advance_counts: list[int] = []
        self.scope_violations = 0
        self.fail_summary = False

    def execute(
        self, tool_name: str, arguments: Mapping[str, object]
    ) -> Mapping[str, object]:
        document_id = str(arguments.get("document_id", ""))
        self.calls.append((tool_name, document_id))
        if document_id != "document-1":
            self.scope_violations += 1
        citation = _reference_payload()
        if tool_name == "build_document_map":
            return {"document_id": document_id, "objectives": []}
        if tool_name == "summarize_document":
            if self.fail_summary:
                raise RuntimeError("deterministic optional artifact failure")
            return {
                "summary": "The selected document's central idea.",
                "citations": [citation],
            }
        if tool_name == "explain_concept":
            return {
                "objective_id": str(arguments.get("objective_id", "objective-1")),
                "explanation": "The cited concept connects the key terms.",
                "check_question": "How do the key terms connect?",
                "citations": [citation],
            }
        if tool_name == "semantic_search":
            return {
                "chunks": [
                    {
                        **citation,
                        "content": "Evidence from the selected document.",
                    }
                ]
            }
        if tool_name == "assess_learner_response":
            return {
                "objective_id": str(arguments.get("objective_id", "objective-1")),
                "score": self.scores.pop(0) if self.scores else 3,
                "feedback": "Connect the cited terms clearly.",
                "missing_concepts": ["the relation"],
                "next_question": "How do the key terms connect?",
                "citations": [citation],
            }
        if tool_name == "generate_quiz":
            return {
                "questions": [
                    {
                        "prompt": "What is the central idea?",
                        "answer": "The cited central idea.",
                        "citations": [citation],
                    }
                ]
            }
        if tool_name == "generate_flashcards":
            return {
                "cards": [
                    {
                        "front": "Central idea",
                        "back": "The cited central idea.",
                        "citations": [citation],
                    }
                ]
            }
        raise AssertionError(f"Unexpected capability: {tool_name}")

    def capabilities(self) -> tuple[object, ...]:
        return ()


class BriefProvider:
    """Application boundary substitute for a cached document brief."""

    def __init__(self, brief: DocumentBrief) -> None:
        self.brief = brief

    def execute(self, document_id: DocumentId) -> BuildDocumentBriefResult:
        return BuildDocumentBriefResult(self.brief, cached=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/mission_phase1_report.json"),
    )
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="scholar-agent-phase1-") as directory:
        results = _run_scenarios(Path(directory))
    report = _report(results)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    passed = sum(item["passed"] for item in results.values())
    print(f"Phase 1 mission evaluator: {passed}/{len(results)} scenarios passed")
    print(f"Report: {arguments.output}")
    return 0 if all(report["release_gates"].values()) and passed == len(results) else 1


def _run_scenarios(root: Path) -> dict[str, dict[str, Any]]:
    return {
        "guided_first_pass_proficiency": _scenario_guided(root),
        "remediation_and_recovery": _scenario_remediation(root),
        "optional_artifact_failure": _scenario_optional_failure(root),
        "save_reload_resume": _scenario_resume(root),
        "explicit_completion": _scenario_explicit_completion(root),
        "session_action_budget": _scenario_session_budget(root),
        "legacy_version_two_upgrade": _scenario_v2_upgrade(root),
        "tampered_ledger_detection": _scenario_tampered_ledger(root),
    }


def _scenario_guided(root: Path) -> dict[str, Any]:
    repository, runner, tools = _mission(root / "guided")
    result = _advance(runner, tools, AdvanceStudyMissionRequest("guided"))
    result = _advance(
        runner, tools, AdvanceStudyMissionRequest("guided", "my first answer")
    )
    result = _advance(
        runner, tools, AdvanceStudyMissionRequest("guided", "my second answer")
    )
    result = _advance(runner, tools, AdvanceStudyMissionRequest("guided"))
    session = repository.get("guided")
    assert session is not None
    return _scenario_result(
        result.session.status is MissionStatus.COMPLETED,
        session,
        runner,
        repository,
        tools,
    )


def _scenario_remediation(root: Path) -> dict[str, Any]:
    repository, runner, tools = _mission(root / "remediation", scores=[0, 3, 3, 3])
    _advance(runner, tools, AdvanceStudyMissionRequest("remediation"))
    _advance(runner, tools, AdvanceStudyMissionRequest("remediation", "not yet"))
    _advance(runner, tools, AdvanceStudyMissionRequest("remediation", "recovered"))
    _advance(
        runner, tools, AdvanceStudyMissionRequest("remediation", "recovered again")
    )
    result = _advance(
        runner, tools, AdvanceStudyMissionRequest("remediation", "fully recovered")
    )
    session = repository.get("remediation")
    assert session is not None
    passed = any(entry.event_type == "remediation_started" for entry in session.ledger)
    return _scenario_result(
        passed and result.session.status is not MissionStatus.FAILED,
        session,
        runner,
        repository,
        tools,
    )


def _scenario_optional_failure(root: Path) -> dict[str, Any]:
    repository, runner, tools = _mission(root / "optional")
    tools.fail_summary = True
    result = _advance(runner, tools, AdvanceStudyMissionRequest("optional"))
    session = repository.get("optional")
    assert session is not None
    failed_optional = any(
        entry.event_type == "capability_failed" for entry in session.ledger
    )
    return _scenario_result(
        failed_optional and result.session.status is not MissionStatus.FAILED,
        session,
        runner,
        repository,
        tools,
    )


def _scenario_resume(root: Path) -> dict[str, Any]:
    repository, runner, tools = _mission(root / "resume")
    _advance(runner, tools, AdvanceStudyMissionRequest("resume"))
    before = repository.get("resume")
    assert before is not None
    repository.close()
    reopened = SQLiteStudySessionRepository(root / "resume" / "catalog.sqlite3")
    resumed = reopened.get("resume")
    assert resumed is not None
    resume_tools = EvaluationTools()
    runner = LangGraphMissionRunner(resume_tools, reopened)
    result = _advance(runner, resume_tools, AdvanceStudyMissionRequest("resume"))
    resume_tools.advance_counts = tools.advance_counts + resume_tools.advance_counts
    same_observable_state = (
        before.document_id == resumed.document_id
        and before.action_count == resumed.action_count
        and before.pending_interaction == resumed.pending_interaction
        and len(before.ledger) == len(resumed.ledger)
    )
    return _scenario_result(
        same_observable_state and result.session.action_count >= before.action_count,
        resumed,
        runner,
        reopened,
        resume_tools,
    )


def _scenario_explicit_completion(root: Path) -> dict[str, Any]:
    repository, runner, tools = _mission(root / "complete")
    result = _advance(
        runner, tools, AdvanceStudyMissionRequest("complete", "finish mission")
    )
    session = repository.get("complete")
    assert session is not None
    return _scenario_result(
        result.session.status is MissionStatus.COMPLETED,
        session,
        runner,
        repository,
        tools,
    )


def _scenario_session_budget(root: Path) -> dict[str, Any]:
    repository, runner, tools = _mission(root / "budget")
    session = repository.get("budget")
    assert session is not None
    repository.save(replace(session, action_count=64))
    result = _advance(runner, tools, AdvanceStudyMissionRequest("budget"))
    session = repository.get("budget")
    assert session is not None
    return _scenario_result(
        result.session.status is MissionStatus.FAILED,
        session,
        runner,
        repository,
        tools,
    )


def _scenario_v2_upgrade(root: Path) -> dict[str, Any]:
    repository, _, _ = _mission(root / "v2")
    session = repository.get("v2")
    assert session is not None
    payload = _session_payload(session)
    payload["schema_version"] = 2
    payload.pop("ledger", None)
    with repository._connection:
        repository._connection.execute(
            "UPDATE study_sessions SET payload = ? WHERE session_id = ?",
            (json.dumps(payload), "v2"),
        )
    restored = repository.get("v2")
    assert restored is not None and not restored.ledger
    repository.save(restored)
    row = repository._connection.execute(
        "SELECT payload FROM study_sessions WHERE session_id = ?", ("v2",)
    ).fetchone()
    return _scenario_result(
        json.loads(row[0])["schema_version"] == 4,
        restored,
        None,
        repository,
    )


def _scenario_tampered_ledger(root: Path) -> dict[str, Any]:
    repository, runner, tools = _mission(root / "tampered")
    _advance(runner, tools, AdvanceStudyMissionRequest("tampered"))
    valid_session = repository.get("tampered")
    assert valid_session is not None
    row = repository._connection.execute(
        "SELECT payload FROM study_sessions WHERE session_id = ?", ("tampered",)
    ).fetchone()
    payload = json.loads(row[0])
    payload["ledger"][0]["current_digest"] = "0" * 64
    with repository._connection:
        repository._connection.execute(
            "UPDATE study_sessions SET payload = ? WHERE session_id = ?",
            (json.dumps(payload), "tampered"),
        )
    try:
        repository.get("tampered")
    except RuntimeError as error:
        result = _scenario_result(
            "ledger" in str(error).lower() or "digest" in str(error).lower(),
            valid_session,
            runner,
            None,
            tools,
        )
        result["ledger_verified"] = False
        result["evidence_integrity"] = _not_applicable(
            "The tampered snapshot is intentionally not evaluated for factual output."
        )
        repository.close()
        result["status"] = "tamper detected"
        return result
    repository.close()
    result = _scenario_result(False, valid_session, runner, None, tools)
    result["ledger_verified"] = False
    result["evidence_integrity"] = _not_applicable(
        "The tampered snapshot is intentionally not evaluated for factual output."
    )
    result["status"] = "tamper not detected"
    return result


def _mission(
    directory: Path, scores: list[int] | None = None
) -> tuple[SQLiteStudySessionRepository, LangGraphMissionRunner, EvaluationTools]:
    directory.mkdir(parents=True, exist_ok=True)
    repository = SQLiteStudySessionRepository(directory / "catalog.sqlite3")
    tools = EvaluationTools(scores)
    brief = _brief()
    start = StartStudySessionUseCase(
        BriefProvider(brief),
        repository,
        RequestValidationService(),
    )
    start.execute(
        StartStudySessionRequest(
            DocumentId("document-1"),
            goal="Understand the cited idea.",
            learner_level=LearnerLevel.INTERMEDIATE,
            mode=StudyMode.GUIDED,
            target_minutes=30,
        )
    )
    session = repository.list()[0]
    session_id = session.identifier
    if session_id != directory.name:
        session = session.__class__(
            identifier=directory.name,
            document_id=session.document_id,
            goal=session.goal,
            learner_level=session.learner_level,
            mode=session.mode,
            target_minutes=session.target_minutes,
            brief=session.brief,
            attempts=session.attempts,
            turns=session.turns,
            created_at=session.created_at,
            updated_at=session.updated_at,
            status=session.status,
            plan=session.plan,
            milestones=session.milestones,
            artifacts=session.artifacts,
            pending_interaction=session.pending_interaction,
            trace=session.trace,
            ledger=session.ledger,
            action_count=session.action_count,
            completed_at=session.completed_at,
        )
        repository.delete(session_id)
        repository.save(session)
    runner = LangGraphMissionRunner(tools, repository)
    return repository, runner, tools


def _brief() -> DocumentBrief:
    document_id = DocumentId("document-1")
    reference = SourceReference(document_id, "chunk-1", 1, "Evidence")
    objective = LearningObjective(
        "objective-1", "The central idea", "One cited idea.", (), (reference,)
    )
    return DocumentBrief(document_id, "A cited brief.", (objective,), (), (), ())


def _scenario_result(
    passed: bool,
    session: StudySession,
    runner: LangGraphMissionRunner | None,
    repository: SQLiteStudySessionRepository | None,
    tools: EvaluationTools | None = None,
) -> dict[str, Any]:
    ledger = session.ledger
    verification = MissionLedgerService.verify(session)
    document_isolation = _document_isolation(session, tools)
    result = {
        "passed": passed,
        "status": session.status.value,
        "action_count": session.action_count,
        "ledger_verified": verification.valid,
        "evidence_integrity": _evidence_integrity(session),
        "transition_counts": dict(Counter(str(entry.event_type) for entry in ledger)),
        "max_advance_actions": max(tools.advance_counts, default=0)
        if tools is not None
        else 0,
        "document_isolation": document_isolation,
        "document_isolated": document_isolation.get("passed") is True,
    }
    if repository is not None:
        repository.close()
    return result


def _not_applicable(reason: str) -> dict[str, object]:
    """Represent an intentionally unmeasurable scenario without a pass."""
    return {
        "applicable": False,
        "passed": None,
        "checked_outputs": 0,
        "failures": [],
        "reason": reason,
    }


def _evidence_integrity(session: StudySession) -> dict[str, object]:
    """Check factual learner-facing output for selected-document evidence."""
    checks: list[tuple[str, tuple[SourceReference, ...], bool]] = []
    for index, artifact in enumerate(session.artifacts):
        if isinstance(artifact, SummaryArtifact):
            checks.append(
                (
                    f"artifact[{index}].summary",
                    artifact.citations,
                    bool(artifact.text.strip()),
                )
            )
        elif isinstance(artifact, QuizArtifact):
            checks.append(
                (
                    f"artifact[{index}].quiz",
                    artifact.citations,
                    bool(artifact.questions),
                )
            )
            for question_index, question in enumerate(artifact.questions):
                checks.append(
                    (
                        f"artifact[{index}].quiz[{question_index}]",
                        question.citations,
                        bool(question.prompt.strip() and question.answer.strip()),
                    )
                )
        elif isinstance(artifact, FlashcardArtifact):
            checks.append(
                (
                    f"artifact[{index}].flashcards",
                    artifact.citations,
                    bool(artifact.cards),
                )
            )
            for card_index, card in enumerate(artifact.cards):
                checks.append(
                    (
                        f"artifact[{index}].flashcards[{card_index}]",
                        card.citations,
                        bool(card.front.strip() and card.back.strip()),
                    )
                )
    for index, turn in enumerate(session.turns):
        if turn.kind in {TutorTurnKind.EXPLANATION, TutorTurnKind.ASSESSMENT}:
            checks.append(
                (
                    f"turn[{index}].{turn.kind.value}",
                    turn.citations,
                    bool(turn.tutor_message.strip()),
                )
            )
    if session.pending_interaction is not None:
        checks.append(
            (
                "pending_interaction",
                session.pending_interaction.citations,
                bool(session.pending_interaction.question.strip()),
            )
        )
    if not checks:
        return _not_applicable(
            "This scenario produced no factual learner-facing output."
        )
    failures: list[str] = []
    for label, citations, has_content in checks:
        if not has_content:
            failures.append(f"{label} has no factual content.")
        if not citations:
            failures.append(f"{label} has no citations.")
        elif any(
            reference.document_id != session.document_id for reference in citations
        ):
            failures.append(f"{label} cites another document.")
    return {
        "applicable": True,
        "passed": not failures,
        "checked_outputs": len(checks),
        "failures": failures,
    }


def _session_citations(session: StudySession) -> tuple[SourceReference, ...]:
    references: list[SourceReference] = [
        reference for entry in session.ledger for reference in entry.citations
    ]
    for artifact in session.artifacts:
        references.extend(artifact.citations)
        if isinstance(artifact, QuizArtifact):
            references.extend(
                reference
                for question in artifact.questions
                for reference in question.citations
            )
        elif isinstance(artifact, FlashcardArtifact):
            references.extend(
                reference for card in artifact.cards for reference in card.citations
            )
    references.extend(
        reference for turn in session.turns for reference in turn.citations
    )
    if session.pending_interaction is not None:
        references.extend(session.pending_interaction.citations)
    return tuple(references)


def _document_isolation(
    session: StudySession, tools: EvaluationTools | None
) -> dict[str, object]:
    """Check tool scope and every stored citation identity, without vacuous passes."""
    calls = tools.calls if tools is not None else []
    citations = _session_citations(session)
    if not calls and not citations:
        return _not_applicable("This scenario executed no scoped tool or cited output.")
    failures: list[str] = []
    selected_document = session.document_id.value
    if tools is None:
        failures.append("No tool-call record was available for an executed scenario.")
    else:
        if tools.scope_violations:
            failures.append(
                f"{tools.scope_violations} tool calls violated document scope."
            )
        if any(document_id != selected_document for _, document_id in calls):
            failures.append(
                "A recorded tool call was not injected with the selected document."
            )
    if any(reference.document_id != session.document_id for reference in citations):
        failures.append(
            "A stored ledger, artifact, turn, or pending citation uses "
            "another document."
        )
    return {
        "applicable": True,
        "passed": not failures,
        "checked_tool_calls": len(calls),
        "checked_citations": len(citations),
        "failures": failures,
    }


def _advance(
    runner: LangGraphMissionRunner,
    tools: EvaluationTools,
    request: AdvanceStudyMissionRequest,
) -> object:
    """Invoke the real graph and record capability calls for this advance."""
    before = len(tools.calls)
    result = runner.run(request)
    tools.advance_counts.append(len(tools.calls) - before)
    return result


def _report(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    transition_counts = Counter()
    max_advance = 0
    for result in results.values():
        transition_counts.update(result.get("transition_counts", {}))
        max_advance = max(max_advance, int(result.get("max_advance_actions", 0)))
    evidence_gate, evidence_details = _aggregate_check(results, "evidence_integrity")
    isolation_gate, isolation_details = _aggregate_check(results, "document_isolation")
    ledger_gate = _ledger_gate(results)
    release_gates = {
        "evidence_integrity": evidence_gate,
        "document_isolation": isolation_gate,
        "resume_fidelity": results["save_reload_resume"]["passed"],
        "ledger_verification": ledger_gate,
        "bound_compliance": all(
            result.get("action_count", 0) <= 64 for result in results.values()
        )
        and max_advance <= 4,
    }
    return {
        "report_version": 2,
        "phase": "Phase 1 — Verifiable Learning Intelligence",
        "generated_at": datetime.now(UTC).isoformat(),
        "release_gates": release_gates,
        "metrics": {
            "evidence_integrity": release_gates["evidence_integrity"],
            "document_isolation": release_gates["document_isolation"],
            "resume_fidelity": release_gates["resume_fidelity"],
            "ledger_verification": release_gates["ledger_verification"],
            "bound_compliance": release_gates["bound_compliance"],
            "transition_counts": dict(transition_counts),
            "gate_details": {
                "evidence_integrity": evidence_details,
                "document_isolation": isolation_details,
            },
        },
        "scenarios": results,
    }


def _aggregate_check(
    results: dict[str, dict[str, Any]], key: str
) -> tuple[bool, dict[str, object]]:
    applicable: list[str] = []
    failures: dict[str, object] = {}
    checked_outputs = 0
    for name, result in results.items():
        applicable_flag, passed = _check_value(result.get(key))
        if not applicable_flag:
            continue
        applicable.append(name)
        value = result.get(key)
        if isinstance(value, Mapping):
            checked_outputs += int(value.get("checked_outputs", 0))
            if key == "document_isolation":
                checked_outputs += int(value.get("checked_tool_calls", 0))
                checked_outputs += int(value.get("checked_citations", 0))
            if not passed:
                failures[name] = value.get("failures", ["check failed"])
        elif not passed:
            failures[name] = ["check failed"]
    gate = bool(applicable) and not failures
    return gate, {
        "applicable_scenarios": applicable,
        "applicable_count": len(applicable),
        "checked_outputs": checked_outputs,
        "failures": failures,
    }


def _check_value(value: object) -> tuple[bool, bool]:
    """Normalize detailed checks while tolerating the old boolean test shape."""
    if isinstance(value, Mapping):
        return bool(value.get("applicable")), value.get("passed") is True
    if isinstance(value, bool):
        return True, value
    return False, False


def _ledger_gate(results: dict[str, dict[str, Any]]) -> bool:
    normal = [
        result
        for name, result in results.items()
        if name != "tampered_ledger_detection"
    ]
    if not normal or not all(
        result.get("ledger_verified") is True for result in normal
    ):
        return False
    tampered = results.get("tampered_ledger_detection")
    if tampered is None:
        return True
    return tampered.get("passed") is True and tampered.get("ledger_verified") is False


def _reference_payload() -> dict[str, object]:
    return {
        "document_id": "document-1",
        "chunk_id": "chunk-1",
        "page_number": 1,
        "excerpt": "Evidence",
    }


if __name__ == "__main__":
    raise SystemExit(main())
