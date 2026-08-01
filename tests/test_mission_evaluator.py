"""Regression tests for the Phase 1 mission release-gate evaluator."""

from dataclasses import replace
from pathlib import Path

from test_mission_intelligence import _session

from scholar_agent.application.services.mission_ledger import MissionLedgerService
from scholar_agent.domain.entities.mission_ledger import MissionLedgerEventType
from scholar_agent.domain.entities.study_material import SummaryArtifact
from scripts.evaluate_missions import (
    EvaluationTools,
    _document_isolation,
    _evidence_integrity,
    _not_applicable,
    _report,
    _run_scenarios,
)


def _result(
    *,
    evidence: object | None = None,
    isolation: object | None = None,
    ledger_verified: bool = True,
) -> dict[str, object]:
    return {
        "passed": True,
        "action_count": 0,
        "max_advance_actions": 0,
        "ledger_verified": ledger_verified,
        "evidence_integrity": evidence or _not_applicable("No factual output."),
        "document_isolation": isolation
        or _not_applicable("No scoped execution or cited output."),
        "transition_counts": {},
    }


def _report_inputs(normal: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        "normal": normal,
        "save_reload_resume": _result(),
        "tampered_ledger_detection": {
            **_result(),
            "ledger_verified": False,
        },
    }


def test_missing_citation_factual_result_fails_evidence_gate() -> None:
    session = _session()
    session = replace(
        session,
        artifacts=(SummaryArtifact("An uncited factual summary", ()),),
    )
    evidence = _evidence_integrity(session)

    report = _report(_report_inputs(_result(evidence=evidence)))

    assert evidence["applicable"] is True
    assert evidence["passed"] is False
    assert report["release_gates"]["evidence_integrity"] is False


def test_all_not_applicable_evidence_checks_do_not_pass_vacuously() -> None:
    report = _report(_report_inputs(_result()))

    details = report["metrics"]["gate_details"]["evidence_integrity"]
    assert details["applicable_count"] == 0
    assert report["release_gates"]["evidence_integrity"] is False


def test_wrong_document_tool_call_fails_isolation_gate() -> None:
    session = _session()
    reference = session.brief.objectives[0].citations[0]
    session = MissionLedgerService().append(
        session,
        MissionLedgerEventType.CAPABILITY_COMPLETED,
        "A cited capability completed.",
        capability="semantic_search",
        citations=(reference,),
        transition_key="cited-search",
    )
    tools = EvaluationTools()
    tools.calls.append(("semantic_search", "document-2"))
    tools.scope_violations = 1
    isolation = _document_isolation(session, tools)

    report = _report(_report_inputs(_result(isolation=isolation)))

    assert isolation["applicable"] is True
    assert isolation["passed"] is False
    assert report["release_gates"]["document_isolation"] is False


def test_normal_ledger_verification_failure_fails_ledger_gate() -> None:
    report = _report(
        _report_inputs(
            _result(ledger_verified=False),
        )
    )

    assert report["release_gates"]["ledger_verification"] is False


def test_real_eight_scenario_report_keeps_all_release_gates_green(
    tmp_path: Path,
) -> None:
    results = _run_scenarios(tmp_path)
    report = _report(results)

    assert len(results) == 8
    assert all(result["passed"] for result in results.values())
    assert all(report["release_gates"].values())
    assert (
        report["metrics"]["gate_details"]["evidence_integrity"]["applicable_count"] >= 1
    )
    assert results["tampered_ledger_detection"]["ledger_verified"] is False
