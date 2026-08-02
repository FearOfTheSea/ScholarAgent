"""Release-gate regressions for the Phase 2 deterministic evaluator."""

from pathlib import Path

from scripts.evaluate_reviews import _not_applicable, _report, _run_scenarios


def test_phase2_evaluator_requires_applicable_passing_checks() -> None:
    result = {
        "passed": True,
        "checks": {
            "privacy": {"applicable": True, "passed": False},
            "scheduling": _not_applicable("not measured"),
        },
    }
    report = _report({"broken": result})

    assert report["release_gates"]["privacy_redaction"] is False
    assert report["release_gates"]["deterministic_scheduling"] is False


def test_phase2_evaluator_real_nine_scenario_report_is_green(tmp_path: Path) -> None:
    report = _report(_run_scenarios(tmp_path))

    assert len(report["scenarios"]) == 9
    assert all(item["passed"] for item in report["scenarios"].values())
    assert all(report["release_gates"].values())
