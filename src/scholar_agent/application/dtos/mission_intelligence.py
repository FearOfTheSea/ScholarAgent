"""Presentation-neutral DTOs for verified mission intelligence."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class MissionInsights:
    """Deterministic learning signals calculated from one mission snapshot."""

    progress_percent: float | None
    mastery_counts: dict[str, int] = field(default_factory=dict)
    assessment_count: int = 0
    first_pass_proficiency_rate: float | None = None
    remediation_cycles: int = 0
    evidence_coverage: float | None = None
    action_budget_used: int = 0
    action_budget_remaining: int = 0
    ledger_verified: bool = True
    next_action: str = ""
    signals: tuple[str, ...] = ()
