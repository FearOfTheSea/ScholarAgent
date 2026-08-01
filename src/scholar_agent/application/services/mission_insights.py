"""Pure deterministic calculation of verifiable mission learning signals."""

from scholar_agent.application.dtos.mission_intelligence import MissionInsights
from scholar_agent.application.services.mission_ledger import MissionLedgerService
from scholar_agent.domain.entities.study_session import (
    LearningObjective,
    MilestoneStatus,
    MissionStatus,
    ObjectiveProgress,
    StudyMilestone,
    StudySession,
    objective_progress,
)


class MissionInsightsService:
    """Calculate insights without invoking models or changing session state."""

    def __init__(self, maximum_actions_per_session: int = 64) -> None:
        self._maximum_actions_per_session = maximum_actions_per_session

    def calculate(self, session: StudySession) -> MissionInsights:
        """Return stable signals from the current session and verified ledger."""
        planned = _planned_objectives(session)
        progress: dict[str, ObjectiveProgress] = {
            objective.identifier: objective_progress(
                objective.identifier, session.attempts
            )
            for objective in planned
        }
        mastery_counts = {
            label: 0 for label in ("unseen", "developing", "proficient", "mastered")
        }
        for item in progress.values():
            mastery_counts[item.label.value] += 1

        assessed = {
            objective_id: tuple(
                attempt
                for attempt in session.attempts
                if attempt.objective_id == objective_id
            )
            for objective_id in progress
        }
        assessed_objectives = tuple(
            attempts for attempts in assessed.values() if attempts
        )
        first_pass_rate = (
            sum(attempts[0].score >= 2 for attempts in assessed_objectives)
            / len(assessed_objectives)
            if assessed_objectives
            else None
        )

        completed_milestones = tuple(
            milestone
            for milestone in session.milestones
            if milestone.status is MilestoneStatus.COMPLETED
        )
        evidence_coverage = _evidence_coverage(session, completed_milestones)
        verification = MissionLedgerService.verify(session)
        remaining = max(0, self._maximum_actions_per_session - session.action_count)
        signals = _signals(
            session,
            progress,
            evidence_coverage,
            remaining,
            self._maximum_actions_per_session,
        )
        return MissionInsights(
            progress_percent=(
                len(completed_milestones) / len(session.milestones) * 100
                if session.milestones
                else None
            ),
            mastery_counts=mastery_counts,
            assessment_count=len(session.attempts),
            first_pass_proficiency_rate=first_pass_rate,
            remediation_cycles=sum(
                entry.event_type == "remediation_started" for entry in session.ledger
            ),
            evidence_coverage=evidence_coverage,
            action_budget_used=session.action_count,
            action_budget_remaining=remaining,
            ledger_verified=verification.valid,
            next_action=_next_action(session, remaining),
            signals=signals,
        )


def _planned_objectives(session: StudySession) -> tuple[LearningObjective, ...]:
    identifiers = session.plan.objective_ids if session.plan is not None else ()
    return tuple(
        objective
        for objective in session.brief.objectives
        if not identifiers or objective.identifier in identifiers
    )


def _evidence_coverage(
    session: StudySession, completed: tuple[StudyMilestone, ...]
) -> float | None:
    if not completed:
        return None
    covered = 0
    for milestone in completed:
        if any(
            entry.capability == milestone.capability
            and (
                milestone.objective_id is None
                or entry.objective_id == milestone.objective_id
            )
            and bool(entry.citations)
            for entry in session.ledger
        ):
            covered += 1
    return covered / len(completed)


def _signals(
    session: StudySession,
    progress: dict[str, ObjectiveProgress],
    evidence_coverage: float | None,
    remaining: int,
    maximum: int,
) -> tuple[str, ...]:
    signals: list[str] = []
    if any(item.label.value == "developing" for item in progress.values()):
        signals.append("needs_remediation")
    if any(item.attempt_count == 0 for item in progress.values()):
        signals.append("unassessed_objectives")
    if evidence_coverage is not None and evidence_coverage < 1:
        signals.append("low_evidence_coverage")
    if (
        remaining <= max(1, maximum // 16)
        and session.status is not MissionStatus.COMPLETED
    ):
        signals.append("near_action_limit")
    if session.status is MissionStatus.COMPLETED:
        signals.append("mission_complete")
    return tuple(signals)


def _next_action(session: StudySession, remaining: int) -> str:
    if session.status is MissionStatus.COMPLETED:
        return "Mission complete; review the verified record."
    if session.status is MissionStatus.FAILED:
        return "Resume the recoverable mission failure when ready."
    if session.pending_interaction is not None:
        return (
            "Answer the pending question for objective "
            f"{session.pending_interaction.objective_id}."
        )
    if remaining == 0:
        return "The action budget is exhausted; complete or resume the mission."
    milestone = next(
        (
            item
            for item in session.milestones
            if item.status
            not in {
                MilestoneStatus.COMPLETED,
                MilestoneStatus.SKIPPED,
                MilestoneStatus.FAILED,
            }
        ),
        None,
    )
    if milestone is not None:
        return f"Continue with the next milestone: {milestone.title}."
    return "Continue the mission when ready."
