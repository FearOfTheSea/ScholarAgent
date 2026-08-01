"""Framework-free construction and validation of mission ledger entries."""

from dataclasses import replace
from datetime import UTC, datetime

from scholar_agent.domain.entities.mission_ledger import (
    LedgerVerificationResult,
    MissionLedgerEntry,
    MissionLedgerEventType,
    MissionStateProjection,
    compute_ledger_digest,
    verify_ledger,
)
from scholar_agent.domain.entities.study_session import (
    MilestoneStatus,
    MissionStatus,
    SourceReference,
    StudySession,
    objective_progress,
)

MAX_MISSION_LEDGER_ENTRIES = 512


class MissionLedgerService:
    """Append and verify redacted, bounded mission transition records."""

    def append(
        self,
        session: StudySession,
        event_type: MissionLedgerEventType | str,
        summary: str,
        *,
        objective_id: str | None = None,
        capability: str | None = None,
        citations: tuple[SourceReference, ...] = (),
        transition_key: str | None = None,
    ) -> StudySession:
        """Append one entry, or return the unchanged state for an idempotent key."""
        if transition_key is not None and any(
            entry.transition_key == transition_key for entry in session.ledger
        ):
            return session
        if any(reference.document_id != session.document_id for reference in citations):
            raise ValueError("Mission ledger citations must use the selected document.")
        if len(session.ledger) >= MAX_MISSION_LEDGER_ENTRIES:
            return replace(
                session,
                status=MissionStatus.FAILED,
                updated_at=datetime.now(UTC),
            )
        sequence = len(session.ledger) + 1
        previous_digest = session.ledger[-1].current_digest if session.ledger else ""
        projection = project_session(session)
        current_digest = compute_ledger_digest(
            previous_digest=previous_digest,
            sequence=sequence,
            event_type=event_type,
            projection=projection,
            objective_id=objective_id,
            capability=capability,
            citations=citations,
        )
        entry = MissionLedgerEntry(
            sequence=sequence,
            event_type=event_type,
            summary=summary,
            objective_id=objective_id,
            capability=capability,
            citations=citations,
            projection=projection,
            previous_digest=previous_digest,
            current_digest=current_digest,
            created_at=datetime.now(UTC),
            transition_key=transition_key,
        )
        return replace(session, ledger=session.ledger + (entry,))

    @staticmethod
    def verify(session: StudySession) -> LedgerVerificationResult:
        """Verify a complete session ledger."""
        return verify_ledger(session.ledger)


def project_session(session: StudySession) -> MissionStateProjection:
    """Project only replay-safe mission state into a ledger entry."""
    active = next(
        (
            milestone.identifier
            for milestone in session.milestones
            if milestone.status is MilestoneStatus.ACTIVE
        ),
        None,
    )
    next_milestone = next(
        (
            milestone.identifier
            for milestone in session.milestones
            if milestone.status
            not in {
                MilestoneStatus.COMPLETED,
                MilestoneStatus.SKIPPED,
                MilestoneStatus.FAILED,
            }
        ),
        None,
    )
    planned_ids = session.plan.objective_ids if session.plan is not None else ()
    mastery = tuple(
        (
            objective.identifier,
            objective_progress(objective.identifier, session.attempts).label.value,
        )
        for objective in session.brief.objectives
        if not planned_ids or objective.identifier in planned_ids
    )
    return MissionStateProjection(
        status=session.status.value,
        active_milestone_id=active,
        pending_objective_id=(
            session.pending_interaction.objective_id
            if session.pending_interaction is not None
            else None
        ),
        action_count=session.action_count,
        attempt_count=len(session.attempts),
        artifact_count=len(session.artifacts),
        completed_milestone_count=sum(
            milestone.status is MilestoneStatus.COMPLETED
            for milestone in session.milestones
        ),
        mastery_by_objective=mastery,
        next_milestone_id=next_milestone,
    )
