"""Framework-free Study Mission persistence and lifecycle transitions."""

from dataclasses import replace
from datetime import UTC, datetime

from scholar_agent.application.dtos.tutor import TutorActivity
from scholar_agent.application.services.mission_ledger import MissionLedgerService
from scholar_agent.domain.entities.study_session import (
    MissionLedgerEventType,
    MissionStatus,
    MissionTraceEvent,
    PendingLearnerInteraction,
    SourceReference,
    StudySession,
    TutorTurnKind,
)
from scholar_agent.domain.repositories.study_session_repository import (
    StudySessionRepository,
)


class MissionStateService:
    """Persist redacted mission state and lifecycle transitions."""

    def __init__(
        self,
        session_repository: StudySessionRepository,
        ledger_service: MissionLedgerService | None = None,
    ) -> None:
        self._session_repository = session_repository
        self._ledger = ledger_service or MissionLedgerService()

    def load(self, session_id: str) -> StudySession:
        """Load one session or raise the existing not-found contract."""
        session, _ = self.load_with_trace(session_id)
        return session

    def load_with_trace(self, session_id: str) -> tuple[StudySession, int]:
        """Load a session and retain the trace offset for result redaction."""
        session = self._session_repository.get(session_id)
        if session is None:
            raise ValueError(f"Study session '{session_id}' was not found.")
        trace_start = len(session.trace)
        if session.status is MissionStatus.FAILED:
            session = self.checkpoint(
                replace(session, status=MissionStatus.ACTIVE),
                "retry",
                "Resumed after a recoverable mission failure.",
            )
        return session, trace_start

    def save(self, session: StudySession) -> StudySession:
        """Save a state checkpoint without adding a public trace event."""
        self._session_repository.save(session)
        return session

    def current(self, session_id: str) -> StudySession | None:
        """Read the latest persisted snapshot after a capability-side failure."""
        return self._session_repository.get(session_id)

    def checkpoint(
        self,
        session: StudySession,
        event_type: str,
        summary: str,
        capability: str | None = None,
        *,
        objective_id: str | None = None,
        citations: tuple[SourceReference, ...] = (),
        transition_key: str | None = None,
        ledger_event_type: MissionLedgerEventType | str | None = None,
    ) -> StudySession:
        """Append one trace and ledger event, then persist one snapshot."""
        if transition_key is not None and any(
            entry.transition_key == transition_key for entry in session.ledger
        ):
            return session
        event = MissionTraceEvent(
            event_type=event_type,
            summary=summary,
            capability=capability,
            state=session.status.value,
            created_at=datetime.now(UTC),
        )
        updated = replace(session, trace=session.trace + (event,))
        updated = self._ledger.append(
            updated,
            ledger_event_type or _ledger_event_type(event_type),
            summary,
            objective_id=objective_id,
            capability=capability,
            citations=citations,
            transition_key=transition_key,
        )
        self._session_repository.save(updated)
        return updated

    def set_pending(
        self, session: StudySession, pending: PendingLearnerInteraction
    ) -> StudySession:
        """Persist the learner wait boundary with no reference answer output."""
        return self.checkpoint(
            replace(
                session,
                pending_interaction=pending,
                status=MissionStatus.AWAITING_LEARNER,
                updated_at=datetime.now(UTC),
            ),
            "wait",
            "Waiting for the learner response.",
            objective_id=pending.objective_id,
            citations=pending.citations,
        )

    def complete(self, session: StudySession, summary: str) -> StudySession:
        """Mark a mission complete and clear pending learner state."""
        now = datetime.now(UTC)
        return self.checkpoint(
            replace(
                session,
                status=MissionStatus.COMPLETED,
                completed_at=now,
                pending_interaction=None,
                updated_at=now,
            ),
            "completion",
            summary,
            ledger_event_type=MissionLedgerEventType.MISSION_COMPLETED,
            transition_key=f"mission-completed:{session.identifier}",
        )

    def fail(self, session: StudySession) -> StudySession:
        """Persist a resumable failure without recording model output."""
        return self.checkpoint(
            replace(session, status=MissionStatus.FAILED, updated_at=datetime.now(UTC)),
            "failure",
            "Mission paused after a recoverable failure.",
            ledger_event_type=MissionLedgerEventType.MISSION_FAILED,
        )

    def wait_activity(
        self,
        session: StudySession,
        message: str,
        objective_id: str | None,
        citations: tuple[SourceReference, ...],
        event_type: str = "wait",
    ) -> tuple[StudySession, TutorActivity]:
        """Persist a concise wait/unsupported interaction."""
        updated = self.checkpoint(session, event_type, "Mission is waiting.")
        kind = (
            TutorTurnKind.UNSUPPORTED
            if event_type == "unsupported"
            else TutorTurnKind.RECAP
        )
        return updated, TutorActivity(kind, message, objective_id, citations)


def _ledger_event_type(event_type: str) -> MissionLedgerEventType:
    mapping = {
        "start": MissionLedgerEventType.MISSION_STARTED,
        "plan": MissionLedgerEventType.PLAN_CREATED,
        "capability": MissionLedgerEventType.CAPABILITY_COMPLETED,
        "failure": MissionLedgerEventType.CAPABILITY_FAILED,
        "assessment": MissionLedgerEventType.LEARNER_ASSESSED,
        "remediation": MissionLedgerEventType.REMEDIATION_STARTED,
        "mastery": MissionLedgerEventType.MASTERY_CHANGED,
        "artifact": MissionLedgerEventType.ARTIFACT_CREATED,
        "wait": MissionLedgerEventType.WAITING_FOR_LEARNER,
        "completion": MissionLedgerEventType.MISSION_COMPLETED,
        "retry": MissionLedgerEventType.WAITING_FOR_LEARNER,
        "unsupported": MissionLedgerEventType.WAITING_FOR_LEARNER,
        "state": MissionLedgerEventType.MASTERY_CHANGED,
    }
    try:
        return mapping[event_type]
    except KeyError as error:
        raise ValueError(f"Unsupported mission trace event '{event_type}'.") from error
