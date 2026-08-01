"""Framework-free Study Mission result construction."""

from scholar_agent.application.dtos.tutor import StudySessionResult, TutorActivity
from scholar_agent.application.services.mission_policy import MissionPolicy
from scholar_agent.domain.entities.study_session import StudySession


class MissionResultService:
    """Build the presentation-neutral result returned by mission advances."""

    def __init__(self, policy: MissionPolicy) -> None:
        self._policy = policy

    def build(
        self,
        session: StudySession,
        activity: TutorActivity | None,
        trace_start: int,
    ) -> StudySessionResult:
        """Return current progress and only trace events from this advance."""
        current = self._policy.current_objective(session)
        return StudySessionResult(
            session=session,
            progress=self._policy.progress(session),
            current_objective_id=current.identifier if current else None,
            activity=activity,
            new_trace_events=session.trace[trace_start:],
        )
