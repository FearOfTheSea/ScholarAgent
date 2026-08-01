"""Framework-free checks performed between mission graph actions."""

from scholar_agent.application.services.mission_steps import MissionStep
from scholar_agent.domain.entities.study_session import MissionStatus, StudySession


class MissionVerificationService:
    """Verify state and execution budgets before reflection."""

    def verify(
        self,
        before: StudySession,
        step: MissionStep,
        turn_executions: int,
        maximum_actions_per_turn: int,
        maximum_actions_per_session: int,
    ) -> None:
        """Reject impossible state transitions before the graph loops."""
        session = step.session
        if session.document_id != before.document_id:
            raise ValueError("Mission action changed the selected document.")
        if session.action_count < before.action_count:
            raise ValueError("Mission action count cannot decrease.")
        if session.action_count - before.action_count != step.capability_executions:
            raise ValueError(
                "Mission action count does not match capability executions."
            )
        if turn_executions > maximum_actions_per_turn:
            raise ValueError("The per-advance mission action limit has been reached.")
        if session.action_count > maximum_actions_per_session:
            raise ValueError("The mission action limit has been reached.")
        if session.status is MissionStatus.AWAITING_LEARNER and (
            session.pending_interaction is None
        ):
            raise ValueError("Awaiting-learner missions require pending interaction.")
