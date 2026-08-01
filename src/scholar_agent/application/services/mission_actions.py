"""Application dispatcher for one selected Study Mission route."""

from dataclasses import replace

from scholar_agent.application.dtos.tutor import TutorActivity
from scholar_agent.application.services.mission_assessment import (
    MissionAssessmentService,
)
from scholar_agent.application.services.mission_interactions import (
    MissionInteractionService,
)
from scholar_agent.application.services.mission_materials import MissionMaterialService
from scholar_agent.application.services.mission_policy import (
    MissionPolicy,
    MissionRoute,
    MissionSelection,
)
from scholar_agent.application.services.mission_state import MissionStateService
from scholar_agent.application.services.mission_steps import MissionStep
from scholar_agent.domain.entities.study_session import (
    MilestoneStatus,
    StudySession,
    TutorTurnKind,
)


class MissionActionService:
    """Dispatch graph selections to focused pedagogical application services."""

    def __init__(
        self,
        state_service: MissionStateService,
        policy: MissionPolicy,
        interactions: MissionInteractionService,
        materials: MissionMaterialService,
        assessment: MissionAssessmentService,
    ) -> None:
        self._state = state_service
        self._policy = policy
        self._interactions = interactions
        self._materials = materials
        self._assessment = assessment

    def execute(
        self,
        session: StudySession,
        selection: MissionSelection,
        message: str | None,
        remaining_capability_budget: int = 64,
    ) -> MissionStep:
        """Dispatch one route without owning graph control flow."""
        route = selection.route
        if route is MissionRoute.UNSUPPORTED:
            return self._interactions.unsupported(session)
        if route is MissionRoute.FINISH:
            return self._interactions.finish(
                session, "Learner marked the mission complete."
            )
        if route is MissionRoute.HINT:
            return self._interactions.hint(session)
        if route is MissionRoute.RECAP:
            return self._interactions.recap(session)
        if route is MissionRoute.SIDE_QUESTION:
            return self._interactions.side_question(session, message or "")
        if route is MissionRoute.ASSESS:
            return self._assessment.assess(
                session,
                message or "",
                remaining_capability_budget,
            )
        if route is MissionRoute.WAIT:
            return self._interactions.wait(session)
        if route is MissionRoute.AUTO and selection.milestone is not None:
            return self._materials.execute(session, selection.milestone)
        return MissionStep(session, None, 0)

    def fail(self, session: StudySession) -> MissionStep:
        """Persist a resumable core failure."""
        updated = self._state.fail(session)
        return MissionStep(
            updated,
            TutorActivity(
                TutorTurnKind.UNSUPPORTED,
                "Mission paused after a recoverable failure.",
                None,
                (),
            ),
            0,
        )

    def skip_optional(
        self,
        session: StudySession,
        selection: MissionSelection,
        capability_executions: int = 0,
    ) -> MissionStep:
        """Skip an optional artifact after a bounded capability failure."""
        milestone = selection.milestone
        if milestone is None or not self._policy.is_optional_artifact(milestone):
            raise ValueError("Only optional artifact milestones may be skipped.")
        milestones = tuple(
            replace(
                item,
                status=(
                    MilestoneStatus.FAILED
                    if item.identifier == milestone.identifier
                    else item.status
                ),
            )
            for item in session.milestones
        )
        updated = self._state.checkpoint(
            replace(session, milestones=milestones),
            "failure",
            f"Optional artifact {milestone.capability} was unavailable; "
            "mission continues.",
            milestone.capability,
        )
        return MissionStep(updated, None, capability_executions, True)

    @staticmethod
    def combine_activity(
        first: TutorActivity | None, second: TutorActivity | None
    ) -> TutorActivity | None:
        """Preserve an assessment message when automatic follow-up also speaks."""
        if first is None:
            return second
        if second is None:
            return first
        return TutorActivity(
            second.kind,
            first.message + "\n\n" + second.message,
            second.objective_id,
            second.citations,
        )
