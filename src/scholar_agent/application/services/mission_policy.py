"""Framework-free policies for bounded Study Mission transitions."""

from dataclasses import dataclass
from enum import StrEnum

from scholar_agent.domain.entities.study_session import (
    LearningObjective,
    MilestoneStatus,
    MissionStatus,
    ObjectiveProgress,
    PendingLearnerInteraction,
    SourceReference,
    StudyMilestone,
    StudySession,
    objective_progress,
)


class MissionRoute(StrEnum):
    """Explicit route selected from one advance request."""

    UNSUPPORTED = "unsupported"
    FINISH = "finish"
    HINT = "hint"
    RECAP = "recap"
    SIDE_QUESTION = "side_question"
    ASSESS = "assess"
    AUTO = "auto"
    WAIT = "wait"
    FINALIZE = "finalize"


@dataclass(frozen=True, slots=True)
class MissionSelection:
    """The next pedagogical action selected from current mission state."""

    route: MissionRoute
    milestone: StudyMilestone | None = None


class MissionPolicy:
    """Select mission routes without knowing about LangGraph or persistence."""

    def classify(self, session: StudySession, message: str | None) -> MissionRoute:
        """Classify explicit learner intent before selecting automatic work."""
        if session.status is MissionStatus.COMPLETED:
            return MissionRoute.FINALIZE
        if message is None:
            return MissionRoute.AUTO
        if self.is_unsupported(message):
            return MissionRoute.UNSUPPORTED
        if self.is_finish(message):
            return MissionRoute.FINISH
        if self.is_hint(message):
            return MissionRoute.HINT
        if self.is_recap(message):
            return MissionRoute.RECAP
        if self.is_side_question(message):
            return MissionRoute.SIDE_QUESTION
        if session.pending_interaction is not None and self.is_answer(message):
            return MissionRoute.ASSESS
        return MissionRoute.AUTO

    def select(self, session: StudySession, route: MissionRoute) -> MissionSelection:
        """Select one milestone for the automatic route."""
        if route is not MissionRoute.AUTO:
            return MissionSelection(route)
        if session.pending_interaction is not None:
            return MissionSelection(MissionRoute.WAIT)
        milestone = self.next_milestone(session)
        if milestone is None:
            return MissionSelection(MissionRoute.FINISH)
        return MissionSelection(route, milestone)

    def next_route(
        self,
        session: StudySession,
        step_activity: object,
        continue_auto: bool,
        executions: int,
        maximum_executions: int,
    ) -> MissionRoute:
        """Choose whether the graph loops, waits, or finalizes."""
        if session.status in {MissionStatus.COMPLETED, MissionStatus.FAILED}:
            return MissionRoute.FINALIZE
        if session.pending_interaction is not None:
            return MissionRoute.FINALIZE
        if not continue_auto:
            return MissionRoute.FINALIZE
        if executions >= maximum_executions:
            return MissionRoute.FINALIZE
        return MissionRoute.AUTO

    def current_objective(self, session: StudySession) -> LearningObjective | None:
        """Return the first objective selected by the mission plan."""
        objective_ids = session.plan.objective_ids if session.plan else ()
        return next(
            (
                objective
                for objective in session.brief.objectives
                if not objective_ids or objective.identifier in objective_ids
            ),
            None,
        )

    def progress(self, session: StudySession) -> tuple[ObjectiveProgress, ...]:
        """Calculate application-owned mastery for planned objectives."""
        return tuple(
            objective_progress(objective.identifier, session.attempts)
            for objective in session.brief.objectives
            if session.plan is None
            or objective.identifier in session.plan.objective_ids
        )

    @staticmethod
    def next_milestone(session: StudySession) -> StudyMilestone | None:
        """Return the next milestone not terminally completed or skipped."""
        return next(
            (
                milestone
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

    @staticmethod
    def practice_milestone(
        session: StudySession, objective_id: str
    ) -> StudyMilestone | None:
        """Find the practice milestone for one objective."""
        return next(
            (
                milestone
                for milestone in session.milestones
                if milestone.objective_id == objective_id
                and milestone.capability == "assess_learner_response"
            ),
            None,
        )

    @staticmethod
    def is_optional_artifact(milestone: StudyMilestone) -> bool:
        """Identify artifact work that may fail without stopping a mission."""
        return milestone.capability in {
            "summarize_document",
            "generate_quiz",
            "generate_flashcards",
        }

    @staticmethod
    def is_unsupported(message: str) -> bool:
        lowered = message.casefold()
        return any(
            word in lowered for word in ("web", "internet", "another pdf", "compare")
        )

    @staticmethod
    def is_finish(message: str) -> bool:
        return message.casefold().strip() in {
            "finish",
            "finish mission",
            "complete",
            "done",
            "end mission",
        }

    @staticmethod
    def is_hint(message: str) -> bool:
        return "hint" in message.casefold()

    @staticmethod
    def is_recap(message: str) -> bool:
        return any(
            word in message.casefold() for word in ("recap", "progress", "review")
        )

    @staticmethod
    def is_answer(message: str) -> bool:
        return message.casefold().strip() not in {"continue", "next"}

    @staticmethod
    def is_side_question(message: str) -> bool:
        lowered = message.casefold()
        return message.rstrip().endswith("?") or any(
            phrase in lowered for phrase in ("explain", "what is", "why ", "how ")
        )

    @staticmethod
    def pending_question(
        objective_id: str,
        question: str,
        citations: tuple[SourceReference, ...],
        *,
        reference_answer: str | None = None,
        attempts: int = 0,
    ) -> PendingLearnerInteraction:
        """Construct learner-visible pending state without exposing answers."""
        return PendingLearnerInteraction(
            objective_id=objective_id,
            question=question,
            reference_answer=reference_answer,
            citations=citations,
            attempts=attempts,
        )
