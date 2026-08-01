"""Application service for assessment, mastery, and cited remediation."""

from dataclasses import replace
from datetime import UTC, datetime

from scholar_agent.application.dtos.tutor import TutorActivity
from scholar_agent.application.services.mission_capabilities import (
    MissionCapabilityService,
)
from scholar_agent.application.services.mission_payloads import (
    integer_value,
    references,
    search_chunk_ids,
    string_values,
    text_value,
)
from scholar_agent.application.services.mission_policy import MissionPolicy
from scholar_agent.application.services.mission_state import MissionStateService
from scholar_agent.application.services.mission_steps import MissionStep
from scholar_agent.domain.entities.study_session import (
    LearnerAttempt,
    MissionStatus,
    PendingLearnerInteraction,
    StudySession,
    TutorTurn,
    TutorTurnKind,
    objective_progress,
)


class MissionAssessmentService:
    """Score pending responses and choose remediation or the next check."""

    def __init__(
        self,
        capabilities: MissionCapabilityService,
        state_service: MissionStateService,
        policy: MissionPolicy,
    ) -> None:
        self._capabilities = capabilities
        self._state = state_service
        self._policy = policy

    def assess(
        self,
        session: StudySession,
        message: str,
        remaining_capability_budget: int = 64,
    ) -> MissionStep:
        """Assess one learner response and apply deterministic mastery policy."""
        pending = session.pending_interaction
        if pending is None:
            raise ValueError("There is no pending learner question.")
        session, payload = self._capabilities.execute(
            session,
            "assess_learner_response",
            {
                "objective_id": pending.objective_id,
                "pending_question": pending.question,
                "learner_response": message,
                "source_chunk_ids": [
                    reference.chunk_id for reference in pending.citations
                ],
            },
        )
        score = integer_value(payload, "score")
        feedback = text_value(payload, "feedback")
        missing = string_values(payload.get("missing_concepts"))
        next_question = text_value(payload, "next_question")
        citations = references(payload.get("citations"), session.document_id)
        now = datetime.now(UTC)
        attempt = LearnerAttempt(
            pending.objective_id,
            message,
            score,
            feedback,
            missing,
            citations,
            now,
        )
        turn = TutorTurn(
            TutorTurnKind.ASSESSMENT,
            message,
            f"**Assessment: {score}/3.** {feedback}",
            pending.objective_id,
            citations,
            attempt,
            now,
        )
        session = self._state.checkpoint(
            replace(
                session,
                attempts=session.attempts + (attempt,),
                turns=session.turns + (turn,),
                pending_interaction=None,
                status=MissionStatus.ACTIVE,
                updated_at=now,
            ),
            "assessment",
            "Learner response was scored against the selected document.",
            objective_id=pending.objective_id,
            citations=citations,
            transition_key=(
                f"assessment:{session.identifier}:{len(session.attempts) + 1}"
            ),
        )
        if score < 2:
            if remaining_capability_budget < 2:
                session = self._state.set_pending(
                    session,
                    PendingLearnerInteraction(
                        pending.objective_id,
                        next_question,
                        citations=citations,
                        attempts=pending.attempts + 1,
                    ),
                )
                return MissionStep(
                    session,
                    TutorActivity(
                        TutorTurnKind.HINT,
                        f"**Assessment: {score}/3.** {feedback}\n\n"
                        f"**Try again:** {next_question}",
                        pending.objective_id,
                        citations,
                    ),
                    1,
                )
            session, activity, used = self._remediate(session, pending, missing)
            return MissionStep(session, activity, 1 + used)

        progress = objective_progress(pending.objective_id, session.attempts)
        activity_message = f"**Assessment: {score}/3.** {feedback}"
        if score == 2 or progress.label.value != "mastered":
            session = self._state.set_pending(
                session,
                PendingLearnerInteraction(
                    pending.objective_id,
                    next_question,
                    citations=citations,
                ),
            )
            activity_message += f"\n\n**Next challenge:** {next_question}"
            return MissionStep(
                session,
                TutorActivity(
                    TutorTurnKind.ASSESSMENT,
                    activity_message,
                    pending.objective_id,
                    citations,
                ),
                1,
            )

        milestone = self._policy.practice_milestone(session, pending.objective_id)
        if milestone is not None:
            session = self._capabilities.complete_milestone(
                session, milestone.identifier
            )
        return MissionStep(
            session,
            TutorActivity(
                TutorTurnKind.ASSESSMENT,
                activity_message,
                pending.objective_id,
                citations,
            ),
            1,
            True,
        )

    def _remediate(
        self,
        session: StudySession,
        pending: PendingLearnerInteraction,
        missing: tuple[str, ...],
    ) -> tuple[StudySession, TutorActivity, int]:
        question = "Focus on the missing concept: " + ", ".join(
            missing or ("the objective",)
        )
        session = self._state.checkpoint(
            session,
            "remediation",
            "Targeted remediation started for an under-proficient response.",
            objective_id=pending.objective_id,
            citations=pending.citations,
            transition_key=(
                f"remediation:{session.identifier}:{len(session.attempts)}"
            ),
        )
        session, search_payload = self._capabilities.execute(
            session,
            "semantic_search",
            {"query": question, "limit": 4},
        )
        source_ids = search_chunk_ids(search_payload, session.document_id)
        if not source_ids:
            source_ids = [reference.chunk_id for reference in pending.citations]
        session, payload = self._capabilities.execute(
            session,
            "explain_concept",
            {
                "objective_id": pending.objective_id,
                "source_chunk_ids": source_ids,
                "learner_question": question,
                "style": "targeted remediation",
            },
        )
        explanation = text_value(payload, "explanation")
        next_question = text_value(payload, "check_question")
        citations = references(payload.get("citations"), session.document_id)
        session = self._state.set_pending(
            session,
            PendingLearnerInteraction(
                pending.objective_id,
                next_question,
                citations=citations,
                attempts=pending.attempts + 1,
            ),
        )
        return (
            session,
            TutorActivity(
                TutorTurnKind.HINT,
                explanation + f"\n\n**Try again:** {next_question}",
                pending.objective_id,
                citations,
            ),
            2,
        )
