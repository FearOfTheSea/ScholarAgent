"""Application service for explicit learner interactions and side questions."""

from scholar_agent.application.dtos.tutor import TutorActivity
from scholar_agent.application.services.mission_capabilities import (
    MissionCapabilityService,
)
from scholar_agent.application.services.mission_payloads import references, text_value
from scholar_agent.application.services.mission_policy import MissionPolicy
from scholar_agent.application.services.mission_state import MissionStateService
from scholar_agent.application.services.mission_steps import MissionStep
from scholar_agent.domain.entities.study_session import (
    StudySession,
    TutorTurnKind,
)


class MissionInteractionService:
    """Handle hints, recaps, unsupported requests, and side questions."""

    def __init__(
        self,
        capabilities: MissionCapabilityService,
        state_service: MissionStateService,
        policy: MissionPolicy,
    ) -> None:
        self._capabilities = capabilities
        self._state = state_service
        self._policy = policy

    def unsupported(self, session: StudySession) -> MissionStep:
        """Explain the single-document boundary without executing inference."""
        objective = self._policy.current_objective(session)
        updated, activity = self._state.wait_activity(
            session,
            "This mission is grounded in one selected document and cannot browse, "
            "compare, or switch documents.",
            objective.identifier if objective else None,
            objective.citations if objective else (),
            "unsupported",
        )
        return MissionStep(updated, activity, 0)

    def finish(self, session: StudySession, summary: str) -> MissionStep:
        """Complete a mission explicitly."""
        updated = self._state.complete(session, summary)
        return MissionStep(
            updated,
            TutorActivity(TutorTurnKind.RECAP, summary, None, ()),
            0,
        )

    def hint(self, session: StudySession) -> MissionStep:
        """Give a non-generative hint or explain that none is pending."""
        pending = session.pending_interaction
        if pending is None:
            objective = self._policy.current_objective(session)
            updated, activity = self._state.wait_activity(
                session,
                "There is no pending question to hint.",
                objective.identifier if objective else None,
                objective.citations if objective else (),
            )
            return MissionStep(updated, activity, 0)
        updated = self._state.checkpoint(session, "wait", "A hint was requested.")
        return MissionStep(
            updated,
            TutorActivity(
                TutorTurnKind.HINT,
                "Focus on the key terms in the cited objective, then try the "
                "question again.",
                pending.objective_id,
                pending.citations,
            ),
            0,
        )

    def recap(self, session: StudySession) -> MissionStep:
        """Render deterministic mastery progress without model reasoning."""
        lines = [
            f"- {item.objective_id}: {item.label.value} ({item.percentage}%)"
            for item in self._policy.progress(session)
        ]
        updated = self._state.checkpoint(
            session, "wait", "A mastery recap was requested."
        )
        objective = self._policy.current_objective(updated)
        return MissionStep(
            updated,
            TutorActivity(
                TutorTurnKind.RECAP,
                "Evidence-based mastery snapshot:\n" + "\n".join(lines),
                objective.identifier if objective else None,
                objective.citations if objective else (),
            ),
            0,
        )

    def side_question(self, session: StudySession, message: str) -> MissionStep:
        """Answer a side question while preserving any pending learner state."""
        objective = self._policy.current_objective(session)
        if objective is None:
            updated, activity = self._state.wait_activity(
                session, "There is no active objective yet.", None, ()
            )
            return MissionStep(updated, activity, 0)
        session, payload = self._capabilities.execute(
            session,
            "explain_concept",
            {
                "objective_id": objective.identifier,
                "source_chunk_ids": [
                    reference.chunk_id for reference in objective.citations
                ],
                "learner_question": message,
                "style": "answer the side question concisely",
            },
        )
        citations = references(payload.get("citations"), session.document_id)
        return MissionStep(
            session,
            TutorActivity(
                TutorTurnKind.EXPLANATION,
                text_value(payload, "explanation"),
                objective.identifier,
                citations,
            ),
            1,
        )

    def wait(self, session: StudySession) -> MissionStep:
        """Record that automatic work cannot cross a pending learner boundary."""
        objective = self._policy.current_objective(session)
        updated, activity = self._state.wait_activity(
            session,
            "Mission is ready for the next learner action.",
            objective.identifier if objective else None,
            objective.citations if objective else (),
        )
        return MissionStep(updated, activity, 0)
