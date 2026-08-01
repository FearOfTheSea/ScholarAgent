"""Application service for executing mission milestones and artifacts."""

from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime

from scholar_agent.application.dtos.tutor import TutorActivity
from scholar_agent.application.services.mission_capabilities import (
    MissionCapabilityService,
)
from scholar_agent.application.services.mission_payloads import (
    first_pending_question,
    flashcard_artifact,
    question_activity,
    quiz_artifact,
    references,
    summary_artifact,
    text_value,
)
from scholar_agent.application.services.mission_policy import MissionPolicy
from scholar_agent.application.services.mission_state import MissionStateService
from scholar_agent.application.services.mission_steps import MissionStep
from scholar_agent.domain.entities.study_session import (
    PendingLearnerInteraction,
    StudyMilestone,
    StudySession,
    TutorTurn,
    TutorTurnKind,
)


class MissionMaterialService:
    """Execute one learn/orient/review milestone."""

    def __init__(
        self,
        capabilities: MissionCapabilityService,
        state_service: MissionStateService,
        policy: MissionPolicy,
    ) -> None:
        self._capabilities = capabilities
        self._state = state_service
        self._policy = policy

    def execute(self, session: StudySession, milestone: StudyMilestone) -> MissionStep:
        """Run one milestone and retain all cited artifact/item state."""
        capability = milestone.capability
        source_ids = [reference.chunk_id for reference in milestone.citations]
        if capability == "assess_learner_response":
            objective = self._policy.current_objective(session)
            if objective is None:
                raise ValueError("Practice milestone has no objective.")
            pending = PendingLearnerInteraction(
                objective.identifier,
                f"Explain the key idea of {objective.title} in your own words.",
                citations=objective.citations,
            )
            updated = self._state.set_pending(session, pending)
            return MissionStep(updated, question_activity(pending), 0)

        arguments: dict[str, object] = {}
        if capability == "explain_concept":
            if milestone.objective_id is None:
                raise ValueError("Explanation milestone has no objective.")
            arguments = {
                "objective_id": milestone.objective_id,
                "source_chunk_ids": source_ids,
                "style": "concise",
            }
        elif capability == "generate_quiz":
            arguments = {"question_count": 3}
        elif capability == "generate_flashcards":
            arguments = {"card_count": 5}

        session, payload = self._capabilities.execute(session, capability, arguments)
        if capability == "build_document_map":
            return MissionStep(
                self._capabilities.complete_milestone(session, milestone.identifier),
                None,
                1,
                True,
            )
        if capability == "summarize_document":
            summary = summary_artifact(payload, session.document_id)
            updated = replace(session, artifacts=session.artifacts + (summary,))
            updated = self._state.checkpoint(
                updated,
                "artifact",
                "Created a cited summary artifact.",
                capability=capability,
                citations=summary.citations,
                transition_key=f"artifact:{milestone.identifier}:{session.action_count}",
            )
            return MissionStep(
                self._capabilities.complete_milestone(updated, milestone.identifier),
                None,
                1,
                True,
            )
        if capability == "generate_flashcards":
            flashcards = flashcard_artifact(payload, session.document_id)
            updated = replace(session, artifacts=session.artifacts + (flashcards,))
            updated = self._state.checkpoint(
                updated,
                "artifact",
                "Created cited flashcards.",
                capability=capability,
                citations=flashcards.citations,
                transition_key=f"artifact:{milestone.identifier}:{session.action_count}",
            )
            return MissionStep(
                self._capabilities.complete_milestone(updated, milestone.identifier),
                None,
                1,
                True,
            )
        if capability == "generate_quiz":
            return self._quiz(session, milestone, payload)
        if capability == "explain_concept":
            return self._explanation(session, milestone, payload)
        raise ValueError(f"Unsupported mission capability '{capability}'.")

    def _quiz(
        self,
        session: StudySession,
        milestone: StudyMilestone,
        payload: Mapping[str, object],
    ) -> MissionStep:
        artifact = quiz_artifact(payload, session.document_id)
        updated = replace(session, artifacts=session.artifacts + (artifact,))
        updated = self._state.checkpoint(
            updated,
            "artifact",
            "Created a cited quiz artifact.",
            capability=milestone.capability,
            citations=artifact.citations,
            transition_key=f"artifact:{milestone.identifier}:{session.action_count}",
        )
        updated = self._capabilities.complete_milestone(updated, milestone.identifier)
        questions = payload.get("questions")
        if milestone.identifier == "milestone-diagnostic" and isinstance(
            questions, list
        ):
            objective = self._policy.current_objective(updated)
            if objective is not None:
                pending = first_pending_question(
                    questions, objective.identifier, updated.document_id
                )
                if pending is not None:
                    return MissionStep(
                        self._state.set_pending(updated, pending),
                        question_activity(pending),
                        1,
                    )
        if milestone.identifier == "milestone-review":
            completed = self._state.complete(updated, "Final review completed.")
            return MissionStep(
                completed,
                TutorActivity(TutorTurnKind.RECAP, "Final review completed.", None, ()),
                1,
            )
        return MissionStep(updated, None, 1, True)

    def _explanation(
        self,
        session: StudySession,
        milestone: StudyMilestone,
        payload: Mapping[str, object],
    ) -> MissionStep:
        objective_id = milestone.objective_id
        if objective_id is None:
            raise ValueError("Explanation milestone has no objective.")
        explanation = text_value(payload, "explanation")
        check_question = text_value(payload, "check_question")
        citations = references(payload.get("citations"), session.document_id)
        updated = self._capabilities.complete_milestone(session, milestone.identifier)
        turn = TutorTurn(
            TutorTurnKind.EXPLANATION,
            "",
            explanation,
            objective_id,
            citations,
            None,
            datetime.now(UTC),
        )
        updated = replace(updated, turns=updated.turns + (turn,))
        pending = PendingLearnerInteraction(
            objective_id=objective_id,
            question=check_question,
            citations=citations,
        )
        updated = self._state.set_pending(updated, pending)
        return MissionStep(
            updated,
            TutorActivity(
                TutorTurnKind.EXPLANATION,
                explanation + f"\n\n**Check:** {check_question}",
                objective_id,
                citations,
            ),
            1,
        )
