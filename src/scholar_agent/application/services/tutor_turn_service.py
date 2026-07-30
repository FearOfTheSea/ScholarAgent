"""Application behavior for one grounded adaptive tutor turn."""

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from scholar_agent.application.dtos.retrieval import RetrievedChunk
from scholar_agent.application.dtos.tutor import (
    ContinueStudySessionRequest,
    TutorActivity,
    TutorTurnResult,
)
from scholar_agent.application.output_ports.llm_provider import ILLMProvider
from scholar_agent.application.output_ports.retriever import IRetriever
from scholar_agent.domain.entities.study_session import (
    LearnerAttempt,
    LearningObjective,
    MasteryLabel,
    ObjectiveProgress,
    SourceReference,
    StudySession,
    TutorTurn,
    TutorTurnKind,
    objective_progress,
)
from scholar_agent.domain.repositories.study_session_repository import (
    StudySessionRepository,
)


@dataclass(frozen=True, slots=True)
class PreparedTutorTurn:
    """A completed turn awaiting persistence."""

    session: StudySession
    result: TutorTurnResult


class TutorTurnService:
    """Owns explicit routing, grounding, assessment, and mastery policy."""

    def __init__(
        self,
        llm_provider: ILLMProvider,
        retriever: IRetriever,
        session_repository: StudySessionRepository,
    ) -> None:
        self._llm_provider = llm_provider
        self._retriever = retriever
        self._session_repository = session_repository

    def classify(self, request: ContinueStudySessionRequest) -> str:
        """Classify a turn with transparent, deterministic rules."""
        lowered = request.message.casefold()
        if any(
            phrase in lowered
            for phrase in (
                "other pdf",
                "another document",
                "compare",
                "internet",
                "web",
            )
        ):
            return "unsupported"
        if "hint" in lowered:
            return "hint"
        if any(word in lowered for word in ("recap", "review", "progress")):
            return "recap"
        if any(
            phrase in lowered
            for phrase in ("quiz me", "test me", "ask me", "next question")
        ):
            return "question"
        if request.message.rstrip().endswith("?") or any(
            word in lowered for word in ("explain", "teach", "what is", "why ", "how ")
        ):
            return "explain"
        return "answer"

    def prepare(
        self,
        request: ContinueStudySessionRequest,
        intent: str,
    ) -> PreparedTutorTurn:
        """Generate, verify, and score one turn without persisting it."""
        session = self._session_repository.get(request.session_id)
        if session is None:
            raise ValueError(f"Study session '{request.session_id}' was not found.")
        objective = self._select_objective(session)
        assessment: LearnerAttempt | None = None
        if intent == "unsupported":
            activity = TutorActivity(
                kind=TutorTurnKind.UNSUPPORTED,
                message=(
                    "This tutor is intentionally grounded in one selected document. "
                    "I cannot compare another source or browse the web in this session."
                ),
                objective_id=objective.identifier,
                citations=(),
            )
        elif intent == "answer":
            assessment, activity = self._assess(session, objective, request.message)
        elif intent == "hint":
            activity = self._grounded_activity(
                session,
                objective,
                TutorTurnKind.HINT,
                "Give one progressive hint without revealing the full answer.",
            )
        elif intent == "recap":
            activity = self._recap(session, objective)
        elif intent == "question":
            activity = self._grounded_activity(
                session,
                objective,
                TutorTurnKind.QUESTION,
                "Ask one challenging short-answer question. Do not provide the answer.",
            )
        else:
            activity = self._explain(session, objective, request.message)

        now = datetime.now(UTC)
        attempts = session.attempts + ((assessment,) if assessment is not None else ())
        turn = TutorTurn(
            kind=activity.kind,
            learner_message=request.message,
            tutor_message=activity.message,
            objective_id=activity.objective_id,
            citations=activity.citations,
            assessment=assessment,
            created_at=now,
        )
        updated = replace(
            session,
            attempts=attempts,
            turns=session.turns + (turn,),
            updated_at=now,
        )
        progress = self._progress(updated)
        current = self._select_objective(updated, progress)
        return PreparedTutorTurn(
            session=updated,
            result=TutorTurnResult(
                intent=intent,
                activity=activity,
                assessment=assessment,
                progress=progress,
                current_objective_id=current.identifier,
            ),
        )

    def persist(self, prepared: PreparedTutorTurn) -> TutorTurnResult:
        """Persist a prepared turn and expose its typed result."""
        self._session_repository.save(prepared.session)
        return prepared.result

    def _assess(
        self,
        session: StudySession,
        objective: LearningObjective,
        response: str,
    ) -> tuple[LearnerAttempt, TutorActivity]:
        prompt = (
            "Assess the learner response using only the sources. Return JSON only: "
            '{"score":0,"feedback":"...","missing_concepts":["..."],'
            '"next_question":"..."}. score is 0-3: 0 unsupported/incorrect, '
            "1 partial, 2 mostly correct, 3 correct and complete. Feedback must be "
            "specific and next_question must not reveal its answer.\n\n"
            f"LEVEL: {session.learner_level.value}\n"
            f"OBJECTIVE: {objective.title}: {objective.description}\n"
            f"LEARNER RESPONSE: {response}\n\n"
            f"SOURCES:\n{_source_text(objective.citations)}"
        )
        raw = self._llm_provider.generate(prompt)
        payload = _json_object(raw)
        score = payload.get("score")
        feedback = payload.get("feedback")
        missing = payload.get("missing_concepts")
        next_question = payload.get("next_question")
        if (
            isinstance(score, bool)
            or not isinstance(score, int)
            or score < 0
            or score > 3
            or not isinstance(feedback, str)
            or not feedback.strip()
            or not isinstance(next_question, str)
            or not next_question.strip()
            or not isinstance(missing, list)
            or not all(isinstance(item, str) for item in missing)
        ):
            raise ValueError("The tutor returned an invalid assessment.")
        verified = self._verify(feedback.strip(), objective.citations)
        attempt = LearnerAttempt(
            objective_id=objective.identifier,
            response=response,
            score=score,
            feedback=verified,
            missing_concepts=tuple(item.strip() for item in missing if item.strip()),
            citations=objective.citations,
            created_at=datetime.now(UTC),
        )
        message = (
            f"**Assessment: {score}/3.** {verified}\n\n"
            f"**Next challenge:** {next_question.strip()}"
        )
        return attempt, TutorActivity(
            kind=TutorTurnKind.ASSESSMENT,
            message=message,
            objective_id=objective.identifier,
            citations=objective.citations,
        )

    def _explain(
        self,
        session: StudySession,
        objective: LearningObjective,
        question: str,
    ) -> TutorActivity:
        chunks = self._retriever.retrieve(
            question,
            document_ids=(session.document_id,),
        )
        references = tuple(_reference(chunk) for chunk in chunks)
        if not references:
            return TutorActivity(
                TutorTurnKind.UNSUPPORTED,
                "The selected document does not provide enough evidence for that.",
                objective.identifier,
                (),
            )
        prompt = (
            f"Answer at a {session.learner_level.value} level using only the sources. "
            "Use a concise explanation, one analogy when useful, and one check-for-"
            "understanding question. Do not claim anything absent from the sources.\n"
            f"LEARNER QUESTION: {question}\nSOURCES:\n{_source_text(references)}"
        )
        message = self._llm_provider.generate(prompt)
        return TutorActivity(
            TutorTurnKind.EXPLANATION,
            self._verify(message, references),
            objective.identifier,
            references,
        )

    def _grounded_activity(
        self,
        session: StudySession,
        objective: LearningObjective,
        kind: TutorTurnKind,
        instruction: str,
    ) -> TutorActivity:
        prompt = (
            f"You are a {session.mode.value} tutor for a {session.learner_level.value} "
            f"learner. {instruction} Use only the sources.\n"
            f"OBJECTIVE: {objective.title}: {objective.description}\n"
            f"SOURCES:\n{_source_text(objective.citations)}"
        )
        message = self._llm_provider.generate(prompt)
        return TutorActivity(
            kind,
            self._verify(message, objective.citations),
            objective.identifier,
            objective.citations,
        )

    def _recap(
        self,
        session: StudySession,
        objective: LearningObjective,
    ) -> TutorActivity:
        progress = self._progress(session)
        lines = [
            f"- {item.objective_id}: {item.label.value} ({item.percentage}%)"
            for item in progress
        ]
        return TutorActivity(
            TutorTurnKind.RECAP,
            (
                "Here is your evidence-based mastery snapshot:\n"
                + "\n".join(lines)
                + f"\n\nRecommended focus: **{objective.title}**."
            ),
            objective.identifier,
            objective.citations,
        )

    def _verify(
        self,
        response: str,
        citations: tuple[SourceReference, ...],
    ) -> str:
        prompt = (
            "Verify whether every factual claim in RESPONSE is supported by SOURCES. "
            "Return JSON only with supported (boolean) and response (a corrected, "
            "fully supported response). Keep the original when already supported.\n"
            f"RESPONSE:\n{response}\n\nSOURCES:\n{_source_text(citations)}"
        )
        raw = self._llm_provider.generate(prompt)
        payload = _json_object(raw)
        supported = payload.get("supported")
        revised = payload.get("response")
        if not isinstance(supported, bool) or not isinstance(revised, str):
            raise ValueError("The tutor returned an invalid grounding verification.")
        if not revised.strip():
            raise ValueError("Grounding verification returned an empty response.")
        return revised.strip()

    def _select_objective(
        self,
        session: StudySession,
        progress: tuple[ObjectiveProgress, ...] | None = None,
    ) -> LearningObjective:
        progress_items = progress or self._progress(session)
        by_id = {item.objective_id: item for item in progress_items}
        ready = [
            objective
            for objective in session.brief.objectives
            if all(
                by_id[prerequisite].label
                in (MasteryLabel.PROFICIENT, MasteryLabel.MASTERED)
                for prerequisite in objective.prerequisite_ids
            )
        ]
        candidates = ready or list(session.brief.objectives)
        return min(
            candidates,
            key=lambda objective: (
                by_id[objective.identifier].percentage,
                by_id[objective.identifier].attempt_count,
            ),
        )

    @staticmethod
    def _progress(session: StudySession) -> tuple[ObjectiveProgress, ...]:
        return tuple(
            objective_progress(objective.identifier, session.attempts)
            for objective in session.brief.objectives
        )


def _reference(chunk: RetrievedChunk) -> SourceReference:
    return SourceReference(
        document_id=chunk.document_id,
        chunk_id=chunk.chunk_id,
        page_number=chunk.page_number,
        excerpt=chunk.content[:500],
    )


def _source_text(references: tuple[SourceReference, ...]) -> str:
    return "\n\n".join(
        f"[{item.chunk_id}|page={item.page_number}]\n{item.excerpt}"
        for item in references
    )


def _json_object(raw: str) -> dict[str, object]:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("The tutor response does not contain JSON.")
    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as error:
        raise ValueError("The tutor response contains invalid JSON.") from error
    if not isinstance(payload, dict):
        raise ValueError("The tutor response must be a JSON object.")
    return {str(key): value for key, value in payload.items()}
