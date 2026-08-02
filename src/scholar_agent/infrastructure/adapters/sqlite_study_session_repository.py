"""SQLite persistence for adaptive study sessions and document briefs."""

import json
import sqlite3
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from threading import RLock

from scholar_agent.domain.entities.mission_ledger import (
    MissionLedgerEntry,
    MissionLedgerEventType,
    MissionStateProjection,
    verify_ledger,
)
from scholar_agent.domain.entities.study_material import (
    Flashcard,
    FlashcardArtifact,
    QuizArtifact,
    QuizQuestion,
    SummaryArtifact,
)
from scholar_agent.domain.entities.study_session import (
    ConceptNode,
    DocumentBrief,
    GlossaryTerm,
    LearnerAttempt,
    LearnerLevel,
    LearningObjective,
    MilestoneKind,
    MilestoneStatus,
    MissionStatus,
    MissionTraceEvent,
    PendingLearnerInteraction,
    SourceReference,
    StudyArtifact,
    StudyMilestone,
    StudyMode,
    StudyPlan,
    StudySession,
    TutorTurn,
    TutorTurnKind,
)
from scholar_agent.domain.repositories.study_session_repository import (
    StudySessionRepository,
)
from scholar_agent.domain.value_objects.document_id import DocumentId


class SQLiteStudySessionRepository(StudySessionRepository):
    """Stores structured tutor state as versionable local JSON documents."""

    def __init__(self, database_path: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = RLock()
        self._initialize_schema()

    def save(self, session: StudySession) -> None:
        """Create or replace a session."""
        payload = json.dumps(_session_payload(session), ensure_ascii=False)
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO study_sessions
                    (session_id, document_id, payload, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    document_id = excluded.document_id,
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (
                    session.identifier,
                    session.document_id.value,
                    payload,
                    session.updated_at.isoformat(),
                ),
            )

    def get(self, session_id: str) -> StudySession | None:
        """Return a session when it exists."""
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM study_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(str(row["payload"]))
        if not isinstance(payload, dict):
            raise RuntimeError("Stored study session is invalid.")
        return _session_from_payload(payload)

    def list(
        self,
        document_id: DocumentId | None = None,
        status: MissionStatus | None = None,
    ) -> tuple[StudySession, ...]:
        """Return sessions ordered by updated timestamp descending."""
        clauses: list[str] = []
        values: list[str] = []
        if document_id is not None:
            clauses.append("document_id = ?")
            values.append(document_id.value)
        query = "SELECT payload FROM study_sessions"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC"
        with self._lock:
            rows = self._connection.execute(query, values).fetchall()
        sessions = tuple(
            _session_from_payload(json.loads(str(row["payload"]))) for row in rows
        )
        if status is None:
            return sessions
        return tuple(session for session in sessions if session.status is status)

    def complete(self, session_id: str) -> StudySession | None:
        """Mark one session complete without discarding its history."""
        from scholar_agent.application.services.mission_state import (
            MissionStateService,
        )

        session = self.get(session_id)
        if session is None:
            return None
        return MissionStateService(self).complete(
            session, "Learner manually completed the mission."
        )

    def delete(self, session_id: str) -> bool:
        """Delete a session and report whether it existed."""
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "DELETE FROM study_sessions WHERE session_id = ?",
                (session_id,),
            )
        return cursor.rowcount > 0

    def delete_for_document(self, document_id: DocumentId) -> None:
        """Delete every session and brief for a document."""
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM study_sessions WHERE document_id = ?",
                (document_id.value,),
            )
            self._connection.execute(
                "DELETE FROM document_briefs WHERE document_id = ?",
                (document_id.value,),
            )

    def detach_profile(self, profile_id: str) -> int:
        """Detach matching sessions while retaining their complete history."""
        with self._lock, self._connection:
            rows = self._connection.execute(
                "SELECT session_id, payload FROM study_sessions"
            ).fetchall()
            detached = 0
            for row in rows:
                payload = json.loads(str(row["payload"]))
                if not isinstance(payload, dict):
                    continue
                if payload.get("learner_profile_id") != profile_id:
                    continue
                session = _session_from_payload(payload)
                detached_session = replace(session, learner_profile_id=None)
                updated_payload = json.dumps(
                    _session_payload(detached_session), ensure_ascii=False
                )
                self._connection.execute(
                    "UPDATE study_sessions SET payload = ?, updated_at = ? "
                    "WHERE session_id = ?",
                    (
                        updated_payload,
                        detached_session.updated_at.isoformat(),
                        detached_session.identifier,
                    ),
                )
                detached += 1
        return detached

    def get_brief(self, document_id: DocumentId) -> DocumentBrief | None:
        """Return a cached document brief."""
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM document_briefs WHERE document_id = ?",
                (document_id.value,),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(str(row["payload"]))
        if not isinstance(payload, dict):
            raise RuntimeError("Stored document brief is invalid.")
        return _brief_from_payload(payload)

    def save_brief(self, brief: DocumentBrief) -> None:
        """Cache a document brief."""
        payload = json.dumps(_brief_payload(brief), ensure_ascii=False)
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO document_briefs (document_id, payload)
                VALUES (?, ?)
                ON CONFLICT(document_id) DO UPDATE SET payload = excluded.payload
                """,
                (brief.document_id.value, payload),
            )

    def close(self) -> None:
        """Close the underlying connection."""
        with self._lock:
            self._connection.close()

    def _initialize_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS study_sessions (
                    session_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_study_sessions_document
                ON study_sessions(document_id)
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS document_briefs (
                    document_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                )
                """
            )


def _reference_payload(reference: SourceReference) -> dict[str, object]:
    return {
        "document_id": reference.document_id.value,
        "chunk_id": reference.chunk_id,
        "page_number": reference.page_number,
        "excerpt": reference.excerpt,
    }


def _reference_from_payload(payload: object) -> SourceReference:
    item = _mapping(payload)
    return SourceReference(
        document_id=DocumentId(_string(item, "document_id")),
        chunk_id=_string(item, "chunk_id"),
        page_number=_optional_integer(item.get("page_number")),
        excerpt=_string(item, "excerpt"),
    )


def _brief_payload(brief: DocumentBrief) -> dict[str, object]:
    return {
        "document_id": brief.document_id.value,
        "synopsis": brief.synopsis,
        "objectives": [
            {
                "id": item.identifier,
                "title": item.title,
                "description": item.description,
                "prerequisites": list(item.prerequisite_ids),
                "citations": [_reference_payload(ref) for ref in item.citations],
            }
            for item in brief.objectives
        ],
        "concepts": [
            {
                "id": item.identifier,
                "label": item.label,
                "explanation": item.explanation,
                "prerequisites": list(item.prerequisite_ids),
                "citations": [_reference_payload(ref) for ref in item.citations],
            }
            for item in brief.concepts
        ],
        "glossary": [
            {
                "term": item.term,
                "definition": item.definition,
                "citations": [_reference_payload(ref) for ref in item.citations],
            }
            for item in brief.glossary
        ],
        "misconceptions": list(brief.misconceptions),
    }


def _brief_from_payload(payload: dict[str, object]) -> DocumentBrief:
    document_id = DocumentId(_string(payload, "document_id"))
    return DocumentBrief(
        document_id=document_id,
        synopsis=_string(payload, "synopsis"),
        objectives=tuple(
            LearningObjective(
                identifier=_string(item, "id"),
                title=_string(item, "title"),
                description=_string(item, "description"),
                prerequisite_ids=_strings(item.get("prerequisites")),
                citations=tuple(
                    _reference_from_payload(ref)
                    for ref in _objects(item.get("citations"))
                ),
            )
            for item in _objects(payload.get("objectives"))
        ),
        concepts=tuple(
            ConceptNode(
                identifier=_string(item, "id"),
                label=_string(item, "label"),
                explanation=_string(item, "explanation"),
                prerequisite_ids=_strings(item.get("prerequisites")),
                citations=tuple(
                    _reference_from_payload(ref)
                    for ref in _objects(item.get("citations"))
                ),
            )
            for item in _objects(payload.get("concepts"))
        ),
        glossary=tuple(
            GlossaryTerm(
                term=_string(item, "term"),
                definition=_string(item, "definition"),
                citations=tuple(
                    _reference_from_payload(ref)
                    for ref in _objects(item.get("citations"))
                ),
            )
            for item in _objects(payload.get("glossary"))
        ),
        misconceptions=_strings(payload.get("misconceptions")),
    )


def _attempt_payload(attempt: LearnerAttempt) -> dict[str, object]:
    return {
        "objective_id": attempt.objective_id,
        "response": attempt.response,
        "score": attempt.score,
        "feedback": attempt.feedback,
        "missing_concepts": list(attempt.missing_concepts),
        "citations": [_reference_payload(ref) for ref in attempt.citations],
        "created_at": attempt.created_at.isoformat(),
    }


def _attempt_from_payload(payload: object) -> LearnerAttempt:
    item = _mapping(payload)
    return LearnerAttempt(
        objective_id=_string(item, "objective_id"),
        response=_string(item, "response"),
        score=_integer(item, "score"),
        feedback=_string(item, "feedback"),
        missing_concepts=_strings(item.get("missing_concepts")),
        citations=tuple(
            _reference_from_payload(ref) for ref in _objects(item.get("citations"))
        ),
        created_at=_datetime(item, "created_at"),
    )


def _turn_payload(turn: TutorTurn) -> dict[str, object]:
    return {
        "kind": turn.kind.value,
        "learner_message": turn.learner_message,
        "tutor_message": turn.tutor_message,
        "objective_id": turn.objective_id,
        "citations": [_reference_payload(ref) for ref in turn.citations],
        "assessment": (
            _attempt_payload(turn.assessment) if turn.assessment is not None else None
        ),
        "created_at": turn.created_at.isoformat(),
    }


def _turn_from_payload(payload: object) -> TutorTurn:
    item = _mapping(payload)
    assessment = item.get("assessment")
    objective_id = item.get("objective_id")
    return TutorTurn(
        kind=TutorTurnKind(_string(item, "kind")),
        learner_message=_string(item, "learner_message"),
        tutor_message=_string(item, "tutor_message"),
        objective_id=str(objective_id) if objective_id is not None else None,
        citations=tuple(
            _reference_from_payload(ref) for ref in _objects(item.get("citations"))
        ),
        assessment=(
            _attempt_from_payload(assessment) if assessment is not None else None
        ),
        created_at=_datetime(item, "created_at"),
    )


def _session_payload(session: StudySession) -> dict[str, object]:
    return {
        "schema_version": 4,
        "identifier": session.identifier,
        "document_id": session.document_id.value,
        "goal": session.goal,
        "learner_level": session.learner_level.value,
        "mode": session.mode.value,
        "target_minutes": session.target_minutes,
        "brief": _brief_payload(session.brief),
        "attempts": [_attempt_payload(item) for item in session.attempts],
        "turns": [_turn_payload(item) for item in session.turns],
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "status": session.status.value,
        "plan": _plan_payload(session.plan),
        "milestones": [_milestone_payload(item) for item in session.milestones],
        "artifacts": [_artifact_payload(item) for item in session.artifacts],
        "pending_interaction": _pending_payload(session.pending_interaction),
        "trace": [_trace_payload(item) for item in session.trace],
        "ledger": [_ledger_payload(item) for item in session.ledger],
        "action_count": session.action_count,
        "learner_profile_id": session.learner_profile_id,
        "completed_at": (
            session.completed_at.isoformat()
            if session.completed_at is not None
            else None
        ),
    }


def _session_from_payload(payload: dict[str, object]) -> StudySession:
    schema_version = payload.get("schema_version", 1)
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version not in (1, 2, 3, 4)
    ):
        raise RuntimeError("Stored study session has an unsupported schema version.")
    is_v1 = schema_version == 1
    has_ledger = schema_version in (3, 4)
    is_v4 = schema_version == 4
    document_id = DocumentId(_string(payload, "document_id"))
    brief = _brief_from_payload(_mapping(payload.get("brief")))
    milestones = (
        _legacy_milestones(brief)
        if is_v1
        else tuple(
            _milestone_from_payload(item)
            for item in _objects(payload.get("milestones", []))
        )
    )
    raw_status = payload.get("status")
    status = (
        MissionStatus.ACTIVE
        if is_v1
        else MissionStatus(_string_value(raw_status, "status"))
    )
    completed_at = payload.get("completed_at")
    ledger = (
        tuple(
            _ledger_from_payload(item) for item in _objects(payload.get("ledger", []))
        )
        if has_ledger
        else ()
    )
    if has_ledger:
        if any(
            reference.document_id != document_id
            for entry in ledger
            for reference in entry.citations
        ):
            raise RuntimeError(
                "Stored study session ledger contains another document's citation."
            )
        verification = verify_ledger(ledger)
        if not verification.valid:
            detail = verification.reason or "Ledger verification failed."
            raise RuntimeError(
                f"Stored study session ledger is invalid at sequence "
                f"{verification.sequence}: {detail}"
            )
    return StudySession(
        identifier=_string(payload, "identifier"),
        document_id=document_id,
        goal=_string(payload, "goal"),
        learner_level=LearnerLevel(_string(payload, "learner_level")),
        mode=StudyMode(_string(payload, "mode")),
        target_minutes=_integer(payload, "target_minutes"),
        brief=brief,
        attempts=tuple(
            _attempt_from_payload(item) for item in _objects(payload.get("attempts"))
        ),
        turns=tuple(
            _turn_from_payload(item) for item in _objects(payload.get("turns"))
        ),
        created_at=_datetime(payload, "created_at"),
        updated_at=_datetime(payload, "updated_at"),
        status=status,
        plan=None if is_v1 else _plan_from_payload(payload.get("plan")),
        milestones=milestones,
        artifacts=(
            ()
            if is_v1
            else tuple(
                _artifact_from_payload(item)
                for item in _objects(payload.get("artifacts", []))
            )
        ),
        pending_interaction=(
            None if is_v1 else _pending_from_payload(payload.get("pending_interaction"))
        ),
        trace=(
            ()
            if is_v1
            else tuple(
                _trace_from_payload(item) for item in _objects(payload.get("trace", []))
            )
        ),
        ledger=ledger,
        action_count=(
            0 if is_v1 else _optional_nonnegative_integer(payload.get("action_count"))
        ),
        completed_at=(
            None
            if completed_at is None
            else datetime.fromisoformat(_string_value(completed_at, "completed_at"))
        ),
        learner_profile_id=(
            _optional_string(payload.get("learner_profile_id")) if is_v4 else None
        ),
    )


def _ledger_payload(entry: MissionLedgerEntry) -> dict[str, object]:
    return {
        "sequence": entry.sequence,
        "event_type": (
            entry.event_type.value
            if isinstance(entry.event_type, MissionLedgerEventType)
            else entry.event_type
        ),
        "summary": entry.summary,
        "objective_id": entry.objective_id,
        "capability": entry.capability,
        "citations": [_reference_payload(item) for item in entry.citations],
        "projection": _projection_payload(entry.projection),
        "previous_digest": entry.previous_digest,
        "current_digest": entry.current_digest,
        "created_at": entry.created_at.isoformat(),
        "transition_key": entry.transition_key,
    }


def _ledger_from_payload(payload: object) -> MissionLedgerEntry:
    item = _mapping(payload)
    objective_id = item.get("objective_id")
    capability = item.get("capability")
    transition_key = item.get("transition_key")
    return MissionLedgerEntry(
        sequence=_integer(item, "sequence"),
        event_type=MissionLedgerEventType(_string(item, "event_type")),
        summary=_string(item, "summary"),
        objective_id=str(objective_id) if objective_id is not None else None,
        capability=str(capability) if capability is not None else None,
        citations=tuple(
            _reference_from_payload(reference)
            for reference in _objects(item.get("citations", []))
        ),
        projection=_projection_from_payload(item.get("projection")),
        previous_digest=_string(item, "previous_digest"),
        current_digest=_string(item, "current_digest"),
        created_at=_datetime(item, "created_at"),
        transition_key=(str(transition_key) if transition_key is not None else None),
    )


def _projection_payload(projection: MissionStateProjection) -> dict[str, object]:
    return {
        "status": projection.status,
        "active_milestone_id": projection.active_milestone_id,
        "pending_objective_id": projection.pending_objective_id,
        "action_count": projection.action_count,
        "attempt_count": projection.attempt_count,
        "artifact_count": projection.artifact_count,
        "completed_milestone_count": projection.completed_milestone_count,
        "mastery_by_objective": [
            {"objective_id": objective_id, "label": label}
            for objective_id, label in projection.mastery_by_objective
        ],
        "next_milestone_id": projection.next_milestone_id,
    }


def _projection_from_payload(value: object) -> MissionStateProjection:
    item = _mapping(value)
    mastery = tuple(
        (
            _string(master, "objective_id"),
            _string(master, "label"),
        )
        for master in _objects(item.get("mastery_by_objective", []))
    )
    active = item.get("active_milestone_id")
    pending = item.get("pending_objective_id")
    next_milestone = item.get("next_milestone_id")
    return MissionStateProjection(
        status=_string(item, "status"),
        active_milestone_id=str(active) if active is not None else None,
        pending_objective_id=str(pending) if pending is not None else None,
        action_count=_integer(item, "action_count"),
        attempt_count=_integer(item, "attempt_count"),
        artifact_count=_integer(item, "artifact_count"),
        completed_milestone_count=_integer(item, "completed_milestone_count"),
        mastery_by_objective=mastery,
        next_milestone_id=(str(next_milestone) if next_milestone is not None else None),
    )


def _plan_payload(plan: StudyPlan | None) -> dict[str, object] | None:
    if plan is None:
        return None
    return {
        "focus": plan.focus,
        "objective_ids": list(plan.objective_ids),
        "citations": [_reference_payload(item) for item in plan.citations],
    }


def _plan_from_payload(value: object) -> StudyPlan | None:
    if value is None:
        return None
    item = _mapping(value)
    return StudyPlan(
        focus=_string(item, "focus"),
        objective_ids=_strings(item.get("objective_ids")),
        citations=tuple(
            _reference_from_payload(reference)
            for reference in _objects(item.get("citations", []))
        ),
    )


def _milestone_payload(milestone: StudyMilestone) -> dict[str, object]:
    return {
        "id": milestone.identifier,
        "kind": milestone.kind.value,
        "title": milestone.title,
        "objective_id": milestone.objective_id,
        "capability": milestone.capability,
        "status": milestone.status.value,
        "citations": [_reference_payload(item) for item in milestone.citations],
    }


def _milestone_from_payload(payload: object) -> StudyMilestone:
    item = _mapping(payload)
    objective_id = item.get("objective_id")
    return StudyMilestone(
        identifier=_string(item, "id"),
        kind=MilestoneKind(_string(item, "kind")),
        title=_string(item, "title"),
        objective_id=(str(objective_id) if objective_id is not None else None),
        capability=_string(item, "capability"),
        status=MilestoneStatus(_string(item, "status")),
        citations=tuple(
            _reference_from_payload(reference)
            for reference in _objects(item.get("citations", []))
        ),
    )


def _legacy_milestones(brief: DocumentBrief) -> tuple[StudyMilestone, ...]:
    milestones: list[StudyMilestone] = [
        StudyMilestone(
            identifier="milestone-orient",
            kind=MilestoneKind.ORIENT,
            title="Orient in the document",
            objective_id=None,
            capability="build_document_map",
            status=MilestoneStatus.ACTIVE,
            citations=(),
        )
    ]
    for objective in brief.objectives:
        milestones.extend(
            (
                StudyMilestone(
                    identifier=f"milestone-learn-{objective.identifier}",
                    kind=MilestoneKind.LEARN,
                    title=objective.title,
                    objective_id=objective.identifier,
                    capability="explain_concept",
                    citations=objective.citations,
                ),
                StudyMilestone(
                    identifier=f"milestone-practice-{objective.identifier}",
                    kind=MilestoneKind.PRACTICE,
                    title=f"Practice {objective.title}",
                    objective_id=objective.identifier,
                    capability="assess_learner_response",
                    citations=objective.citations,
                ),
            )
        )
    milestones.append(
        StudyMilestone(
            identifier="milestone-review",
            kind=MilestoneKind.REVIEW,
            title="Review and recap",
            objective_id=None,
            capability="generate_quiz",
            citations=(),
        )
    )
    return tuple(milestones)


def _pending_payload(
    interaction: PendingLearnerInteraction | None,
) -> dict[str, object] | None:
    if interaction is None:
        return None
    return {
        "objective_id": interaction.objective_id,
        "question": interaction.question,
        "capability": interaction.capability,
        "reference_answer": interaction.reference_answer,
        "citations": [_reference_payload(item) for item in interaction.citations],
        "attempts": interaction.attempts,
    }


def _pending_from_payload(value: object) -> PendingLearnerInteraction | None:
    if value is None:
        return None
    item = _mapping(value)
    reference_answer = item.get("reference_answer")
    return PendingLearnerInteraction(
        objective_id=_string(item, "objective_id"),
        question=_string(item, "question"),
        capability=_string(item, "capability"),
        reference_answer=(
            str(reference_answer) if reference_answer is not None else None
        ),
        citations=tuple(
            _reference_from_payload(reference)
            for reference in _objects(item.get("citations", []))
        ),
        attempts=_optional_nonnegative_integer(item.get("attempts")),
    )


def _trace_payload(event: MissionTraceEvent) -> dict[str, object]:
    return {
        "event_type": event.event_type,
        "summary": event.summary,
        "capability": event.capability,
        "state": event.state,
        "created_at": event.created_at.isoformat(),
    }


def _trace_from_payload(payload: object) -> MissionTraceEvent:
    item = _mapping(payload)
    capability = item.get("capability")
    state = item.get("state")
    return MissionTraceEvent(
        event_type=_string(item, "event_type"),
        summary=_string(item, "summary"),
        capability=str(capability) if capability is not None else None,
        state=str(state) if state is not None else None,
        created_at=_datetime(item, "created_at"),
    )


def _artifact_payload(artifact: object) -> dict[str, object]:
    if isinstance(artifact, SummaryArtifact):
        return {
            "kind": "summary",
            "text": artifact.text,
            "citations": [_reference_payload(item) for item in artifact.citations],
            "created_at": artifact.created_at.isoformat(),
        }
    if isinstance(artifact, QuizArtifact):
        return {
            "kind": "quiz",
            "questions": [
                {
                    "prompt": item.prompt,
                    "answer": item.answer,
                    "citations": [
                        _reference_payload(reference) for reference in item.citations
                    ],
                }
                for item in artifact.questions
            ],
            "citations": [_reference_payload(item) for item in artifact.citations],
            "created_at": artifact.created_at.isoformat(),
        }
    if isinstance(artifact, FlashcardArtifact):
        return {
            "kind": "flashcards",
            "cards": [
                {
                    "front": item.front,
                    "back": item.back,
                    "citations": [
                        _reference_payload(reference) for reference in item.citations
                    ],
                }
                for item in artifact.cards
            ],
            "citations": [_reference_payload(item) for item in artifact.citations],
            "created_at": artifact.created_at.isoformat(),
        }
    raise RuntimeError("Stored study artifact has an unsupported type.")


def _artifact_from_payload(payload: object) -> StudyArtifact:
    item = _mapping(payload)
    kind = _string(item, "kind")
    citations = tuple(
        _reference_from_payload(reference)
        for reference in _objects(item.get("citations", []))
    )
    created_at = datetime.fromisoformat(_string(item, "created_at"))
    if kind == "summary":
        return SummaryArtifact(_string(item, "text"), citations, created_at)
    if kind == "quiz":
        questions = tuple(
            QuizQuestion(
                prompt=_string(question, "prompt"),
                answer=_string(question, "answer"),
                citations=tuple(
                    _reference_from_payload(reference)
                    for reference in _objects(question.get("citations", []))
                ),
            )
            for question in _objects(item.get("questions", []))
        )
        return QuizArtifact(questions, citations, created_at)
    if kind == "flashcards":
        cards = tuple(
            Flashcard(
                front=_string(card, "front"),
                back=_string(card, "back"),
                citations=tuple(
                    _reference_from_payload(reference)
                    for reference in _objects(card.get("citations", []))
                ),
            )
            for card in _objects(item.get("cards", []))
        )
        return FlashcardArtifact(cards, citations, created_at)
    raise RuntimeError(f"Stored study artifact kind '{kind}' is unsupported.")


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError("Stored study data must contain an object.")
    return {str(key): item for key, item in value.items()}


def _objects(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        raise RuntimeError("Stored study data must contain an object array.")
    return tuple(_mapping(item) for item in value)


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RuntimeError("Stored study data must contain a string array.")
    return tuple(value)


def _string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise RuntimeError(f"Stored study field '{key}' must be text.")
    return value


def _integer(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"Stored study field '{key}' must be an integer.")
    return value


def _optional_integer(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError("Stored page number must be an integer or null.")
    return value


def _datetime(payload: dict[str, object], key: str) -> datetime:
    return datetime.fromisoformat(_string(payload, key))


def _string_value(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"Stored study field '{field}' must be text.")
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("Stored optional string is invalid.")
    return value


def _optional_nonnegative_integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError("Stored study count must be a non-negative integer.")
    return value
