"""SQLite persistence for adaptive study sessions and document briefs."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from threading import RLock

from scholar_agent.domain.entities.study_session import (
    ConceptNode,
    DocumentBrief,
    GlossaryTerm,
    LearnerAttempt,
    LearnerLevel,
    LearningObjective,
    SourceReference,
    StudyMode,
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
    }


def _session_from_payload(payload: dict[str, object]) -> StudySession:
    return StudySession(
        identifier=_string(payload, "identifier"),
        document_id=DocumentId(_string(payload, "document_id")),
        goal=_string(payload, "goal"),
        learner_level=LearnerLevel(_string(payload, "learner_level")),
        mode=StudyMode(_string(payload, "mode")),
        target_minutes=_integer(payload, "target_minutes"),
        brief=_brief_from_payload(_mapping(payload.get("brief"))),
        attempts=tuple(
            _attempt_from_payload(item) for item in _objects(payload.get("attempts"))
        ),
        turns=tuple(
            _turn_from_payload(item) for item in _objects(payload.get("turns"))
        ),
        created_at=_datetime(payload, "created_at"),
        updated_at=_datetime(payload, "updated_at"),
    )


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
