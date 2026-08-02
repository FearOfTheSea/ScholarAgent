"""SQLite adapter for redacted local learner profiles and evidence."""

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from threading import RLock

from scholar_agent.domain.entities.learner_profile import (
    ConceptEquivalenceCandidate,
    ConceptEquivalenceLink,
    ConceptFingerprint,
    EquivalenceDecision,
    EvidenceObservation,
    LearnerProfile,
    ObservationModality,
    ObservationSource,
)
from scholar_agent.domain.repositories.learner_profile_repository import (
    LearnerProfileRepository,
)
from scholar_agent.domain.repositories.study_session_repository import (
    StudySessionRepository,
)
from scholar_agent.domain.value_objects.citation_identity import CitationIdentity
from scholar_agent.domain.value_objects.document_id import DocumentId


class SQLiteLearnerProfileRepository(LearnerProfileRepository):
    """Store profile state in a separate local database with explicit cascades."""

    def __init__(
        self,
        database_path: Path,
        session_repository: StudySessionRepository | None = None,
    ) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._session_repository = session_repository
        self._lock = RLock()
        self._initialize_schema()

    def save_profile(self, profile: LearnerProfile) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO learner_profiles
                    (profile_id, display_name, target_date, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(profile_id) DO UPDATE SET
                    display_name = excluded.display_name,
                    target_date = excluded.target_date,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at
                """,
                (
                    profile.identifier,
                    profile.display_name,
                    profile.target_date.isoformat() if profile.target_date else None,
                    profile.created_at.isoformat(),
                    profile.updated_at.isoformat(),
                ),
            )

    def get_profile(self, profile_id: str) -> LearnerProfile | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM learner_profiles WHERE profile_id = ?", (profile_id,)
            ).fetchone()
        return _profile_from_row(row) if row is not None else None

    def list_profiles(self) -> tuple[LearnerProfile, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM learner_profiles ORDER BY display_name, profile_id"
            ).fetchall()
        return tuple(_profile_from_row(row) for row in rows)

    def get_or_create_default(self, now: datetime) -> LearnerProfile:
        existing = self.get_profile("local-default")
        if existing is not None:
            return existing
        profile = LearnerProfile.local_default(now)
        self.save_profile(profile)
        return profile

    def delete_profile(self, profile_id: str) -> int:
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM learner_profiles WHERE profile_id = ?", (profile_id,)
            )
            self._connection.execute(
                "DELETE FROM learner_observations WHERE profile_id = ?", (profile_id,)
            )
            self._connection.execute(
                "DELETE FROM equivalence_candidates WHERE profile_id = ?",
                (profile_id,),
            )
            self._connection.execute(
                "DELETE FROM equivalence_links WHERE profile_id = ?", (profile_id,)
            )
        if self._session_repository is None:
            return 0
        return self._session_repository.detach_profile(profile_id)

    def append_observation(self, observation: EvidenceObservation) -> bool:
        payload = json.dumps(_observation_payload(observation), ensure_ascii=False)
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT OR IGNORE INTO learner_observations
                    (observation_id, profile_id, observed_at, payload)
                VALUES (?, ?, ?, ?)
                """,
                (
                    observation.identifier,
                    observation.profile_id,
                    observation.observed_at.isoformat(),
                    payload,
                ),
            )
        return cursor.rowcount > 0

    def list_observations(self, profile_id: str) -> tuple[EvidenceObservation, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT payload FROM learner_observations
                WHERE profile_id = ? ORDER BY observed_at, observation_id
                """,
                (profile_id,),
            ).fetchall()
        return tuple(_observation_from_payload(json.loads(str(row[0]))) for row in rows)

    def save_candidate(self, candidate: ConceptEquivalenceCandidate) -> None:
        if not candidate.profile_id:
            raise ValueError("Equivalence candidate requires a profile id.")
        key = _pair_key(candidate.source, candidate.target)
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO equivalence_candidates
                    (candidate_key, profile_id, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(profile_id, candidate_key)
                DO UPDATE SET payload = excluded.payload
                """,
                (
                    key,
                    candidate.profile_id,
                    json.dumps(_candidate_payload(candidate), ensure_ascii=False),
                ),
            )

    def list_candidates(
        self, profile_id: str
    ) -> tuple[ConceptEquivalenceCandidate, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT payload FROM equivalence_candidates
                WHERE profile_id = ? ORDER BY candidate_key
                """,
                (profile_id,),
            ).fetchall()
        return tuple(_candidate_from_payload(json.loads(str(row[0]))) for row in rows)

    def save_equivalence_link(self, link: ConceptEquivalenceLink) -> None:
        if not link.profile_id:
            raise ValueError("Equivalence decision requires a profile id.")
        key = _pair_key(link.source, link.target)
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO equivalence_links (link_key, profile_id, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(profile_id, link_key)
                DO UPDATE SET payload = excluded.payload
                """,
                (
                    key,
                    link.profile_id,
                    json.dumps(_link_payload(link), ensure_ascii=False),
                ),
            )

    def list_equivalence_links(
        self, profile_id: str
    ) -> tuple[ConceptEquivalenceLink, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT payload FROM equivalence_links
                WHERE profile_id = ? ORDER BY link_key
                """,
                (profile_id,),
            ).fetchall()
        return tuple(_link_from_payload(json.loads(str(row[0]))) for row in rows)

    def replace_profile_data(
        self,
        profile: LearnerProfile,
        observations: tuple[EvidenceObservation, ...],
        candidates: tuple[ConceptEquivalenceCandidate, ...],
        links: tuple[ConceptEquivalenceLink, ...],
    ) -> None:
        if any(item.profile_id != profile.identifier for item in observations):
            raise ValueError("Imported observation belongs to another profile.")
        if any(item.profile_id != profile.identifier for item in candidates + links):
            raise ValueError("Imported equivalence data belongs to another profile.")
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM learner_profiles WHERE profile_id = ?",
                (profile.identifier,),
            )
            self._connection.execute(
                "DELETE FROM learner_observations WHERE profile_id = ?",
                (profile.identifier,),
            )
            self._connection.execute(
                "DELETE FROM equivalence_candidates WHERE profile_id = ?",
                (profile.identifier,),
            )
            self._connection.execute(
                "DELETE FROM equivalence_links WHERE profile_id = ?",
                (profile.identifier,),
            )
            self._connection.execute(
                """
                INSERT INTO learner_profiles
                    (profile_id, display_name, target_date, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    profile.identifier,
                    profile.display_name,
                    profile.target_date.isoformat() if profile.target_date else None,
                    profile.created_at.isoformat(),
                    profile.updated_at.isoformat(),
                ),
            )
            for observation in observations:
                self._connection.execute(
                    "INSERT INTO learner_observations "
                    "(observation_id, profile_id, observed_at, payload) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        observation.identifier,
                        observation.profile_id,
                        observation.observed_at.isoformat(),
                        json.dumps(
                            _observation_payload(observation), ensure_ascii=False
                        ),
                    ),
                )
            for candidate in candidates:
                self._connection.execute(
                    "INSERT INTO equivalence_candidates "
                    "(candidate_key, profile_id, payload) VALUES (?, ?, ?)",
                    (
                        _pair_key(candidate.source, candidate.target),
                        candidate.profile_id,
                        json.dumps(_candidate_payload(candidate), ensure_ascii=False),
                    ),
                )
            for link in links:
                self._connection.execute(
                    "INSERT INTO equivalence_links (link_key, profile_id, payload) "
                    "VALUES (?, ?, ?)",
                    (
                        _pair_key(link.source, link.target),
                        link.profile_id,
                        json.dumps(_link_payload(link), ensure_ascii=False),
                    ),
                )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _initialize_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS learner_profiles (
                    profile_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    target_date TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS learner_observations (
                    observation_id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            self._ensure_equivalence_table("equivalence_candidates", "candidate_key")
            self._ensure_equivalence_table("equivalence_links", "link_key")

    def _ensure_equivalence_table(self, table_name: str, key_name: str) -> None:
        columns = self._connection.execute(
            f"PRAGMA table_info({table_name})"
        ).fetchall()
        if not columns:
            self._create_equivalence_table(table_name, key_name)
            return
        primary_keys = {
            str(row["name"]): int(row["pk"]) for row in columns if int(row["pk"]) > 0
        }
        if primary_keys == {"profile_id": 1, key_name: 2}:
            return

        legacy_table = f"{table_name}_legacy_v1"
        legacy_exists = self._connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (legacy_table,),
        ).fetchone()
        if legacy_exists is not None:
            raise RuntimeError(
                f"Incomplete learner-profile migration table '{legacy_table}'."
            )
        self._connection.execute(f"ALTER TABLE {table_name} RENAME TO {legacy_table}")
        self._create_equivalence_table(table_name, key_name)
        self._migrate_legacy_rows(table_name, key_name, legacy_table)
        self._connection.execute(f"DROP TABLE {legacy_table}")

    def _create_equivalence_table(self, table_name: str, key_name: str) -> None:
        self._connection.execute(
            f"""
            CREATE TABLE {table_name} (
                {key_name} TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (profile_id, {key_name})
            )
            """
        )

    def _migrate_legacy_rows(
        self, table_name: str, key_name: str, legacy_table: str
    ) -> None:
        """Reconcile legacy ownership from validated payloads, failing closed."""
        rows = self._connection.execute(
            f"SELECT {key_name}, profile_id, payload FROM {legacy_table}"
        ).fetchall()
        for row in rows:
            try:
                payload = json.loads(str(row["payload"]))
                item = _mapping(payload)
                owner = item.get("profile_id")
                if not isinstance(owner, str) or not owner.strip():
                    continue
                if self.get_profile(owner) is None:
                    continue
                if table_name == "equivalence_candidates":
                    candidate = _candidate_from_payload(payload)
                    value_profile_id = candidate.profile_id
                    serialized = _candidate_payload(candidate)
                    pair_key = _pair_key(candidate.source, candidate.target)
                else:
                    link = _link_from_payload(payload)
                    value_profile_id = link.profile_id
                    serialized = _link_payload(link)
                    pair_key = _pair_key(link.source, link.target)
                if value_profile_id != owner or pair_key != str(row[key_name]):
                    continue
                self._connection.execute(
                    f"INSERT INTO {table_name} ({key_name}, profile_id, payload) "
                    "VALUES (?, ?, ?)",
                    (
                        pair_key,
                        owner,
                        json.dumps(serialized, ensure_ascii=False),
                    ),
                )
            except (
                KeyError,
                TypeError,
                ValueError,
                RuntimeError,
                json.JSONDecodeError,
            ):
                continue


def _profile_from_row(row: sqlite3.Row) -> LearnerProfile:
    raw_target = row["target_date"]
    return LearnerProfile(
        str(row["profile_id"]),
        str(row["display_name"]),
        date.fromisoformat(str(raw_target)) if raw_target else None,
        datetime.fromisoformat(str(row["created_at"])),
        datetime.fromisoformat(str(row["updated_at"])),
    )


def _fingerprint_payload(fingerprint: ConceptFingerprint) -> dict[str, object]:
    return {
        "algorithm_version": fingerprint.algorithm_version,
        "value": fingerprint.value,
        "document_id": fingerprint.document_id.value,
        "normalized_title": fingerprint.normalized_title,
        "normalized_description": fingerprint.normalized_description,
    }


def _fingerprint_from_payload(payload: object) -> ConceptFingerprint:
    item = _mapping(payload)
    return ConceptFingerprint(
        str(item["algorithm_version"]),
        str(item["value"]),
        DocumentId(str(item["document_id"])),
        str(item["normalized_title"]),
        str(item["normalized_description"]),
    )


def _citation_payload(citation: CitationIdentity) -> dict[str, object]:
    return {
        "document_id": citation.document_id.value,
        "chunk_id": citation.chunk_id,
        "page_number": citation.page_number,
    }


def _citation_from_payload(payload: object) -> CitationIdentity:
    item = _mapping(payload)
    return CitationIdentity(
        DocumentId(str(item["document_id"])),
        str(item["chunk_id"]),
        _integer(item["page_number"]) if item.get("page_number") is not None else None,
    )


def _observation_payload(observation: EvidenceObservation) -> dict[str, object]:
    return {
        "identifier": observation.identifier,
        "profile_id": observation.profile_id,
        "fingerprint": _fingerprint_payload(observation.fingerprint),
        "document_id": observation.document_id.value,
        "objective_id": observation.objective_id,
        "session_id": observation.session_id,
        "source": observation.source.value,
        "modality": observation.modality.value,
        "score": observation.score,
        "difficulty": observation.difficulty,
        "citations": [_citation_payload(item) for item in observation.citations],
        "observed_at": observation.observed_at.isoformat(),
    }


def _observation_from_payload(payload: object) -> EvidenceObservation:
    item = _mapping(payload)
    fingerprint = _fingerprint_from_payload(item["fingerprint"])
    return EvidenceObservation(
        str(item["identifier"]),
        str(item["profile_id"]),
        fingerprint,
        DocumentId(str(item["document_id"])),
        str(item["objective_id"]),
        str(item["session_id"]) if item.get("session_id") is not None else None,
        ObservationSource(str(item["source"])),
        ObservationModality(str(item["modality"])),
        _integer(item["score"]),
        _integer(item["difficulty"]),
        tuple(_citation_from_payload(value) for value in _objects(item["citations"])),
        datetime.fromisoformat(str(item["observed_at"])),
    )


def _candidate_payload(candidate: ConceptEquivalenceCandidate) -> dict[str, object]:
    return {
        "profile_id": candidate.profile_id,
        "source": _fingerprint_payload(candidate.source),
        "target": _fingerprint_payload(candidate.target),
        "similarity": candidate.similarity,
        "created_at": candidate.created_at.isoformat(),
    }


def _candidate_from_payload(payload: object) -> ConceptEquivalenceCandidate:
    item = _mapping(payload)
    return ConceptEquivalenceCandidate(
        _fingerprint_from_payload(item["source"]),
        _fingerprint_from_payload(item["target"]),
        _float(item["similarity"]),
        datetime.fromisoformat(str(item["created_at"])),
        str(item.get("profile_id", "")),
    )


def _link_payload(link: ConceptEquivalenceLink) -> dict[str, object]:
    return {
        "profile_id": link.profile_id,
        "source": _fingerprint_payload(link.source),
        "target": _fingerprint_payload(link.target),
        "decision": link.decision.value,
        "decided_at": link.decided_at.isoformat(),
    }


def _link_from_payload(payload: object) -> ConceptEquivalenceLink:
    item = _mapping(payload)
    return ConceptEquivalenceLink(
        _fingerprint_from_payload(item["source"]),
        _fingerprint_from_payload(item["target"]),
        EquivalenceDecision(str(item["decision"])),
        datetime.fromisoformat(str(item["decided_at"])),
        str(item.get("profile_id", "")),
    )


def _pair_key(left: ConceptFingerprint, right: ConceptFingerprint) -> str:
    return "|".join(sorted((left.value, right.value)))


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError("Stored learner profile payload must be an object.")
    return value


def _objects(value: object) -> list[object]:
    if not isinstance(value, list):
        raise RuntimeError("Stored learner profile collection is invalid.")
    return value


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError("Stored learner profile integer is invalid.")
    return value


def _float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("Stored learner profile number is invalid.")
    return float(value)
