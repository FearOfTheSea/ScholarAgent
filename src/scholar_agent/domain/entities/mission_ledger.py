"""Replay-safe, document-bound records for Study Mission decisions."""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from scholar_agent.domain.value_objects.source_reference import SourceReference


class MissionLedgerEventType(StrEnum):
    """Allowed immutable mission transition records."""

    MISSION_STARTED = "mission_started"
    PLAN_CREATED = "plan_created"
    CAPABILITY_COMPLETED = "capability_completed"
    CAPABILITY_FAILED = "capability_failed"
    LEARNER_ASSESSED = "learner_assessed"
    REMEDIATION_STARTED = "remediation_started"
    MASTERY_CHANGED = "mastery_changed"
    ARTIFACT_CREATED = "artifact_created"
    WAITING_FOR_LEARNER = "waiting_for_learner"
    MISSION_COMPLETED = "mission_completed"
    MISSION_FAILED = "mission_failed"


MISSION_LEDGER_EVENT_TYPES = frozenset(item.value for item in MissionLedgerEventType)


@dataclass(frozen=True, slots=True)
class MissionStateProjection:
    """The state needed to replay a transition without private content."""

    status: str
    active_milestone_id: str | None
    pending_objective_id: str | None
    action_count: int
    attempt_count: int
    artifact_count: int
    completed_milestone_count: int
    mastery_by_objective: tuple[tuple[str, str], ...]
    next_milestone_id: str | None


@dataclass(frozen=True, slots=True)
class MissionLedgerEntry:
    """One append-only, redacted mission transition."""

    sequence: int
    event_type: MissionLedgerEventType | str
    summary: str
    objective_id: str | None
    capability: str | None
    citations: tuple[SourceReference, ...]
    projection: MissionStateProjection
    previous_digest: str
    current_digest: str
    created_at: datetime = datetime.min.replace(tzinfo=UTC)
    transition_key: str | None = None

    def __post_init__(self) -> None:
        event_type = _event_value(self.event_type)
        if event_type not in MISSION_LEDGER_EVENT_TYPES:
            raise ValueError(f"Unsupported mission ledger event '{event_type}'.")
        if self.sequence < 1:
            raise ValueError("Mission ledger sequences start at one.")
        if not _is_digest(self.previous_digest, allow_empty=True):
            raise ValueError("Mission ledger previous digest is invalid.")
        if not _is_digest(self.current_digest):
            raise ValueError("Mission ledger current digest is invalid.")


@dataclass(frozen=True, slots=True)
class LedgerVerificationResult:
    """The first failure, or a successful complete-chain verification."""

    valid: bool
    sequence: int | None = None
    reason: str | None = None


def canonical_projection(projection: MissionStateProjection) -> dict[str, object]:
    """Return stable JSON-compatible projection data."""
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


def canonical_ledger_payload(
    *,
    previous_digest: str,
    sequence: int,
    event_type: MissionLedgerEventType | str,
    projection: MissionStateProjection,
    objective_id: str | None,
    capability: str | None,
    citations: tuple[SourceReference, ...],
) -> dict[str, object]:
    """Build the private-content-free payload used by the digest."""
    return {
        "previous_digest": previous_digest,
        "sequence": sequence,
        "event_type": _event_value(event_type),
        "projection": canonical_projection(projection),
        "objective_id": objective_id,
        "capability": capability,
        "citation_identities": [
            {
                "document_id": reference.document_id.value,
                "chunk_id": reference.chunk_id,
                "page_number": reference.page_number,
            }
            for reference in citations
        ],
    }


def canonical_ledger_json(
    *,
    previous_digest: str,
    sequence: int,
    event_type: MissionLedgerEventType | str,
    projection: MissionStateProjection,
    objective_id: str | None,
    capability: str | None,
    citations: tuple[SourceReference, ...],
) -> str:
    """Serialize digest input with deterministic JSON rules."""
    return json.dumps(
        canonical_ledger_payload(
            previous_digest=previous_digest,
            sequence=sequence,
            event_type=event_type,
            projection=projection,
            objective_id=objective_id,
            capability=capability,
            citations=citations,
        ),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def compute_ledger_digest(
    *,
    previous_digest: str,
    sequence: int,
    event_type: MissionLedgerEventType | str,
    projection: MissionStateProjection,
    objective_id: str | None,
    capability: str | None,
    citations: tuple[SourceReference, ...],
) -> str:
    """Compute a SHA-256 digest without human text or timestamps."""
    return hashlib.sha256(
        canonical_ledger_json(
            previous_digest=previous_digest,
            sequence=sequence,
            event_type=event_type,
            projection=projection,
            objective_id=objective_id,
            capability=capability,
            citations=citations,
        ).encode("utf-8")
    ).hexdigest()


def verify_ledger(entries: tuple[MissionLedgerEntry, ...]) -> LedgerVerificationResult:
    """Validate sequence, event shape, and every digest link."""
    previous_digest = ""
    for expected_sequence, entry in enumerate(entries, start=1):
        try:
            if entry.sequence != expected_sequence:
                return LedgerVerificationResult(
                    False,
                    entry.sequence,
                    f"Expected sequence {expected_sequence}, found {entry.sequence}.",
                )
            if entry.previous_digest != previous_digest:
                return LedgerVerificationResult(
                    False,
                    entry.sequence,
                    "Previous digest does not match the preceding ledger entry.",
                )
            expected_digest = compute_ledger_digest(
                previous_digest=entry.previous_digest,
                sequence=entry.sequence,
                event_type=entry.event_type,
                projection=entry.projection,
                objective_id=entry.objective_id,
                capability=entry.capability,
                citations=entry.citations,
            )
            if entry.current_digest != expected_digest:
                return LedgerVerificationResult(
                    False,
                    entry.sequence,
                    "Ledger digest does not match its canonical transition data.",
                )
        except (TypeError, ValueError) as error:
            return LedgerVerificationResult(False, expected_sequence, str(error))
        previous_digest = entry.current_digest
    return LedgerVerificationResult(True)


def _event_value(event_type: MissionLedgerEventType | str) -> str:
    return (
        event_type.value
        if isinstance(event_type, MissionLedgerEventType)
        else event_type
    )


def _is_digest(value: str, *, allow_empty: bool = False) -> bool:
    if allow_empty and value == "":
        return True
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
