# ADR 0001: Phase 1 mission ledger and schema migration

## Status

Accepted for Phase 1.

## Decision

Persist an immutable `MissionLedgerEntry` tuple inside `StudySession`. Each
entry has a contiguous sequence, a deterministic redacted summary, validated
single-document citation references, a replay-safe state projection, and a
SHA-256 digest chained to the previous entry. Digest input excludes human text,
timestamps, learner responses, prompts, model output, reference answers, and
source excerpts. A repeated deterministic transition key is idempotent.

The ledger is bounded at 512 entries. An attempted append at capacity moves the
mission into a recoverable failed state without truncating history. The
Application `MissionStateService` is the single transition writer. Graph nodes
delegate to it through focused Application services; presentation only reads
insights and redacted records.

SQLite serialization is adapter-owned: missing and schema versions 1 and 2
read into the current domain shape with an empty ledger, while every subsequent
save emitted top-level `schema_version=3`. Phase 2 extends this adapter
contract to read version 3 and emit version 4 with an optional profile
association; current-schema loads still validate the
sequence and digest chain before returning a session.

## Consequences

Mission insights are pure deterministic calculations over state and the
verified ledger. Export is explicit and redacted: it contains session identity,
plan identifiers, ledger projections and summaries, citation identities,
artifact metadata, and insights, but no raw learner or model/source content.
The ledger is local and has no cloud telemetry path.
