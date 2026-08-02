# ADR 0002: Phase 2 learner model and review memory

## Status

Accepted for Phase 2.

## Decision

ScholarAgent stores a local `LearnerProfile` and immutable redacted
`EvidenceObservation` records in a separate SQLite database. A concept
fingerprint is the SHA-256 digest of canonical JSON containing
`nfkc-casefold-punct-v1`, the document id, and normalized objective title and
description. Normalization applies NFKC, casefolding, punctuation-to-space,
and whitespace collapsing. Python's process-randomized `hash()` is never used.

An observation stores only profile and concept identities, score, difficulty,
modality, source, timestamp, session identity, and citation identities. It
never stores learner response, feedback, prompt, reference answer, model
output, or source excerpt. Citation identities contain document, chunk, and
page only.

Knowledge tracing uses score/3, modality weights recall=1.0 and transfer=1.5,
difficulty weights 0.75/1.0/1.25, and recency decay
`0.5 ** (age_days / 30)`. With total weight `W`, weighted success `S`, and
weighted mean absolute deviation `D`, evidence strength is
`1 - exp(-W / 2.5)`, confidence is `round(100 * S * strength)`, and
uncertainty is `round(100 * min(1, 0.7 * (1 - strength) + 0.3 * D))`.
Values are clamped to 0..100. A five-minute clock-skew tolerance clamps small
future timestamps; larger future observations are rejected.

Review intervals are 1/3/7/14/30 days according to confidence and uncertainty,
with a seven-day cap when transfer evidence is absent. Target dates cap the
due time. Queue order is due state, due time, uncertainty descending, then
fingerprint. Review entries retain source-document provenance.

Concept-equivalence proposals come from an Application port with deterministic
descriptor similarity. A proposal or rejection never pools history. Only an
explicit accepted cross-document decision joins evidence groups; same-document
links are invalid.

Mission assessment modality is deterministic: the first check is recall, a
pending challenge after a score of at least 2 is an explicit transfer prompt
using the objective's human title and description, and a retry after an
under-proficient score remains recall. Resynchronization uses the immediately
preceding assessment for the same objective, so remediation retries cannot
inherit transfer weight from an older proficient check.

## Persistence and migration

Study-session JSON serialization remains adapter-owned. Missing and schema
versions 1, 2, and 3 read into the current domain shape with no profile
association. Every current save emits top-level `schema_version=4` and the
optional `learner_profile_id`; ledger entries and digest inputs are unchanged.

Profile metadata, observations, candidates, and decisions use a dedicated
configurable local SQLite database. Import validates the complete redacted
payload before one database transaction replaces an existing profile, and
replacement requires an explicit flag. Delete cascades profile data and
detaches associated sessions while preserving mission history and documents;
the operation reports the detached-session count.

Equivalence candidate and decision keys are composite `(profile_id, pair_key)`
keys. Opening a database created by the earlier Phase 2 adapter migrates the
legacy global-key tables transactionally. Each legacy payload is parsed and
validated; if its payload owner is an existing profile, the row is re-homed to
that owner, otherwise it is dropped. A legacy collision that overwrote the
older payload cannot recover the lost decision, so migration never preserves
the stale column owner or exposes the surviving payload to that profile.

## Consistency and privacy consequences

Mission state is saved first. Profile observation writing is a best-effort
post-save action, so a profile-database failure cannot roll back or corrupt a
valid mission. `SyncMissionObservationsUseCase` reconstructs missing
observations from persisted assessment attempts and deterministic ids before
queue calculation. Repeated sync is idempotent.

Exports contain metadata, concept descriptors, redacted observations, and
equivalence state only. They contain no raw turns, responses, feedback,
prompts, generated text, reference answers, excerpts, PDFs, or telemetry.
