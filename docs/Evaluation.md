# Evaluation

The committed corpus in `tests/fixtures/evaluation/` contains owner-supplied
English Machine Learning lectures. Its manifest pins each PDF's SHA-256 hash,
page count, retrieval question, and expected citation page.

The ordinary test suite evaluates the corpus with deterministic lexical vectors
to keep it offline, reproducible, and free of model downloads:

```bash
uv run pytest tests/test_evaluation_corpus.py
```

Run semantic retrieval with the configured local BGE-M3 model after its first
download:

```bash
uv run --group dev python scripts/evaluate_retrieval.py
```

The command writes its detailed citations and pass/fail result to
`data/evaluation/retrieval.json`. A passing case returns at least one retrieved
chunk from the expected source page within `RETRIEVAL_TOP_K` results.

## Mission contract coverage

`tests/test_mission_contracts.py` covers the offline mission boundary: strict
segment-local citations and one repair, omitted versus explicit retrieval
limits, document binding, the exact eight-capability catalog, direct DTO
validation, v1/v2/v3-read/v4-write persistence, planner fallback and prerequisite
capacity, remediation, bounded execution, resume, and mastery completion.
The default suite also checks additive API serialization and the fresh-process
Streamlit session-list path. No test permits an unfiltered agent search or a
material item without a validated citation.

Phase 1 Mission Intelligence is covered by
`tests/test_mission_intelligence.py`. It verifies canonical digest behavior,
idempotent checkpoints, version-2 read/version-4 write migration, tamper
detection, every insight denominator and signal, redacted export fields, the
three additive record endpoints, and the fresh-process UI panel.

Phase 2 Durable Learner Model and Review Memory is covered by
`tests/test_learner_profile_phase2.py`, `tests/test_learner_profile_api.py`,
and `tests/test_review_evaluator.py`. These cover fingerprint stability,
redacted/idempotent observations, schema-v4 association migration, tracing
decay and modality weights, fixed-clock queue ordering and target dates,
consent-gated equivalence, profile round trips, deletion detachment after
reopen, and additive profile/review routes.

Run the complete deterministic mission evaluator with:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_missions.py
```

It runs eight real-graph scenarios and writes the versioned report to
`data/evaluation/mission_phase1_report.json`. The five release gates are
evidence integrity, single-document isolation, resume fidelity, ledger
verification, and action-bound compliance. They are learning signals rather
than claims of learning effectiveness until a learner study exists.

The version-2 report records detailed `evidence_integrity` and
`document_isolation` checks for each scenario. Evidence integrity inspects
learner-facing summaries, every quiz question and flashcard, explanation and
assessment turns, and pending cited questions. Each applicable output must
contain citations tied to the selected document. Document isolation checks
both every recorded tool-call document argument and every citation stored in
the ledger, artifacts, turns, or pending state. Scenarios without a relevant
factual output are marked `applicable: false` and do not pass vacuously; an
aggregate gate requires at least one applicable scenario and all applicable
checks to pass. The ledger gate additionally requires `ledger_verified: true`
for every normal scenario, while tamper detection must pass with
`ledger_verified: false`.

Run the Phase 2 evaluator with:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_reviews.py
```

It writes the version-1 report to
`data/evaluation/review_phase2_report.json` and runs nine deterministic
scenarios. Its eight release gates require at least one applicable passing
check and cover privacy redaction, scheduling, decay, transfer weighting,
round-trip fidelity, deletion completeness, equivalence consent, and
single-document review dispatch.

## Latest local result

The BGE-M3 CPU evaluation completed on 2026-07-26 with all 6 expected
document-page citations retrieved (`6 passed`, `0 failed`).

The opt-in ScholarGPT evaluator now includes the mission-plan, cited
explanation, cited quiz/flashcard, and cited learner-assessment contracts in
addition to the existing planner, document-map, and verification checks.
