# Agentic Study Missions

ScholarAgent now treats a study mission as a bounded, persistent workflow over
one selected PDF. The existing `StudySession` aggregate holds the plan,
milestones, cited artifacts, pending learner interaction, concise trace, and
completion state. Direct question, summary, quiz, and flashcard endpoints remain
available, while `/agent/requests` remains the stateless Quick Ask contract.

## Implemented contract

- Every summary, quiz question, and flashcard has validated `SourceReference`
  citations. Quiz and flashcard aggregates expose the deduplicated union, while
  each item retains its own citations.
- Summary segmentation preserves labeled chunk identity. Partial responses may
  cite only their segment; a combined response may cite only the deduplicated
  union of partial citations. All structured capabilities allow one repair.
- The mission tool catalog is exactly: `semantic_search`, `summarize_document`,
  `generate_quiz`, `generate_flashcards`, `citation_lookup`,
  `build_document_map`, `explain_concept`, and `assess_learner_response`.
  `answer_question` remains direct-only.
- The application injects the singular document identifier. Semantic search
  rejects missing document scope, and capability adapters reject wrong-document
  or empty evidence. Pending reference answers stay internal.
- Planning validates `{focus, objective_ids}`, applies minute/objective
  capacity, expands prerequisite closure in brief order, repairs once, and
  falls back deterministically. Guided, exam, and cram modes have explicit
  milestone sequences.
- Automatic work stops after four capability executions per advance and 64 per
  session. Optional artifact failures are isolated; core failures are persisted
  and resumable. Trace entries contain state and capability summaries only.
- SQLite reads missing/version-1 and version-2 payloads into the current domain
  shape, initializes an empty verified ledger, and writes top-level
  `schema_version=3`; serialization version is not business state on
  `StudySession`.

## Mission Intelligence

Phase 1 adds a bounded append-only ledger to the aggregate. The Application
mission state service writes chained, redacted entries for starts, plans,
capabilities, assessments, remediation, artifacts, waits, failures, and
completion. Current-schema loads verify sequence and digest integrity; repeated
transition keys are idempotent. Insights are pure deterministic learning
signals, and the three record endpoints expose verification and explicit,
redacted export without learner responses, prompts, model output, reference
answers, or source excerpts.

## Delivery surface

The API adds mission listing with document/status filters, advance, and explicit
completion routes. Existing session routes remain additive. The Streamlit
navigation is **Study Mission** and **Quick Ask**; missions can be started,
listed, resumed, continued, finished, or deleted, with plan/mastery, chat,
artifacts/evidence, and a concise trace visible together.

## Verification record

The deterministic mission tests cover citation repair, retrieval defaults,
single-document binding, strict capability inputs, persistence migration,
planner policies, remediation, bounded execution, resume, and mastery. The
ScholarGPT training/evaluation contracts include mission plans, cited
explanations, cited artifacts, and cited assessments. Real Ollama and
ScholarGPT journeys remain opt-in because they require locally cached models.
