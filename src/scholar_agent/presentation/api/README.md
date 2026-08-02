# FastAPI adapter

The FastAPI adapter exposes health, document lifecycle, direct study, and
unified-agent endpoints. It validates HTTP shapes, delegates to application use
cases, and serializes typed results without business logic.

The adaptive tutor API starts, resumes, advances, and deletes sessions under
`/agent/sessions`. Every session is permanently bound to one document, and
responses expose typed activities, mastery, assessments, and source evidence.

Local learner profiles and review memory are additive under
`/learner-profiles`. They provide profile CRUD, redacted export/import with
explicit replacement, a fixed-clock review queue, review outcomes,
equivalence decisions, and document-bound review-mission dispatch.
