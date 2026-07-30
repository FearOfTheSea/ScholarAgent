# FastAPI adapter

The FastAPI adapter exposes health, document lifecycle, direct study, and
unified-agent endpoints. It validates HTTP shapes, delegates to application use
cases, and serializes typed results without business logic.

The adaptive tutor API starts, resumes, advances, and deletes sessions under
`/agent/sessions`. Every session is permanently bound to one document, and
responses expose typed activities, mastery, assessments, and source evidence.
