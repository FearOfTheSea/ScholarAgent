"""Explicit capability catalog for the adaptive tutor workflow."""

from scholar_agent.application.dtos.tutor import TutorCapability

TUTOR_CAPABILITIES: tuple[tuple[TutorCapability, str], ...] = (
    (
        TutorCapability.BUILD_DOCUMENT_MAP,
        "Build a cited concept map and learning objectives for one selected PDF.",
    ),
    (
        TutorCapability.EXPLAIN_CONCEPT,
        "Teach one concept with selected-document evidence and a comprehension check.",
    ),
    (
        TutorCapability.ASSESS_RESPONSE,
        "Score a learner response, cite feedback, and update deterministic mastery.",
    ),
)
