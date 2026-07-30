# Application

The application layer defines user actions, their data contracts, validation,
and infrastructure-independent behavior. It depends only on the domain layer.

Adaptive tutoring is represented by explicit start, continue, retrieve, delete,
and document-brief use cases. `TutorTurnService` owns single-document routing,
grounding requirements, assessment validation, prerequisite selection, and
deterministic mastery calculations; LangGraph only sequences those operations.
