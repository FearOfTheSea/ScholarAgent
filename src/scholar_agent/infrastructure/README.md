# Infrastructure

Infrastructure contains replaceable implementations of application output ports
and the dependency-injection composition root. It depends inward on application
and domain contracts.

Adaptive sessions use a SQLite repository for resumable state and cached cited
document briefs. `LangGraphTutorRunner` is a thin three-node adapter that
delegates tutoring and mastery behavior to the application layer.
