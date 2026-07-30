# Use cases

Each use case models one user action and receives its collaborators through the
constructor. `AskStudyAgentUseCase` coordinates a free-form request through the
agent-runner port; structured tools immediately delegate selected actions back
to the single-responsibility direct use cases.
