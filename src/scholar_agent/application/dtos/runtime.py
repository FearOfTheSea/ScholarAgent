"""DTOs for local runtime readiness."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeReadinessResult:
    """Availability of the configured local language model."""

    ollama_available: bool
    model_available: bool
