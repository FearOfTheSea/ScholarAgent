"""Input port for local runtime readiness."""

from abc import ABC, abstractmethod

from scholar_agent.application.dtos.runtime import RuntimeReadinessResult


class CheckRuntimeReadiness(ABC):
    """Checks whether the configured local model can be used."""

    @abstractmethod
    def execute(self) -> RuntimeReadinessResult:
        """Return local provider and model readiness."""
