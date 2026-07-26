"""Local runtime readiness use case."""

from scholar_agent.application.dtos.runtime import RuntimeReadinessResult
from scholar_agent.application.input_ports.runtime import CheckRuntimeReadiness
from scholar_agent.application.output_ports.llm_provider import ILLMProvider


class CheckRuntimeReadinessUseCase(CheckRuntimeReadiness):
    """Reports availability of the configured local language model."""

    def __init__(self, llm_provider: ILLMProvider) -> None:
        self._llm_provider = llm_provider

    def execute(self) -> RuntimeReadinessResult:
        """Check the Ollama process and configured model without inference."""
        ollama_available = self._llm_provider.is_available()
        return RuntimeReadinessResult(
            ollama_available=ollama_available,
            model_available=ollama_available and self._llm_provider.has_model(),
        )
