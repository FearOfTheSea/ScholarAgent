"""Language-model provider port."""

from abc import ABC, abstractmethod


class ILLMProvider(ABC):
    """Generates text from a supplied prompt."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text for a prompt."""

    @abstractmethod
    def is_available(self) -> bool:
        """Report whether the local provider is reachable."""

    @abstractmethod
    def has_model(self) -> bool:
        """Report whether the configured model is installed locally."""
