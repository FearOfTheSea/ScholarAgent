"""Port for goal-oriented agent workflow adapters."""

from abc import ABC, abstractmethod
from collections.abc import Mapping


class IAgentRunner(ABC):
    """Runs a multi-step study-agent workflow."""

    @abstractmethod
    def run(self, state: Mapping[str, object]) -> Mapping[str, object]:
        """Run the agent and return its accumulated state."""
