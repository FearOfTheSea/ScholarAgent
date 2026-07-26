"""Workflow-runner port."""

from abc import ABC, abstractmethod
from collections.abc import Mapping


class IGraphRunner(ABC):
    """Runs a workflow using a state mapping."""

    @abstractmethod
    def run(self, state: Mapping[str, object]) -> Mapping[str, object]:
        """Run a workflow and return its resulting state."""
