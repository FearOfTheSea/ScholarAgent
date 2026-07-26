"""Tool-executor port."""

from abc import ABC, abstractmethod
from collections.abc import Mapping


class IToolExecutor(ABC):
    """Executes a named application tool."""

    @abstractmethod
    def execute(
        self,
        tool_name: str,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Execute a tool with structured arguments."""
