"""Internal contracts for structured local tools."""

from collections.abc import Mapping
from typing import Protocol


class StructuredTool(Protocol):
    """Runs one structured local capability."""

    def execute(self, arguments: Mapping[str, object]) -> Mapping[str, object]:
        """Run the tool using structured arguments."""
