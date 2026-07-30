"""Port and metadata contracts for structured study tools."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from scholar_agent.application.dtos.agent import StudyTask


class ToolArgumentKind(StrEnum):
    """Supported planner argument types."""

    TEXT = "text"
    POSITIVE_INTEGER = "positive_integer"


@dataclass(frozen=True, slots=True)
class ToolArgumentDefinition:
    """One model-selectable argument accepted by a study capability."""

    name: str
    kind: ToolArgumentKind
    required: bool
    default: str | int | None = None


@dataclass(frozen=True, slots=True)
class StudyToolDefinition:
    """The explicit model-facing contract for one study capability."""

    task: StudyTask
    description: str
    arguments: tuple[ToolArgumentDefinition, ...] = ()


class IToolExecutor(ABC):
    """Executes a named application tool."""

    @abstractmethod
    def execute(
        self,
        tool_name: str,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Execute a tool with structured arguments."""

    def capabilities(self) -> tuple[StudyToolDefinition, ...]:
        """Return user-facing capabilities available to an agent planner."""
        return ()
