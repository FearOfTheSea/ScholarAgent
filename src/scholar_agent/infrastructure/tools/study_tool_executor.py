"""Structured executor for the permitted study tools."""

from collections.abc import Mapping

from scholar_agent.application.output_ports.tool_executor import (
    IToolExecutor,
    StudyToolDefinition,
)
from scholar_agent.infrastructure.tools.contracts import StructuredTool


class StudyToolExecutor(IToolExecutor):
    """Routes an approved tool name to its single-responsibility implementation."""

    def __init__(
        self,
        tools: Mapping[str, StructuredTool],
        capabilities: tuple[StudyToolDefinition, ...] = (),
    ) -> None:
        self._tools = dict(tools)
        self._capabilities = capabilities
        missing_tools = {
            definition.task.value
            for definition in capabilities
            if definition.task.value not in self._tools
        }
        if missing_tools:
            names = ", ".join(sorted(missing_tools))
            raise ValueError(f"Capabilities require unregistered tools: {names}.")

    def capabilities(self) -> tuple[StudyToolDefinition, ...]:
        """Return the explicit capabilities available to the agent."""
        return self._capabilities

    def execute(
        self,
        tool_name: str,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Execute one approved structured study tool."""
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ValueError(f"Unsupported tool: {tool_name}.")
        return tool.execute(arguments)
