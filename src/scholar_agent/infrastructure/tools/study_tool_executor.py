"""Structured executor for the permitted study tools."""

from collections.abc import Mapping

from scholar_agent.application.output_ports.tool_executor import IToolExecutor
from scholar_agent.infrastructure.tools.contracts import StructuredTool


class StudyToolExecutor(IToolExecutor):
    """Routes an approved tool name to its single-responsibility implementation."""

    def __init__(self, tools: Mapping[str, StructuredTool]) -> None:
        self._tools = dict(tools)

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
