"""Placeholder tool-executor implementation."""

from collections.abc import Mapping

from scholar_agent.application.output_ports.tool_executor import IToolExecutor


class PlaceholderToolExecutor(IToolExecutor):
    """Defines a tool-execution seam without registering tools."""

    def execute(
        self,
        tool_name: str,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Execute a tool when Phase 2 introduces structured tools."""
        raise NotImplementedError("Tool execution is not implemented in Phase 1.")
