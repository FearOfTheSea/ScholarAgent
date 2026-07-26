"""Thin LangGraph runner for explicitly selected local study tools."""

from collections.abc import Mapping
from typing import TypedDict, cast

from scholar_agent.application.output_ports.graph_runner import IGraphRunner
from scholar_agent.application.output_ports.tool_executor import IToolExecutor


class ToolExecutionState(TypedDict, total=False):
    """State passed through the single-node local tool graph."""

    tool_name: str
    arguments: dict[str, object]
    result: dict[str, object]


class LangGraphRunner(IGraphRunner):
    """Runs a one-tool graph with all tool behavior delegated outward."""

    def __init__(self, tool_executor: IToolExecutor) -> None:
        self._tool_executor = tool_executor

    def run(self, state: Mapping[str, object]) -> Mapping[str, object]:
        """Run a selected structured tool through a minimal LangGraph workflow."""
        from langgraph.graph import END, START, StateGraph

        tool_name = state.get("tool_name")
        arguments = state.get("arguments", {})
        if not isinstance(tool_name, str):
            raise ValueError("Graph state requires a string 'tool_name'.")
        if not isinstance(arguments, Mapping):
            raise ValueError("Graph state requires mapping 'arguments'.")

        graph = StateGraph(ToolExecutionState)
        graph.add_node("execute_tool", self._execute_tool)
        graph.add_edge(START, "execute_tool")
        graph.add_edge("execute_tool", END)
        initial_state: ToolExecutionState = {
            "tool_name": tool_name,
            "arguments": dict(arguments),
        }
        result = graph.compile().invoke(initial_state)
        return dict(cast(Mapping[str, object], result))

    def _execute_tool(self, state: ToolExecutionState) -> ToolExecutionState:
        tool_name = state.get("tool_name")
        arguments = state.get("arguments", {})
        if not isinstance(tool_name, str):
            raise ValueError("Graph state requires a string 'tool_name'.")
        if not isinstance(arguments, Mapping):
            raise ValueError("Graph state requires mapping 'arguments'.")
        return {"result": dict(self._tool_executor.execute(tool_name, arguments))}
