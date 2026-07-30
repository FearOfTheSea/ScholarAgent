"""Constrained LangGraph adapter for unified study requests."""

from typing import TypedDict, cast

from scholar_agent.application.dtos.agent import (
    AskStudyAgentRequest,
    AskStudyAgentResult,
    StudyAgentPlanStep,
    StudyAgentStatus,
    StudyAgentTaskError,
    StudyAgentTaskResult,
    StudyTask,
)
from scholar_agent.application.output_ports.agent_runner import IAgentRunner
from scholar_agent.application.output_ports.llm_provider import ILLMProvider
from scholar_agent.application.output_ports.tool_executor import IToolExecutor
from scholar_agent.infrastructure.adapters.study_agent_planning import (
    PlannedAction,
    build_planner_prompt,
    build_repair_prompt,
    parse_study_plan,
)
from scholar_agent.infrastructure.adapters.study_agent_results import task_result


class AgentState(TypedDict, total=False):
    """State accumulated by the unified study graph."""

    prompt: str
    document_id: str
    quiz_count_default: int
    actions: list[PlannedAction]
    action_index: int
    plan: list[StudyAgentPlanStep]
    results: list[StudyAgentTaskResult]
    notices: list[str]
    errors: list[StudyAgentTaskError]
    message: str | None
    planning_error: str | None


class LangGraphAgentRunner(IAgentRunner):
    """Plan and execute only registered study capabilities."""

    def __init__(
        self,
        tool_executor: IToolExecutor,
        llm_provider: ILLMProvider,
    ) -> None:
        self._tool_executor = tool_executor
        self._llm_provider = llm_provider

    def run(self, request: AskStudyAgentRequest) -> AskStudyAgentResult:
        """Build, validate, and execute one constrained study graph."""
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(AgentState)
        graph.add_node("plan", self._plan)
        graph.add_node("execute_action", self._execute_action)
        graph.add_node("finalize", self._finalize)
        graph.add_edge(START, "plan")
        graph.add_conditional_edges(
            "plan",
            self._after_plan,
            {"execute_action": "execute_action", "finalize": "finalize"},
        )
        graph.add_conditional_edges(
            "execute_action",
            self._after_execution,
            {"execute_action": "execute_action", "finalize": "finalize"},
        )
        graph.add_edge("finalize", END)
        result = graph.compile().invoke(_initial_state(request))
        return _agent_result(cast(AgentState, result))

    def _plan(self, state: AgentState) -> AgentState:
        definitions = self._tool_executor.capabilities()
        if not definitions:
            return {
                "planning_error": "No study capabilities are registered.",
                "message": "The study agent is not configured.",
            }

        prompt = build_planner_prompt(
            state["prompt"],
            definitions,
            state["quiz_count_default"],
        )
        raw_plan = self._llm_provider.generate(prompt)
        try:
            actions, message = parse_study_plan(
                raw_plan,
                definitions,
                state["quiz_count_default"],
            )
        except ValueError as first_error:
            repaired_output = self._llm_provider.generate(
                build_repair_prompt(prompt, raw_plan, str(first_error))
            )
            try:
                actions, message = parse_study_plan(
                    repaired_output,
                    definitions,
                    state["quiz_count_default"],
                )
            except ValueError as second_error:
                return _invalid_plan_state(second_error)

        definitions_by_name = {
            definition.task.value: definition for definition in definitions
        }
        plan = [
            StudyAgentPlanStep(
                task=StudyTask(action["tool_name"]),
                description=definitions_by_name[action["tool_name"]].description,
            )
            for action in actions
        ]
        return {
            "actions": actions,
            "plan": plan,
            "action_index": 0,
            "message": message,
        }

    @staticmethod
    def _after_plan(state: AgentState) -> str:
        if state.get("actions"):
            return "execute_action"
        return "finalize"

    def _execute_action(self, state: AgentState) -> AgentState:
        actions = state.get("actions", [])
        index = state.get("action_index", 0)
        if index >= len(actions):
            return {}

        action = actions[index]
        task = StudyTask(action["tool_name"])
        arguments = dict(action["arguments"])
        arguments["document_id"] = state["document_id"]
        results = list(state.get("results", []))
        notices = list(state.get("notices", []))
        errors = list(state.get("errors", []))
        try:
            payload = self._tool_executor.execute(task.value, arguments)
            results.append(task_result(task, payload))
            notice = payload.get("notice")
            if isinstance(notice, str) and notice and notice not in notices:
                notices.append(notice)
        except (RuntimeError, ValueError) as error:
            errors.append(StudyAgentTaskError(task=task, message=str(error)))
        return {
            "action_index": index + 1,
            "results": results,
            "notices": notices,
            "errors": errors,
        }

    @staticmethod
    def _after_execution(state: AgentState) -> str:
        if state.get("action_index", 0) < len(state.get("actions", [])):
            return "execute_action"
        return "finalize"

    @staticmethod
    def _finalize(state: AgentState) -> AgentState:
        return {}


def _initial_state(request: AskStudyAgentRequest) -> AgentState:
    return {
        "prompt": request.prompt,
        "document_id": request.document_id.value,
        "quiz_count_default": request.quiz_count_default,
        "actions": [],
        "action_index": 0,
        "plan": [],
        "results": [],
        "notices": [],
        "errors": [],
        "message": None,
        "planning_error": None,
    }


def _invalid_plan_state(error: ValueError) -> AgentState:
    return {
        "planning_error": (
            "The local model returned an invalid study plan after one repair "
            f"attempt: {error}"
        ),
        "message": "The study agent could not safely choose a supported task.",
    }


def _agent_result(state: AgentState) -> AskStudyAgentResult:
    results = tuple(state.get("results", []))
    errors = tuple(state.get("errors", []))
    planning_error = state.get("planning_error")
    message: str | None
    if planning_error is not None:
        status = StudyAgentStatus.FAILED
        message = planning_error
    elif results and errors:
        status = StudyAgentStatus.PARTIAL
        message = state.get("message")
    elif errors:
        status = StudyAgentStatus.FAILED
        message = state.get("message")
    elif results:
        status = StudyAgentStatus.COMPLETED
        message = state.get("message")
    else:
        status = StudyAgentStatus.NEEDS_CLARIFICATION
        message = state.get("message")
    return AskStudyAgentResult(
        status=status,
        plan=tuple(state.get("plan", [])),
        results=results,
        notices=tuple(state.get("notices", [])),
        errors=errors,
        message=message,
    )
