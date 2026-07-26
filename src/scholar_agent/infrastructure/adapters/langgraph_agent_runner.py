"""Conditional LangGraph adapter for the goal-oriented study agent."""

from collections.abc import Mapping
from typing import TypedDict, cast

from scholar_agent.application.output_ports.agent_runner import IAgentRunner
from scholar_agent.application.output_ports.tool_executor import IToolExecutor


class AgentAction(TypedDict):
    """One approved tool invocation in the agent plan."""

    tool_name: str
    description: str
    arguments: dict[str, object]


class AgentState(TypedDict, total=False):
    """State accumulated by the multi-step study graph."""

    goal: str
    document_ids: list[str]
    question_count: int
    session_id: str | None
    plan: list[dict[str, str]]
    actions: list[AgentAction]
    action_index: int
    completed_tools: list[str]
    citations: list[dict[str, object]]
    summaries: list[str]
    quiz: list[dict[str, str]]
    recommendations: list[str]
    errors: list[str]
    summary: str


class LangGraphAgentRunner(IAgentRunner):
    """Runs a constrained, multi-step study workflow with LangGraph."""

    _allowed_tools = {
        "semantic_search",
        "summarize_document",
        "compare_documents",
        "generate_quiz",
        "generate_flashcards",
        "citation_lookup",
    }

    def __init__(self, tool_executor: IToolExecutor) -> None:
        self._tool_executor = tool_executor

    def run(self, state: Mapping[str, object]) -> Mapping[str, object]:
        """Build and execute the conditional study graph."""
        from langgraph.graph import END, START, StateGraph

        initial = _initial_state(state)
        graph = StateGraph(AgentState)
        graph.add_node("plan", self._plan)
        graph.add_node("execute_action", self._execute_action)
        graph.add_node("finalize", self._finalize)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "execute_action")
        graph.add_conditional_edges(
            "execute_action",
            self._next_step,
            {"execute_action": "execute_action", "finalize": "finalize"},
        )
        graph.add_edge("finalize", END)
        result = graph.compile().invoke(initial)
        return dict(cast(Mapping[str, object], result))

    def _plan(self, state: AgentState) -> AgentState:
        goal = state["goal"]
        document_ids = state["document_ids"]
        actions: list[AgentAction] = [
            {
                "tool_name": "semantic_search",
                "description": "Find the most relevant evidence for the study goal.",
                "arguments": {
                    "query": goal,
                    "document_ids": document_ids,
                    "limit": 6,
                },
            },
        ]
        for document_id in document_ids:
            actions.append(
                {
                    "tool_name": "summarize_document",
                    "description": (
                        f"Build a focused summary of document {document_id}."
                    ),
                    "arguments": {"document_id": document_id},
                },
            )
        if len(document_ids) >= 2:
            actions.append(
                {
                    "tool_name": "compare_documents",
                    "description": (
                        "Compare the selected documents to identify connections "
                        "and differences."
                    ),
                    "arguments": {
                        "first_document_id": document_ids[0],
                        "second_document_id": document_ids[1],
                    },
                },
            )
        if _requests_flashcards(goal):
            actions.append(
                {
                    "tool_name": "generate_flashcards",
                    "description": "Create flashcards for active recall.",
                    "arguments": {"document_id": document_ids[0], "card_count": 8},
                },
            )
        actions.append(
            {
                "tool_name": "generate_quiz",
                "description": (
                    "Generate a quiz to check understanding of the material."
                ),
                "arguments": {
                    "document_id": document_ids[0],
                    "question_count": state["question_count"],
                },
            },
        )
        if any(action["tool_name"] not in self._allowed_tools for action in actions):
            raise ValueError("Planner produced an unsupported tool.")
        return {
            "actions": actions,
            "plan": [
                {
                    "tool_name": action["tool_name"],
                    "description": action["description"],
                }
                for action in actions
            ],
            "action_index": 0,
        }

    def _execute_action(self, state: AgentState) -> AgentState:
        actions = state.get("actions", [])
        index = state.get("action_index", 0)
        if index >= len(actions):
            return {}
        action = actions[index]
        completed = list(state.get("completed_tools", []))
        citations = list(state.get("citations", []))
        summaries = list(state.get("summaries", []))
        quiz = list(state.get("quiz", []))
        errors = list(state.get("errors", []))
        try:
            result = self._tool_executor.execute(
                action["tool_name"], action["arguments"]
            )
            completed.append(action["tool_name"])
            _collect_result(result, citations, summaries, quiz)
        except (RuntimeError, ValueError) as error:
            errors.append(f"{action['tool_name']}: {error}")
        return {
            "action_index": index + 1,
            "completed_tools": completed,
            "citations": citations,
            "summaries": summaries,
            "quiz": quiz,
            "errors": errors,
        }

    def _next_step(self, state: AgentState) -> str:
        if state.get("action_index", 0) < len(state.get("actions", [])):
            return "execute_action"
        return "finalize"

    def _finalize(self, state: AgentState) -> AgentState:
        summaries = state.get("summaries", [])
        summary = "\n\n".join(summaries)
        recommendations = [
            (
                "Review the summary, then answer the quiz questions without "
                "looking at the answers."
            ),
            "Use the cited pages to revisit concepts you cannot explain confidently.",
        ]
        if len(state.get("document_ids", [])) >= 2:
            recommendations.append(
                "Pay special attention to the similarities and differences "
                "between the documents."
            )
        if state.get("errors"):
            recommendations.append(
                "Some optional study steps were unavailable; run them again "
                "after checking local model readiness."
            )
        return {"summary": summary, "recommendations": recommendations}


def _initial_state(state: Mapping[str, object]) -> AgentState:
    goal = state.get("goal")
    document_ids = state.get("document_ids")
    question_count = state.get("question_count", 5)
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("Graph state requires a non-blank 'goal'.")
    if (
        not isinstance(document_ids, list)
        or not document_ids
        or not all(isinstance(item, str) and item for item in document_ids)
    ):
        raise ValueError("Graph state requires a non-empty list of document IDs.")
    if not isinstance(question_count, int) or question_count < 1:
        raise ValueError("Graph state requires a positive 'question_count'.")
    session_id = state.get("session_id")
    if session_id is not None and not isinstance(session_id, str):
        raise ValueError("Graph state 'session_id' must be text or null.")
    return {
        "goal": goal.strip(),
        "document_ids": document_ids,
        "question_count": question_count,
        "session_id": session_id,
        "completed_tools": [],
        "citations": [],
        "summaries": [],
        "quiz": [],
        "errors": [],
    }


def _collect_result(
    result: Mapping[str, object],
    citations: list[dict[str, object]],
    summaries: list[str],
    quiz: list[dict[str, str]],
) -> None:
    raw_chunks = result.get("chunks", [])
    if isinstance(raw_chunks, list):
        for chunk in raw_chunks:
            if isinstance(chunk, Mapping):
                candidate = dict(chunk)
                candidate.setdefault("section", None)
                if candidate not in citations:
                    citations.append(candidate)
    raw_summary = result.get("summary")
    if isinstance(raw_summary, str) and raw_summary:
        summaries.append(raw_summary)
    raw_questions = result.get("questions", [])
    if isinstance(raw_questions, list):
        for question in raw_questions:
            if isinstance(question, Mapping):
                prompt = question.get("prompt")
                answer = question.get("answer")
                if isinstance(prompt, str) and isinstance(answer, str):
                    quiz.append({"prompt": prompt, "answer": answer})


def _requests_flashcards(goal: str) -> bool:
    words = {"flashcard", "flashcards", "memorize", "memorisation", "memory"}
    return any(word in goal.lower() for word in words)
