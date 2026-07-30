"""Thin LangGraph adapter for bounded adaptive tutoring turns."""

from typing import TypedDict, cast

from scholar_agent.application.dtos.tutor import (
    ContinueStudySessionRequest,
    TutorTurnResult,
)
from scholar_agent.application.output_ports.tutor_runner import ITutorRunner
from scholar_agent.application.services.tutor_turn_service import (
    PreparedTutorTurn,
    TutorTurnService,
)


class TutorState(TypedDict, total=False):
    """State passed through the explicit tutor workflow."""

    request: ContinueStudySessionRequest
    intent: str
    prepared: PreparedTutorTurn
    result: TutorTurnResult


class LangGraphTutorRunner(ITutorRunner):
    """Sequence application-owned tutoring behavior through thin nodes."""

    def __init__(self, turn_service: TutorTurnService) -> None:
        self._turn_service = turn_service

    def run(self, request: ContinueStudySessionRequest) -> TutorTurnResult:
        """Run exactly one classify, prepare, and persist cycle."""
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(TutorState)
        graph.add_node("classify", self._classify)
        graph.add_node("prepare_and_verify", self._prepare)
        graph.add_node("persist", self._persist)
        graph.add_edge(START, "classify")
        graph.add_edge("classify", "prepare_and_verify")
        graph.add_edge("prepare_and_verify", "persist")
        graph.add_edge("persist", END)
        state = cast(TutorState, graph.compile().invoke({"request": request}))
        return state["result"]

    def _classify(self, state: TutorState) -> TutorState:
        return {"intent": self._turn_service.classify(state["request"])}

    def _prepare(self, state: TutorState) -> TutorState:
        return {
            "prepared": self._turn_service.prepare(
                state["request"],
                state["intent"],
            )
        }

    def _persist(self, state: TutorState) -> TutorState:
        return {"result": self._turn_service.persist(state["prepared"])}
