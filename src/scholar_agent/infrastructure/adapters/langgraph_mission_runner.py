"""LangGraph adapter for the bounded persistent Study Mission workflow."""

from typing import TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from scholar_agent.application.dtos.mission import AdvanceStudyMissionRequest
from scholar_agent.application.dtos.tutor import StudySessionResult, TutorActivity
from scholar_agent.application.output_ports.mission_runner import IMissionRunner
from scholar_agent.application.output_ports.tool_executor import IToolExecutor
from scholar_agent.application.services.mission_actions import (
    MissionActionService,
)
from scholar_agent.application.services.mission_assessment import (
    MissionAssessmentService,
)
from scholar_agent.application.services.mission_capabilities import (
    MissionCapabilityService,
)
from scholar_agent.application.services.mission_interactions import (
    MissionInteractionService,
)
from scholar_agent.application.services.mission_materials import MissionMaterialService
from scholar_agent.application.services.mission_policy import (
    MissionPolicy,
    MissionRoute,
    MissionSelection,
)
from scholar_agent.application.services.mission_results import MissionResultService
from scholar_agent.application.services.mission_state import MissionStateService
from scholar_agent.application.services.mission_steps import MissionStep
from scholar_agent.application.services.mission_verification import (
    MissionVerificationService,
)
from scholar_agent.domain.entities.study_session import StudySession
from scholar_agent.domain.repositories.study_session_repository import (
    StudySessionRepository,
)


class MissionGraphState(TypedDict, total=False):
    """State carried through explicit mission graph nodes."""

    request: AdvanceStudyMissionRequest
    session: StudySession
    trace_start: int
    message: str | None
    route: str
    selection: MissionSelection
    before_action: StudySession
    step: MissionStep
    activity: TutorActivity | None
    turn_executions: int
    error: str
    result: StudySessionResult


class LangGraphMissionRunner(IMissionRunner):
    """Compile and invoke a bounded graph whose nodes delegate inward."""

    def __init__(
        self,
        tool_executor: IToolExecutor,
        session_repository: StudySessionRepository,
        maximum_actions_per_turn: int = 4,
        maximum_actions_per_session: int = 64,
        policy: MissionPolicy | None = None,
        state_service: MissionStateService | None = None,
        action_service: MissionActionService | None = None,
        result_service: MissionResultService | None = None,
        verification_service: MissionVerificationService | None = None,
    ) -> None:
        self._maximum_actions_per_turn = maximum_actions_per_turn
        self._maximum_actions_per_session = maximum_actions_per_session
        self._policy = policy or MissionPolicy()
        self._state = state_service or MissionStateService(session_repository)
        if action_service is None:
            capabilities = MissionCapabilityService(
                tool_executor, self._state, maximum_actions_per_session
            )
            interactions = MissionInteractionService(
                capabilities, self._state, self._policy
            )
            materials = MissionMaterialService(capabilities, self._state, self._policy)
            assessment = MissionAssessmentService(
                capabilities, self._state, self._policy
            )
            action_service = MissionActionService(
                self._state,
                self._policy,
                interactions,
                materials,
                assessment,
            )
        self._actions = action_service
        self._results = result_service or MissionResultService(self._policy)
        self._verification = verification_service or MissionVerificationService()

    def run(self, request: AdvanceStudyMissionRequest) -> StudySessionResult:
        """Invoke one graph run bounded by the configured per-advance budget."""
        initial: MissionGraphState = {
            "request": request,
            "turn_executions": 0,
        }
        graph = self.build_graph()
        result = cast(MissionGraphState, graph.invoke(initial))
        return result["result"]

    def build_graph(
        self,
    ) -> CompiledStateGraph[
        MissionGraphState, None, MissionGraphState, MissionGraphState
    ]:
        """Build the inspectable load-to-finalize StateGraph."""
        graph = StateGraph(MissionGraphState)
        graph.add_node("load", self._load)
        graph.add_node("classify", self._classify)
        graph.add_node("select", self._select)
        graph.add_node("execute", self._execute)
        graph.add_node("verify", self._verify)
        graph.add_node("reflect", self._reflect)
        graph.add_node("checkpoint", self._checkpoint)
        graph.add_node("finalize", self._finalize)
        graph.add_edge(START, "load")
        graph.add_edge("load", "classify")
        graph.add_edge("classify", "select")
        graph.add_edge("select", "execute")
        graph.add_edge("execute", "verify")
        graph.add_edge("verify", "reflect")
        graph.add_edge("reflect", "checkpoint")
        graph.add_conditional_edges(
            "checkpoint",
            self._checkpoint_route,
            {
                MissionRoute.AUTO.value: "select",
                MissionRoute.FINALIZE.value: "finalize",
            },
        )
        graph.add_edge("finalize", END)
        return graph.compile()

    def _load(self, state: MissionGraphState) -> MissionGraphState:
        request = state["request"]
        session, trace_start = self._state.load_with_trace(request.session_id)
        return {
            "session": session,
            "trace_start": trace_start,
            "message": request.message.strip() if request.message else None,
            "turn_executions": 0,
        }

    def _classify(self, state: MissionGraphState) -> MissionGraphState:
        route = self._policy.classify(state["session"], state.get("message"))
        return {"route": route.value}

    def _select(self, state: MissionGraphState) -> MissionGraphState:
        selection = self._policy.select(state["session"], MissionRoute(state["route"]))
        return {"selection": selection, "before_action": state["session"]}

    def _execute(self, state: MissionGraphState) -> MissionGraphState:
        selection = state["selection"]
        try:
            if selection.milestone is not None and self._policy.is_optional_artifact(
                selection.milestone
            ):
                step = self._actions.execute(
                    state["session"],
                    selection,
                    state.get("message"),
                    self._maximum_actions_per_turn
                    - state.get("turn_executions", 0)
                    - 1,
                )
            else:
                step = self._actions.execute(
                    state["session"],
                    selection,
                    state.get("message"),
                    self._maximum_actions_per_turn
                    - state.get("turn_executions", 0)
                    - 1,
                )
        except (RuntimeError, ValueError) as error:
            persisted = self._state.current(state["session"].identifier)
            session = persisted or state["session"]
            if selection.milestone is not None and self._policy.is_optional_artifact(
                selection.milestone
            ):
                step = self._actions.skip_optional(
                    session,
                    selection,
                    session.action_count - state["session"].action_count,
                )
            else:
                return {"error": str(error), "session": session}
        return {
            "session": step.session,
            "step": step,
            "activity": self._actions.combine_activity(
                state.get("activity"), step.activity
            ),
            "turn_executions": state.get("turn_executions", 0)
            + step.capability_executions,
        }

    def _verify(self, state: MissionGraphState) -> MissionGraphState:
        if "error" in state:
            return {}
        step = state["step"]
        try:
            self._verification.verify(
                state["before_action"],
                step,
                state.get("turn_executions", 0),
                self._maximum_actions_per_turn,
                self._maximum_actions_per_session,
            )
        except (RuntimeError, ValueError) as error:
            return {"error": str(error)}
        return {}

    def _reflect(self, state: MissionGraphState) -> MissionGraphState:
        if "error" in state:
            step = self._actions.fail(state["session"])
            return {
                "session": step.session,
                "activity": step.activity,
                "route": MissionRoute.FINALIZE.value,
            }
        step = state["step"]
        route = self._policy.next_route(
            state["session"],
            state.get("activity"),
            step.continue_auto,
            state.get("turn_executions", 0),
            self._maximum_actions_per_turn,
        )
        return {"route": route.value}

    def _checkpoint(self, state: MissionGraphState) -> MissionGraphState:
        session = state["session"]
        if (
            state["route"] == MissionRoute.FINALIZE.value
            and session.status.value == "active"
            and state.get("activity") is None
            and session.pending_interaction is None
        ):
            session = self._state.checkpoint(
                session,
                "wait",
                "Mission is ready for the next learner action.",
            )
        else:
            session = self._state.save(session)
        return {"session": session}

    def _checkpoint_route(self, state: MissionGraphState) -> str:
        if state.get("route") == MissionRoute.AUTO.value:
            return MissionRoute.AUTO.value
        return MissionRoute.FINALIZE.value

    def _finalize(self, state: MissionGraphState) -> MissionGraphState:
        return {
            "result": self._results.build(
                state["session"],
                state.get("activity"),
                state["trace_start"],
            )
        }
