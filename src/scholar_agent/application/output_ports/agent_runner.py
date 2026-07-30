"""Port for the unified study-agent workflow adapter."""

from abc import ABC, abstractmethod

from scholar_agent.application.dtos.agent import (
    AskStudyAgentRequest,
    AskStudyAgentResult,
)


class IAgentRunner(ABC):
    """Plans and runs a constrained study-agent workflow."""

    @abstractmethod
    def run(self, request: AskStudyAgentRequest) -> AskStudyAgentResult:
        """Run the agent for one validated application request."""
