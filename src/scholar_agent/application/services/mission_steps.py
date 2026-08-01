"""Small application result types shared by mission policy services."""

from dataclasses import dataclass

from scholar_agent.application.dtos.tutor import TutorActivity
from scholar_agent.domain.entities.study_session import StudySession


@dataclass(frozen=True, slots=True)
class MissionStep:
    """Result of one graph execute node."""

    session: StudySession
    activity: TutorActivity | None
    capability_executions: int
    continue_auto: bool = False
