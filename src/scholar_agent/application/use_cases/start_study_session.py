"""Start one persistent adaptive study session."""

from datetime import UTC, datetime
from uuid import uuid4

from scholar_agent.application.dtos.tutor import (
    StartStudySessionRequest,
    StudySessionResult,
    TutorActivity,
)
from scholar_agent.application.services.request_validation_service import (
    RequestValidationService,
)
from scholar_agent.application.use_cases.build_document_brief import (
    BuildDocumentBriefUseCase,
)
from scholar_agent.domain.entities.study_session import (
    StudySession,
    TutorTurnKind,
    objective_progress,
)
from scholar_agent.domain.repositories.study_session_repository import (
    StudySessionRepository,
)


class StartStudySessionUseCase:
    """Create a session and select its first prerequisite-ready objective."""

    def __init__(
        self,
        brief_use_case: BuildDocumentBriefUseCase,
        session_repository: StudySessionRepository,
        validation_service: RequestValidationService,
    ) -> None:
        self._brief_use_case = brief_use_case
        self._session_repository = session_repository
        self._validation_service = validation_service

    def execute(self, request: StartStudySessionRequest) -> StudySessionResult:
        """Validate, persist, and return a new guided session."""
        goal = self._validation_service.validate_text(request.goal, "goal")
        if request.target_minutes < 5 or request.target_minutes > 240:
            raise ValueError("target_minutes must be between 5 and 240.")
        brief = self._brief_use_case.execute(request.document_id).brief
        now = datetime.now(UTC)
        session = StudySession(
            identifier=str(uuid4()),
            document_id=request.document_id,
            goal=goal,
            learner_level=request.learner_level,
            mode=request.mode,
            target_minutes=request.target_minutes,
            brief=brief,
            created_at=now,
            updated_at=now,
        )
        self._session_repository.save(session)
        objective = brief.objectives[0]
        activity = TutorActivity(
            kind=TutorTurnKind.QUESTION,
            message=(
                f"We'll begin with **{objective.title}**. Before I explain it, "
                f"what do you already understand about this idea?"
            ),
            objective_id=objective.identifier,
            citations=objective.citations,
        )
        return StudySessionResult(
            session=session,
            progress=tuple(
                objective_progress(item.identifier, ()) for item in brief.objectives
            ),
            current_objective_id=objective.identifier,
            activity=activity,
        )
