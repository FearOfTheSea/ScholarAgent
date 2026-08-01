"""Verify one persisted mission ledger and report its first broken link."""

from scholar_agent.domain.entities.mission_ledger import (
    LedgerVerificationResult,
    verify_ledger,
)
from scholar_agent.domain.repositories.study_session_repository import (
    StudySessionRepository,
)


class VerifyMissionLedgerUseCase:
    """Expose actionable sequence and digest verification."""

    def __init__(self, session_repository: StudySessionRepository) -> None:
        self._session_repository = session_repository

    def execute(self, session_id: str) -> LedgerVerificationResult:
        try:
            session = self._session_repository.get(session_id)
        except RuntimeError as error:
            return LedgerVerificationResult(False, None, str(error))
        if session is None:
            raise ValueError(f"Study session '{session_id}' was not found.")
        return verify_ledger(session.ledger)
