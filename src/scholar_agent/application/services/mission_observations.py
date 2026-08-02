"""Redacted, idempotent reconstruction of mission assessment evidence."""

from collections.abc import Callable
from datetime import UTC, datetime

from scholar_agent.domain.entities.learner_profile import (
    ConceptFingerprint,
    EvidenceObservation,
    ObservationModality,
    ObservationSource,
)
from scholar_agent.domain.entities.study_session import StudySession
from scholar_agent.domain.repositories.learner_profile_repository import (
    LearnerProfileRepository,
)
from scholar_agent.domain.repositories.study_session_repository import (
    StudySessionRepository,
)
from scholar_agent.domain.value_objects.citation_identity import CitationIdentity


class MissionObservationSyncService:
    """Rebuild missing observations after a profile-write failure."""

    def __init__(
        self,
        profile_repository: LearnerProfileRepository,
        session_repository: StudySessionRepository,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._profiles = profile_repository
        self._sessions = session_repository
        self._clock = clock or (lambda: datetime.now(UTC))

    def sync_session(self, session: StudySession) -> int:
        """Append all reconstructable assessment observations exactly once."""
        if session.learner_profile_id is None:
            return 0
        profile_id = session.learner_profile_id
        objective_by_id = {item.identifier: item for item in session.brief.objectives}
        inserted = 0
        for index, attempt in enumerate(session.attempts):
            objective = objective_by_id.get(attempt.objective_id)
            if objective is None:
                continue
            fingerprint = ConceptFingerprint.from_descriptor(
                session.document_id, objective.title, objective.description
            )
            previous = tuple(
                item
                for item in session.attempts[:index]
                if item.objective_id == attempt.objective_id
            )
            modality = (
                ObservationModality.TRANSFER
                if previous and previous[-1].score >= 2
                else ObservationModality.RECALL
            )
            observation = EvidenceObservation.for_mission(
                profile_id,
                fingerprint,
                attempt.objective_id,
                session.identifier,
                index,
                ObservationSource.MISSION,
                modality,
                attempt.score,
                1 + min(2, len(attempt.missing_concepts)),
                tuple(
                    CitationIdentity.from_reference(item) for item in attempt.citations
                ),
                attempt.created_at,
            )
            if self._profiles.append_observation(observation):
                inserted += 1
        return inserted

    def sync_profile(self, profile_id: str) -> int:
        """Repair all associated sessions for one profile."""
        return sum(
            self.sync_session(session)
            for session in self._sessions.list()
            if session.learner_profile_id == profile_id
        )

    def sync_best_effort(self, session: StudySession) -> int:
        """Attempt post-save profile persistence without invalidating mission state."""
        try:
            return self.sync_session(session)
        except Exception:
            return 0
