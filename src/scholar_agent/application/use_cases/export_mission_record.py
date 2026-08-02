"""Export a versioned, redacted, locally generated mission record."""

from collections.abc import Iterable

from scholar_agent.application.dtos.mission_intelligence import MissionInsights
from scholar_agent.application.use_cases.get_mission_insights import (
    GetMissionInsightsUseCase,
)
from scholar_agent.domain.entities.mission_ledger import (
    MissionLedgerEntry,
    MissionStateProjection,
)
from scholar_agent.domain.entities.study_artifacts import (
    FlashcardArtifact,
    QuizArtifact,
    SummaryArtifact,
)
from scholar_agent.domain.entities.study_session import StudyArtifact, StudySession
from scholar_agent.domain.repositories.study_session_repository import (
    StudySessionRepository,
)
from scholar_agent.domain.value_objects.source_reference import SourceReference


class ExportMissionRecordUseCase:
    """Build an export with no learner responses, prompts, or source text."""

    def __init__(
        self,
        session_repository: StudySessionRepository,
        insights_use_case: GetMissionInsightsUseCase,
    ) -> None:
        self._session_repository = session_repository
        self._insights = insights_use_case

    def execute(self, session_id: str) -> dict[str, object]:
        session = self._session_repository.get(session_id)
        if session is None:
            raise ValueError(f"Study session '{session_id}' was not found.")
        insights = self._insights.execute(session_id)
        if not insights.ledger_verified:
            raise ValueError("The mission ledger must verify before export.")
        return _record(session, insights)


def _record(session: StudySession, insights: MissionInsights) -> dict[str, object]:
    return {
        "record_version": 1,
        "session_schema_version": 4,
        "session": {
            "session_id": session.identifier,
            "document_id": session.document_id.value,
            "goal": session.goal,
            "status": session.status.value,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "completed_at": (
                session.completed_at.isoformat()
                if session.completed_at is not None
                else None
            ),
        },
        "plan": (
            {
                "focus": session.plan.focus,
                "objective_ids": list(session.plan.objective_ids),
                "citations": _citation_identities(session.plan.citations),
            }
            if session.plan is not None
            else None
        ),
        "ledger": [_ledger_record(entry) for entry in session.ledger],
        "insights": _insights_record(insights),
        "citations": _all_citation_identities(session),
        "artifacts": [_artifact_metadata(item) for item in session.artifacts],
    }


def _ledger_record(entry: MissionLedgerEntry) -> dict[str, object]:
    return {
        "sequence": entry.sequence,
        "event_type": _event_value(entry.event_type),
        "summary": entry.summary,
        "objective_id": entry.objective_id,
        "capability": entry.capability,
        "citations": _citation_identities(entry.citations),
        "projection": _projection_record(entry.projection),
        "previous_digest": entry.previous_digest,
        "current_digest": entry.current_digest,
        "created_at": entry.created_at.isoformat(),
    }


def _projection_record(projection: MissionStateProjection) -> dict[str, object]:
    return {
        "status": projection.status,
        "active_milestone_id": projection.active_milestone_id,
        "pending_objective_id": projection.pending_objective_id,
        "action_count": projection.action_count,
        "attempt_count": projection.attempt_count,
        "artifact_count": projection.artifact_count,
        "completed_milestone_count": projection.completed_milestone_count,
        "mastery_by_objective": [
            {"objective_id": objective_id, "label": label}
            for objective_id, label in projection.mastery_by_objective
        ],
        "next_milestone_id": projection.next_milestone_id,
    }


def _insights_record(insights: MissionInsights) -> dict[str, object]:
    return {
        "progress_percent": insights.progress_percent,
        "mastery_counts": dict(insights.mastery_counts),
        "assessment_count": insights.assessment_count,
        "first_pass_proficiency_rate": insights.first_pass_proficiency_rate,
        "remediation_cycles": insights.remediation_cycles,
        "evidence_coverage": insights.evidence_coverage,
        "action_budget_used": insights.action_budget_used,
        "action_budget_remaining": insights.action_budget_remaining,
        "ledger_verified": insights.ledger_verified,
        "next_action": insights.next_action,
        "signals": list(insights.signals),
    }


def _artifact_metadata(artifact: StudyArtifact) -> dict[str, object]:
    if isinstance(artifact, SummaryArtifact):
        count = 1
    elif isinstance(artifact, QuizArtifact):
        count = len(artifact.questions)
    elif isinstance(artifact, FlashcardArtifact):
        count = len(artifact.cards)
    else:
        raise ValueError("Unsupported study artifact.")
    return {
        "kind": _artifact_kind(artifact),
        "item_count": count,
        "citations": _citation_identities(artifact.citations),
        "created_at": artifact.created_at.isoformat(),
    }


def _artifact_kind(artifact: StudyArtifact) -> str:
    if isinstance(artifact, SummaryArtifact):
        return "summary"
    if isinstance(artifact, QuizArtifact):
        return "quiz"
    if isinstance(artifact, FlashcardArtifact):
        return "flashcards"
    raise ValueError("Unsupported study artifact.")


def _all_citation_identities(session: StudySession) -> list[dict[str, object]]:
    references: list[SourceReference] = []
    if session.plan is not None:
        references.extend(session.plan.citations)
    references.extend(
        reference for entry in session.ledger for reference in entry.citations
    )
    references.extend(
        reference for artifact in session.artifacts for reference in artifact.citations
    )
    return _citation_identities(references)


def _citation_identities(
    references: Iterable[SourceReference],
) -> list[dict[str, object]]:
    identities: list[dict[str, object]] = []
    seen: set[tuple[str, str, int | None]] = set()
    for reference in references:
        identity = (
            reference.document_id.value,
            reference.chunk_id,
            reference.page_number,
        )
        if identity in seen:
            continue
        seen.add(identity)
        identities.append(
            {
                "document_id": identity[0],
                "chunk_id": identity[1],
                "page_number": identity[2],
            }
        )
    return identities


def _event_value(event_type: object) -> str:
    return event_type.value if hasattr(event_type, "value") else str(event_type)
