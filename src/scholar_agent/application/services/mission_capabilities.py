"""Application service for bounded, document-bound capability execution."""

from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime

from scholar_agent.application.output_ports.tool_executor import IToolExecutor
from scholar_agent.application.services.mission_state import MissionStateService
from scholar_agent.domain.entities.study_session import (
    MilestoneStatus,
    SourceReference,
    StudySession,
)
from scholar_agent.domain.value_objects.document_id import DocumentId


class MissionCapabilityService:
    """Inject scope, enforce the session budget, and checkpoint each call."""

    def __init__(
        self,
        tool_executor: IToolExecutor,
        state_service: MissionStateService,
        maximum_actions_per_session: int = 64,
    ) -> None:
        self._tool_executor = tool_executor
        self._state = state_service
        self._maximum_actions_per_session = maximum_actions_per_session

    def execute(
        self,
        session: StudySession,
        capability: str,
        arguments: Mapping[str, object],
    ) -> tuple[StudySession, Mapping[str, object]]:
        """Execute one capability with the selected document injected."""
        if session.action_count >= self._maximum_actions_per_session:
            raise ValueError("The mission action limit has been reached.")
        tool_arguments = dict(arguments)
        tool_arguments["document_id"] = session.document_id.value
        payload = self._tool_executor.execute(capability, tool_arguments)
        updated = replace(
            session,
            action_count=session.action_count + 1,
            updated_at=datetime.now(UTC),
        )
        objective_id = arguments.get("objective_id")
        citations = _payload_citations(payload, session.document_id)
        return (
            self._state.checkpoint(
                updated,
                "capability",
                f"Completed {capability}.",
                capability,
                objective_id=objective_id if isinstance(objective_id, str) else None,
                citations=citations,
                transition_key=(
                    f"capability:{session.identifier}:{updated.action_count}:"
                    f"{capability}"
                ),
            ),
            payload,
        )

    def complete_milestone(
        self, session: StudySession, identifier: str
    ) -> StudySession:
        """Complete one milestone and activate its immediate successor."""
        target_index = next(
            (
                index
                for index, item in enumerate(session.milestones)
                if item.identifier == identifier
            ),
            None,
        )
        if target_index is None:
            raise ValueError(f"Unknown milestone '{identifier}'.")
        milestones = tuple(
            replace(
                item,
                status=(
                    MilestoneStatus.COMPLETED
                    if index == target_index
                    else (
                        MilestoneStatus.ACTIVE
                        if index == target_index + 1
                        and item.status is MilestoneStatus.PENDING
                        else item.status
                    )
                ),
            )
            for index, item in enumerate(session.milestones)
        )
        return self._state.checkpoint(
            replace(session, milestones=milestones),
            "state",
            f"Milestone {identifier} completed.",
            objective_id=next(
                (
                    item.objective_id
                    for item in session.milestones
                    if item.identifier == identifier
                ),
                None,
            ),
            transition_key=f"milestone:{session.identifier}:{identifier}:{session.action_count}",
        )


def _payload_citations(
    payload: Mapping[str, object], document_id: DocumentId
) -> tuple[SourceReference, ...]:
    """Extract only validated source references returned by a capability."""
    raw_values: list[object] = []
    citations = payload.get("citations")
    if isinstance(citations, list):
        raw_values.extend(citations)
    chunks = payload.get("chunks")
    if isinstance(chunks, list):
        raw_values.extend(chunks)
    references: list[SourceReference] = []
    for value in raw_values:
        if not isinstance(value, Mapping):
            continue
        raw_document = value.get("document_id")
        chunk_id = value.get("chunk_id")
        if raw_document != document_id.value or not isinstance(chunk_id, str):
            continue
        page_number = value.get("page_number")
        excerpt = value.get("excerpt", "")
        if page_number is not None and (
            isinstance(page_number, bool) or not isinstance(page_number, int)
        ):
            continue
        if not isinstance(excerpt, str):
            excerpt = ""
        reference = SourceReference(
            document_id=document_id,
            chunk_id=chunk_id,
            page_number=page_number,
            excerpt=excerpt,
        )
        if reference not in references:
            references.append(reference)
    return tuple(references)
