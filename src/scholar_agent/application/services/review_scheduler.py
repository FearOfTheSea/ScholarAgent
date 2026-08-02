"""Pure fixed-clock review scheduling for document-local concepts."""

from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, time, timedelta

from scholar_agent.application.dtos.learner_profile import ReviewQueueEntry
from scholar_agent.application.services.knowledge_tracing import KnowledgeTracingService
from scholar_agent.domain.entities.learner_profile import (
    ConceptEquivalenceLink,
    ConceptFingerprint,
    EquivalenceDecision,
    EvidenceObservation,
    LearnerProfile,
)


class ReviewScheduler:
    """Schedule deterministic, private review recommendations."""

    def __init__(
        self,
        tracing: KnowledgeTracingService | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))
        self._tracing = tracing or KnowledgeTracingService(self._clock)

    def queue(
        self,
        profile: LearnerProfile,
        observations: Iterable[EvidenceObservation],
        links: Iterable[ConceptEquivalenceLink] = (),
        as_of: datetime | None = None,
    ) -> tuple[ReviewQueueEntry, ...]:
        """Build and order queue entries from explicitly accepted history."""
        moment = _utc(as_of or self._clock())
        all_observations = tuple(observations)
        groups = _accepted_groups(all_observations, links)
        entries: list[ReviewQueueEntry] = []
        for fingerprint in sorted(
            {item.fingerprint for item in all_observations}, key=lambda item: item.value
        ):
            group = groups[_group_key(fingerprint)]
            estimate = self._tracing.estimate(group, fingerprint, moment)
            source = min(
                (item for item in all_observations if item.fingerprint == fingerprint),
                key=lambda item: (item.observed_at, item.objective_id),
            )
            transfer_gap = estimate.transfer_count == 0
            interval_days = _interval_days(estimate.confidence, estimate.uncertainty)
            if transfer_gap:
                interval_days = min(interval_days, 7)
            last_observed = estimate.last_observed_at or moment
            due_at = last_observed + timedelta(days=interval_days)
            reason_codes = _reason_codes(
                estimate.confidence,
                estimate.uncertainty,
                transfer_gap,
            )
            if profile.target_date is not None:
                target = datetime.combine(profile.target_date, time.min, tzinfo=UTC)
                if due_at > target:
                    due_at = target
                    reason_codes = reason_codes + ("target_date",)
            expected_minutes = _expected_minutes(
                estimate.confidence, estimate.uncertainty, transfer_gap
            )
            entries.append(
                ReviewQueueEntry(
                    fingerprint,
                    fingerprint.document_id.value,
                    source.objective_id,
                    fingerprint.normalized_title,
                    fingerprint.normalized_description,
                    estimate.confidence,
                    estimate.uncertainty,
                    estimate.observation_count,
                    estimate.recall_count,
                    estimate.transfer_count,
                    estimate.last_observed_at,
                    due_at,
                    min(15, expected_minutes),
                    reason_codes,
                    tuple(sorted({item.document_id.value for item in group})),
                )
            )
        return tuple(sorted(entries, key=lambda item: _queue_sort_key(item, moment)))


def _accepted_groups(
    observations: tuple[EvidenceObservation, ...],
    links: Iterable[ConceptEquivalenceLink],
) -> dict[str, tuple[EvidenceObservation, ...]]:
    fingerprints = {item.fingerprint for item in observations}
    parents = {item.value: item.value for item in fingerprints}

    def find(value: str) -> str:
        parent = parents[value]
        if parent != value:
            parents[value] = find(parent)
        return parents[value]

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for link in links:
        if link.decision is EquivalenceDecision.ACCEPTED:
            if link.source.value in parents and link.target.value in parents:
                union(link.source.value, link.target.value)
    grouped: dict[str, list[EvidenceObservation]] = defaultdict(list)
    for observation in observations:
        grouped[find(observation.fingerprint.value)].append(observation)
    return {
        fingerprint.value: tuple(grouped[find(fingerprint.value)])
        for fingerprint in fingerprints
    }


def _group_key(fingerprint: ConceptFingerprint) -> str:
    return fingerprint.value


def _interval_days(confidence: int, uncertainty: int) -> int:
    if confidence < 35 or uncertainty >= 70:
        return 1
    if confidence < 55:
        return 3
    if confidence < 70:
        return 7
    if confidence < 85:
        return 14
    return 30


def _reason_codes(
    confidence: int, uncertainty: int, transfer_gap: bool
) -> tuple[str, ...]:
    reasons: list[str] = []
    if confidence < 55:
        reasons.append("low_confidence")
    if uncertainty >= 70:
        reasons.append("high_uncertainty")
    if transfer_gap:
        reasons.append("transfer_gap")
    if not reasons:
        reasons.append("scheduled_review")
    return tuple(reasons)


def _expected_minutes(confidence: int, uncertainty: int, transfer_gap: bool) -> int:
    base = 10 if confidence < 40 else (7 if confidence < 70 else 5)
    bonus = 3 if uncertainty >= 70 or transfer_gap else 0
    return min(15, base + bonus)


def _queue_sort_key(
    entry: ReviewQueueEntry, moment: datetime
) -> tuple[bool, datetime, int, str]:
    return (
        entry.due_at > moment,
        entry.due_at,
        -entry.uncertainty,
        entry.fingerprint.value,
    )


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
