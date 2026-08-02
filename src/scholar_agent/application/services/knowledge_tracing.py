"""Pure deterministic knowledge tracing over redacted observations."""

from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from math import exp

from scholar_agent.application.dtos.learner_profile import KnowledgeEstimate
from scholar_agent.domain.entities.learner_profile import (
    ConceptFingerprint,
    EvidenceObservation,
    ObservationModality,
)

CLOCK_SKEW_TOLERANCE = timedelta(minutes=5)


class KnowledgeTracingService:
    """Calculate confidence and uncertainty without mutable global mastery."""

    def __init__(self, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))

    def estimate(
        self,
        observations: Iterable[EvidenceObservation],
        fingerprint: ConceptFingerprint | None = None,
        as_of: datetime | None = None,
    ) -> KnowledgeEstimate:
        """Return the fixed-formula estimate for one concept's observations."""
        ordered = tuple(observations)
        if not ordered and fingerprint is None:
            raise ValueError("An empty estimate requires a fingerprint.")
        selected = fingerprint or ordered[0].fingerprint
        moment = _utc(as_of or self._clock())
        weighted: list[tuple[float, float]] = []
        for observation in ordered:
            observed_at = _utc(observation.observed_at)
            if observed_at > moment + CLOCK_SKEW_TOLERANCE:
                raise ValueError("Future learner evidence is outside clock tolerance.")
            effective_observed_at = min(observed_at, moment)
            age_days = max(
                0.0,
                (moment - effective_observed_at).total_seconds() / 86400.0,
            )
            decay = 0.5 ** (age_days / 30.0)
            modality_weight = (
                1.5 if observation.modality is ObservationModality.TRANSFER else 1.0
            )
            difficulty_weight = {1: 0.75, 2: 1.0, 3: 1.25}[observation.difficulty]
            weight = modality_weight * difficulty_weight * decay
            weighted.append((weight, observation.score / 3.0))
        total_weight = sum(weight for weight, _ in weighted)
        weighted_success = (
            sum(weight * score for weight, score in weighted) / total_weight
            if total_weight
            else 0.0
        )
        evidence_strength = 1 - exp(-total_weight / 2.5) if total_weight else 0.0
        confidence = round(100 * weighted_success * evidence_strength)
        dispersion = (
            sum(weight * abs(score - weighted_success) for weight, score in weighted)
            / total_weight
            if total_weight
            else 0.0
        )
        uncertainty = round(
            100 * min(1.0, 0.7 * (1 - evidence_strength) + 0.3 * dispersion)
        )
        return KnowledgeEstimate(
            selected,
            len(ordered),
            sum(item.modality is ObservationModality.RECALL for item in ordered),
            sum(item.modality is ObservationModality.TRANSFER for item in ordered),
            max(
                (_utc(item.observed_at) for item in ordered),
                default=None,
            ),
            max(0, min(100, confidence)),
            max(0, min(100, uncertainty)),
            _mastery_label(confidence, len(ordered)),
            total_weight,
        )


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _mastery_label(confidence: int, observation_count: int) -> str:
    if observation_count == 0:
        return "unseen"
    if confidence >= 80:
        return "mastered"
    if confidence >= 50:
        return "proficient"
    return "developing"
