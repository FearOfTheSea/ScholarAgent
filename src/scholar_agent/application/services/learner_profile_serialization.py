"""Versioned redacted learner-profile import/export serialization."""

from datetime import date, datetime

from scholar_agent.application.dtos.learner_profile import LearnerProfileExport
from scholar_agent.domain.entities.learner_profile import (
    ConceptEquivalenceCandidate,
    ConceptEquivalenceLink,
    ConceptFingerprint,
    EquivalenceDecision,
    EvidenceObservation,
    LearnerProfile,
    ObservationModality,
    ObservationSource,
)
from scholar_agent.domain.value_objects.citation_identity import CitationIdentity
from scholar_agent.domain.value_objects.document_id import DocumentId

FORBIDDEN_EXPORT_KEYS = frozenset(
    {
        "response",
        "feedback",
        "prompt",
        "model_output",
        "reference_answer",
        "excerpt",
        "source_text",
        "turns",
        "pdf_content",
    }
)


class LearnerProfileSerializationService:
    """Build and validate profile payloads without learner or source content."""

    def export(
        self,
        profile: LearnerProfile,
        observations: tuple[EvidenceObservation, ...],
        candidates: tuple[ConceptEquivalenceCandidate, ...],
        links: tuple[ConceptEquivalenceLink, ...],
    ) -> dict[str, object]:
        fingerprints = (
            {item.fingerprint for item in observations}
            | {item.source for item in candidates}
            | {item.target for item in candidates}
        )
        fingerprints |= {item.source for item in links} | {
            item.target for item in links
        }
        return {
            "profile_export_version": 1,
            "profile": {
                "identifier": profile.identifier,
                "display_name": profile.display_name,
                "target_date": (
                    profile.target_date.isoformat() if profile.target_date else None
                ),
                "created_at": profile.created_at.isoformat(),
                "updated_at": profile.updated_at.isoformat(),
            },
            "concepts": [
                _fingerprint_payload(item)
                for item in sorted(fingerprints, key=lambda value: value.value)
            ],
            "observations": [_observation_payload(item) for item in observations],
            "equivalence_candidates": [_candidate_payload(item) for item in candidates],
            "equivalence_decisions": [_link_payload(item) for item in links],
            "scheduler_preferences": {},
        }

    def import_payload(
        self, payload: object, expected_profile_id: str
    ) -> LearnerProfileExport:
        _reject_forbidden_keys(payload)
        root = _mapping(payload)
        if root.get("profile_export_version") != 1:
            raise ValueError("Unsupported learner profile export version.")
        profile_payload = _mapping(root.get("profile"))
        profile = LearnerProfile(
            _required_string(profile_payload, "identifier"),
            _required_string(profile_payload, "display_name"),
            _optional_date(profile_payload.get("target_date")),
            _required_datetime(profile_payload, "created_at"),
            _required_datetime(profile_payload, "updated_at"),
        )
        if profile.identifier != expected_profile_id:
            raise ValueError("Imported profile id does not match the route.")
        concepts = {
            item.value: item
            for item in (
                _fingerprint_from_payload(value)
                for value in _objects(root.get("concepts"))
            )
        }
        observations = tuple(
            _observation_from_payload(value, concepts)
            for value in _objects(root.get("observations"))
        )
        candidates = tuple(
            _candidate_from_payload(value, profile.identifier)
            for value in _objects(root.get("equivalence_candidates"))
        )
        links = tuple(
            _link_from_payload(value, profile.identifier)
            for value in _objects(root.get("equivalence_decisions"))
        )
        if root.get("scheduler_preferences") != {}:
            raise ValueError("Unsupported scheduler preferences in profile export.")
        return LearnerProfileExport(profile, observations, candidates, links)


def _fingerprint_payload(fingerprint: ConceptFingerprint) -> dict[str, object]:
    return {
        "algorithm_version": fingerprint.algorithm_version,
        "value": fingerprint.value,
        "document_id": fingerprint.document_id.value,
        "normalized_title": fingerprint.normalized_title,
        "normalized_description": fingerprint.normalized_description,
    }


def _fingerprint_from_payload(payload: object) -> ConceptFingerprint:
    item = _mapping(payload)
    fingerprint = ConceptFingerprint(
        _required_string(item, "algorithm_version"),
        _required_string(item, "value"),
        DocumentId(_required_string(item, "document_id")),
        _required_string(item, "normalized_title"),
        _required_string(item, "normalized_description"),
    )
    expected = ConceptFingerprint.from_descriptor(
        fingerprint.document_id,
        fingerprint.normalized_title,
        fingerprint.normalized_description,
        fingerprint.algorithm_version,
    )
    if expected.value != fingerprint.value:
        raise ValueError("Concept fingerprint does not match its descriptor.")
    return fingerprint


def _citation_payload(citation: CitationIdentity) -> dict[str, object]:
    return {
        "document_id": citation.document_id.value,
        "chunk_id": citation.chunk_id,
        "page_number": citation.page_number,
    }


def _citation_from_payload(payload: object) -> CitationIdentity:
    item = _mapping(payload)
    page = item.get("page_number")
    if page is not None and (isinstance(page, bool) or not isinstance(page, int)):
        raise ValueError("Citation page number is invalid.")
    return CitationIdentity(
        DocumentId(_required_string(item, "document_id")),
        _required_string(item, "chunk_id"),
        page,
    )


def _observation_payload(observation: EvidenceObservation) -> dict[str, object]:
    return {
        "identifier": observation.identifier,
        "profile_id": observation.profile_id,
        "fingerprint": _fingerprint_payload(observation.fingerprint),
        "document_id": observation.document_id.value,
        "objective_id": observation.objective_id,
        "session_id": observation.session_id,
        "source": observation.source.value,
        "modality": observation.modality.value,
        "score": observation.score,
        "difficulty": observation.difficulty,
        "citations": [_citation_payload(item) for item in observation.citations],
        "observed_at": observation.observed_at.isoformat(),
    }


def _observation_from_payload(
    payload: object, concepts: dict[str, ConceptFingerprint]
) -> EvidenceObservation:
    item = _mapping(payload)
    fingerprint = _fingerprint_from_payload(item.get("fingerprint"))
    if concepts.get(fingerprint.value) != fingerprint:
        raise ValueError("Observation fingerprint is missing from concepts.")
    return EvidenceObservation(
        _required_string(item, "identifier"),
        _required_string(item, "profile_id"),
        fingerprint,
        DocumentId(_required_string(item, "document_id")),
        _required_string(item, "objective_id"),
        _optional_string(item.get("session_id")),
        ObservationSource(_required_string(item, "source")),
        ObservationModality(_required_string(item, "modality")),
        _integer(item, "score"),
        _integer(item, "difficulty"),
        tuple(
            _citation_from_payload(value) for value in _objects(item.get("citations"))
        ),
        _required_datetime(item, "observed_at"),
    )


def _candidate_payload(candidate: ConceptEquivalenceCandidate) -> dict[str, object]:
    return {
        "profile_id": candidate.profile_id,
        "source": _fingerprint_payload(candidate.source),
        "target": _fingerprint_payload(candidate.target),
        "similarity": candidate.similarity,
        "created_at": candidate.created_at.isoformat(),
    }


def _candidate_from_payload(
    payload: object, profile_id: str
) -> ConceptEquivalenceCandidate:
    item = _mapping(payload)
    if _required_string(item, "profile_id") != profile_id:
        raise ValueError("Candidate profile id is invalid.")
    similarity = item.get("similarity")
    if isinstance(similarity, bool) or not isinstance(similarity, (int, float)):
        raise ValueError("Candidate similarity is invalid.")
    return ConceptEquivalenceCandidate(
        _fingerprint_from_payload(item.get("source")),
        _fingerprint_from_payload(item.get("target")),
        float(similarity),
        _required_datetime(item, "created_at"),
        profile_id,
    )


def _link_payload(link: ConceptEquivalenceLink) -> dict[str, object]:
    return {
        "profile_id": link.profile_id,
        "source": _fingerprint_payload(link.source),
        "target": _fingerprint_payload(link.target),
        "decision": link.decision.value,
        "decided_at": link.decided_at.isoformat(),
    }


def _link_from_payload(payload: object, profile_id: str) -> ConceptEquivalenceLink:
    item = _mapping(payload)
    if _required_string(item, "profile_id") != profile_id:
        raise ValueError("Equivalence decision profile id is invalid.")
    return ConceptEquivalenceLink(
        _fingerprint_from_payload(item.get("source")),
        _fingerprint_from_payload(item.get("target")),
        EquivalenceDecision(_required_string(item, "decision")),
        _required_datetime(item, "decided_at"),
        profile_id,
    )


def _reject_forbidden_keys(value: object) -> None:
    if isinstance(value, dict):
        if FORBIDDEN_EXPORT_KEYS.intersection(value):
            raise ValueError("Profile export contains forbidden private content.")
        for item in value.values():
            _reject_forbidden_keys(item)
    elif isinstance(value, list):
        for item in value:
            _reject_forbidden_keys(item)


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Profile export object is invalid.")
    return value


def _objects(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("Profile export collection is invalid.")
    return value


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Profile export field '{key}' is invalid.")
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Profile export optional string is invalid.")
    return value


def _required_datetime(payload: dict[str, object], key: str) -> datetime:
    value = _required_string(payload, key)
    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"Profile export datetime '{key}' is invalid.") from error


def _optional_date(value: object) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Profile export target_date is invalid.")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError("Profile export target_date is invalid.") from error


def _integer(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Profile export integer '{key}' is invalid.")
    return value
