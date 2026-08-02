"""Focused Phase 2 domain, persistence, tracing, and review tests."""

import json
import sqlite3
from dataclasses import asdict, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from test_mission_intelligence import _session

from scholar_agent.application.dtos.learner_profile import RecordReviewOutcomeRequest
from scholar_agent.application.services.knowledge_tracing import KnowledgeTracingService
from scholar_agent.application.services.mission_assessment import (
    MissionAssessmentService,
)
from scholar_agent.application.services.mission_observations import (
    MissionObservationSyncService,
)
from scholar_agent.application.services.mission_policy import MissionPolicy
from scholar_agent.application.services.mission_state import MissionStateService
from scholar_agent.application.services.review_scheduler import (
    ReviewScheduler,
    _expected_minutes,
    _interval_days,
    _queue_sort_key,
    _reason_codes,
)
from scholar_agent.application.use_cases.export_learner_profile import (
    ExportLearnerProfileUseCase,
)
from scholar_agent.application.use_cases.import_learner_profile import (
    ImportLearnerProfileRequest,
    ImportLearnerProfileUseCase,
)
from scholar_agent.application.use_cases.record_review_outcome import (
    RecordReviewOutcomeUseCase,
)
from scholar_agent.domain.entities.learner_profile import (
    ConceptEquivalenceCandidate,
    ConceptEquivalenceLink,
    ConceptFingerprint,
    EquivalenceDecision,
    EvidenceObservation,
    LearnerProfile,
    ObservationModality,
)
from scholar_agent.domain.entities.study_session import (
    LearnerAttempt,
    PendingLearnerInteraction,
)
from scholar_agent.domain.value_objects.citation_identity import CitationIdentity
from scholar_agent.domain.value_objects.document_id import DocumentId
from scholar_agent.infrastructure.adapters.sqlite_learner_profile_repository import (
    SQLiteLearnerProfileRepository,
    _candidate_payload,
    _link_payload,
    _pair_key,
)
from scholar_agent.infrastructure.adapters.sqlite_study_session_repository import (
    SQLiteStudySessionRepository,
    _session_payload,
)

NOW = datetime(2026, 8, 1, 12, tzinfo=UTC)


def _fingerprint(document: str, title: str = "Memory") -> ConceptFingerprint:
    return ConceptFingerprint.from_descriptor(
        DocumentId(document), title, "A durable learning concept."
    )


def _observation(
    profile_id: str,
    fingerprint: ConceptFingerprint,
    score: int = 3,
    modality: ObservationModality = ObservationModality.RECALL,
    observed_at: datetime = NOW,
    suffix: str = "",
) -> EvidenceObservation:
    citation = CitationIdentity(fingerprint.document_id, "chunk-1", 1)
    return EvidenceObservation.for_review(
        profile_id,
        fingerprint,
        "objective-1",
        modality,
        score,
        2,
        (citation,),
        observed_at,
        session_id=suffix or None,
    )


def test_fingerprint_is_stable_and_explainable() -> None:
    first = ConceptFingerprint.from_descriptor(
        DocumentId("doc-1"), "  Neural—Networks! ", "Learn\tfeatures."
    )
    second = ConceptFingerprint.from_descriptor(
        DocumentId("doc-1"), "neural networks", "learn features"
    )

    assert first == second
    assert first.algorithm_version == "nfkc-casefold-punct-v1"
    assert first.descriptor == "neural networks: learn features"


def test_fingerprint_rejects_forged_digest_and_unsupported_algorithm() -> None:
    fingerprint = _fingerprint("doc-1")
    with pytest.raises(ValueError, match="does not match"):
        ConceptFingerprint(
            fingerprint.algorithm_version,
            "0" * 64,
            fingerprint.document_id,
            fingerprint.normalized_title,
            fingerprint.normalized_description,
        )
    with pytest.raises(ValueError, match="Unsupported"):
        ConceptFingerprint(
            "future-v2",
            fingerprint.value,
            fingerprint.document_id,
            fingerprint.normalized_title,
            fingerprint.normalized_description,
        )


def test_tracing_weights_transfer_more_and_stale_evidence_decays() -> None:
    fingerprint = _fingerprint("doc-1")
    recall = _observation("profile", fingerprint, modality=ObservationModality.RECALL)
    transfer = _observation(
        "profile", fingerprint, modality=ObservationModality.TRANSFER, suffix="r"
    )
    tracing = KnowledgeTracingService(lambda: NOW)
    recall_estimate = tracing.estimate((recall,), as_of=NOW)
    transfer_estimate = tracing.estimate((transfer,), as_of=NOW)
    stale_estimate = tracing.estimate(
        (_observation("profile", fingerprint, observed_at=NOW - timedelta(days=90)),),
        as_of=NOW,
    )

    assert transfer_estimate.confidence > recall_estimate.confidence
    assert stale_estimate.confidence < recall_estimate.confidence


def test_tracing_matches_hand_calculated_decay_weights() -> None:
    fingerprint = _fingerprint("doc-1")
    observation = _observation(
        "profile",
        fingerprint,
        score=3,
        modality=ObservationModality.TRANSFER,
        observed_at=NOW - timedelta(days=30),
    )
    observation = replace(observation, difficulty=3)

    estimate = KnowledgeTracingService(lambda: NOW).estimate((observation,), as_of=NOW)

    # 1.5 transfer * 1.25 difficulty * 0.5 thirty-day decay = 0.9375.
    assert estimate.total_weight == pytest.approx(0.9375)
    assert estimate.confidence == 31
    assert estimate.uncertainty == 48


def test_tracing_rejects_future_evidence_beyond_clock_tolerance() -> None:
    fingerprint = _fingerprint("doc-1")
    future = _observation(
        "profile", fingerprint, observed_at=NOW + timedelta(minutes=6)
    )
    with pytest.raises(ValueError, match="Future learner evidence"):
        KnowledgeTracingService(lambda: NOW).estimate((future,), as_of=NOW)

    tolerated = replace(
        future,
        observed_at=NOW + timedelta(minutes=4),
        identifier=future.identifier[:-1] + "0",
    )
    estimate = KnowledgeTracingService(lambda: NOW).estimate((tolerated,), as_of=NOW)
    assert estimate.total_weight == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("confidence", "uncertainty", "expected_days"),
    (
        (34, 0, 1),
        (35, 0, 3),
        (54, 0, 3),
        (55, 0, 7),
        (69, 0, 7),
        (70, 0, 14),
        (84, 0, 14),
        (85, 0, 30),
        (85, 70, 1),
    ),
)
def test_scheduler_interval_boundaries(
    confidence: int, uncertainty: int, expected_days: int
) -> None:
    assert _interval_days(confidence, uncertainty) == expected_days


@pytest.mark.parametrize(
    ("confidence", "uncertainty", "transfer_gap", "expected"),
    (
        (54, 0, False, ("low_confidence",)),
        (55, 0, False, ("scheduled_review",)),
        (85, 70, False, ("high_uncertainty",)),
        (85, 0, True, ("transfer_gap",)),
    ),
)
def test_scheduler_reason_boundaries(
    confidence: int,
    uncertainty: int,
    transfer_gap: bool,
    expected: tuple[str, ...],
) -> None:
    assert _reason_codes(confidence, uncertainty, transfer_gap) == expected


@pytest.mark.parametrize(
    ("confidence", "uncertainty", "transfer_gap", "expected_minutes"),
    (
        (39, 0, False, 10),
        (40, 0, False, 7),
        (69, 0, False, 7),
        (70, 0, False, 5),
        (70, 70, False, 8),
        (70, 0, True, 8),
        (39, 70, True, 13),
    ),
)
def test_scheduler_expected_minute_bands_and_bonuses(
    confidence: int,
    uncertainty: int,
    transfer_gap: bool,
    expected_minutes: int,
) -> None:
    assert _expected_minutes(confidence, uncertainty, transfer_gap) == expected_minutes


def test_scheduler_order_is_due_then_uncertainty_then_fingerprint() -> None:
    fingerprint_a = _fingerprint("doc-1", "A")
    fingerprint_b = _fingerprint("doc-2", "B")
    fingerprint_c = _fingerprint("doc-3", "C")
    from scholar_agent.application.dtos.learner_profile import ReviewQueueEntry

    entries = (
        ReviewQueueEntry(
            fingerprint_c,
            "doc-3",
            "objective-c",
            "C",
            "C",
            80,
            10,
            1,
            1,
            0,
            NOW,
            NOW + timedelta(days=1),
            5,
            ("transfer_gap",),
            ("doc-3",),
        ),
        ReviewQueueEntry(
            fingerprint_b,
            "doc-2",
            "objective-b",
            "B",
            "B",
            80,
            20,
            1,
            1,
            0,
            NOW,
            NOW,
            5,
            ("transfer_gap",),
            ("doc-2",),
        ),
        ReviewQueueEntry(
            fingerprint_a,
            "doc-1",
            "objective-a",
            "A",
            "A",
            80,
            10,
            1,
            1,
            0,
            NOW,
            NOW,
            5,
            ("transfer_gap",),
            ("doc-1",),
        ),
    )
    ordered = sorted(entries, key=lambda item: _queue_sort_key(item, NOW))

    assert [item.document_id for item in ordered] == ["doc-2", "doc-1", "doc-3"]


def test_scheduler_orders_due_items_and_caps_at_target_date() -> None:
    fingerprint = _fingerprint("doc-1")
    profile = LearnerProfile("profile", "Learner", date(2026, 8, 5), NOW, NOW)
    observations = tuple(
        _observation("profile", fingerprint, observed_at=NOW, suffix=str(index))
        for index in range(4)
    )

    queue = ReviewScheduler(clock=lambda: NOW).queue(profile, observations, as_of=NOW)

    assert len(queue) == 1
    assert queue[0].due_at == datetime(2026, 8, 5, tzinfo=UTC)
    assert "target_date" in queue[0].reason_codes


def test_scheduler_transfer_gap_caps_even_high_confidence_interval() -> None:
    fingerprint = _fingerprint("doc-1")
    profile = LearnerProfile("profile", "Learner", None, NOW, NOW)
    observations = tuple(
        _observation("profile", fingerprint, observed_at=NOW, suffix=str(index))
        for index in range(12)
    )

    queue = ReviewScheduler(clock=lambda: NOW).queue(profile, observations, as_of=NOW)

    assert queue[0].due_at == NOW + timedelta(days=7)
    assert "transfer_gap" in queue[0].reason_codes


def test_profile_export_delete_import_round_trips_redacted_history(
    tmp_path: Path,
) -> None:
    repository = SQLiteLearnerProfileRepository(tmp_path / "profiles.sqlite3")
    profile = repository.get_or_create_default(NOW)
    observation = _observation(profile.identifier, _fingerprint("doc-1"))
    repository.append_observation(observation)
    export = ExportLearnerProfileUseCase(repository).execute(profile.identifier)
    assert "excerpt" not in json.dumps(export)
    repository.delete_profile(profile.identifier)
    ImportLearnerProfileUseCase(repository).execute(
        ImportLearnerProfileRequest(profile.identifier, export)
    )
    restored = ExportLearnerProfileUseCase(repository).execute(profile.identifier)

    assert restored == export
    repository.close()


def test_mission_observation_sync_is_idempotent_and_redacted(tmp_path: Path) -> None:
    sessions = SQLiteStudySessionRepository(tmp_path / "sessions.sqlite3")
    profiles = SQLiteLearnerProfileRepository(tmp_path / "profiles.sqlite3", sessions)
    profile = profiles.get_or_create_default(NOW)
    session = replace(_session(), learner_profile_id=profile.identifier)
    reference = session.brief.objectives[0].citations[0]
    attempt = LearnerAttempt(
        "objective-1", "private response", 3, "private feedback", (), (reference,), NOW
    )
    session = replace(session, attempts=(attempt,))
    sessions.save(session)
    sync = MissionObservationSyncService(profiles, sessions, lambda: NOW)

    assert sync.sync_profile(profile.identifier) == 1
    assert sync.sync_profile(profile.identifier) == 0
    stored = profiles.list_observations(profile.identifier)
    assert len(stored) == 1
    assert stored[0].modality is ObservationModality.RECALL
    assert "private response" not in json.dumps(asdict(stored[0]), default=str)
    profiles.close()
    sessions.close()


def test_proficient_mission_check_creates_transfer_challenge_and_observation(
    tmp_path: Path,
) -> None:
    sessions = SQLiteStudySessionRepository(tmp_path / "sessions.sqlite3")
    profiles = SQLiteLearnerProfileRepository(tmp_path / "profiles.sqlite3", sessions)
    profile = profiles.get_or_create_default(NOW)
    reference = _session().brief.objectives[0].citations[0]
    session = replace(
        _session(),
        learner_profile_id=profile.identifier,
        pending_interaction=PendingLearnerInteraction(
            "objective-1", "Recall the concept.", citations=(reference,)
        ),
    )

    class FakeCapabilities:
        def execute(self, current, capability, arguments):
            assert capability == "assess_learner_response"
            return current, {
                "score": 3,
                "feedback": "Good.",
                "missing_concepts": [],
                "next_question": "Repeat the definition.",
                "citations": [
                    {
                        "document_id": "document-1",
                        "chunk_id": "chunk-1",
                        "page_number": 1,
                        "excerpt": "Evidence",
                    }
                ],
            }

        def complete_milestone(self, current, identifier):
            return current

    sync = MissionObservationSyncService(profiles, sessions, lambda: NOW)
    state = MissionStateService(sessions, observation_sync=sync)
    assessment = MissionAssessmentService(FakeCapabilities(), state, MissionPolicy())

    first = assessment.assess(session, "First answer")
    assert first.session.pending_interaction is not None
    question = first.session.pending_interaction.question
    assert "Transfer/application challenge" in question
    assert "new situation or example" in question
    assert "One" in question
    assert "One idea" in question
    assert "objective-1" not in question
    assert [
        item.modality for item in profiles.list_observations(profile.identifier)
    ] == [ObservationModality.RECALL]

    assessment.assess(first.session, "Transfer answer")
    assert [
        item.modality for item in profiles.list_observations(profile.identifier)
    ] == [ObservationModality.RECALL, ObservationModality.TRANSFER]
    profiles.close()
    sessions.close()


def test_remediation_retry_is_reconstructed_as_recall(tmp_path: Path) -> None:
    sessions = SQLiteStudySessionRepository(tmp_path / "sessions.sqlite3")
    profiles = SQLiteLearnerProfileRepository(tmp_path / "profiles.sqlite3", sessions)
    profile = profiles.get_or_create_default(NOW)
    session = _session()
    reference = session.brief.objectives[0].citations[0]
    attempts = tuple(
        LearnerAttempt(
            "objective-1",
            f"answer-{score}",
            score,
            "feedback",
            (),
            (reference,),
            NOW + timedelta(seconds=index),
        )
        for index, score in enumerate((3, 1, 3))
    )
    session = replace(session, learner_profile_id=profile.identifier, attempts=attempts)
    sessions.save(session)
    MissionObservationSyncService(profiles, sessions, lambda: NOW).sync_session(session)

    assert [
        item.modality for item in profiles.list_observations(profile.identifier)
    ] == [
        ObservationModality.RECALL,
        ObservationModality.TRANSFER,
        ObservationModality.RECALL,
    ]
    profiles.close()
    sessions.close()


def test_review_outcome_requires_citations_and_is_idempotent(tmp_path: Path) -> None:
    repository = SQLiteLearnerProfileRepository(tmp_path / "profiles.sqlite3")
    profile = repository.get_or_create_default(NOW)
    fingerprint = _fingerprint("doc-1")
    request = RecordReviewOutcomeRequest(
        profile.identifier,
        fingerprint,
        "objective-1",
        ObservationModality.TRANSFER,
        2,
        2,
        (CitationIdentity(DocumentId("doc-1"), "chunk-1", 1),),
        NOW,
    )
    use_case = RecordReviewOutcomeUseCase(repository)
    first = use_case.execute(request)
    second = use_case.execute(request)

    assert first.identifier == second.identifier
    assert len(repository.list_observations(profile.identifier)) == 1
    repository.close()


def test_review_use_case_rejects_forged_fingerprint_without_writing(
    tmp_path: Path,
) -> None:
    repository = SQLiteLearnerProfileRepository(tmp_path / "profiles.sqlite3")
    profile = repository.get_or_create_default(NOW)
    valid = _fingerprint("doc-1")
    forged = object.__new__(ConceptFingerprint)
    object.__setattr__(forged, "algorithm_version", valid.algorithm_version)
    object.__setattr__(forged, "value", "0" * 64)
    object.__setattr__(forged, "document_id", valid.document_id)
    object.__setattr__(forged, "normalized_title", valid.normalized_title)
    object.__setattr__(forged, "normalized_description", valid.normalized_description)
    request = RecordReviewOutcomeRequest(
        profile.identifier,
        forged,
        "objective-1",
        ObservationModality.RECALL,
        2,
        2,
        (CitationIdentity(DocumentId("doc-1"), "chunk-1", 1),),
        NOW,
    )

    with pytest.raises(ValueError, match="does not match"):
        RecordReviewOutcomeUseCase(repository).execute(request)
    assert repository.list_observations(profile.identifier) == ()
    repository.close()


def test_accepted_equivalence_pools_history_but_rejected_does_not() -> None:
    first = _fingerprint("doc-1", "Shared idea")
    second = _fingerprint("doc-2", "Shared idea")
    profile = LearnerProfile("profile", "Learner", None, NOW, NOW)
    observations = (
        _observation("profile", first),
        _observation("profile", second, suffix="2"),
    )
    scheduler = ReviewScheduler(clock=lambda: NOW)
    rejected = ConceptEquivalenceLink(
        first, second, EquivalenceDecision.REJECTED, NOW, profile.identifier
    )
    accepted = replace(rejected, decision=EquivalenceDecision.ACCEPTED)

    rejected_queue = scheduler.queue(profile, observations, (rejected,), NOW)
    accepted_queue = scheduler.queue(profile, observations, (accepted,), NOW)

    assert {item.observation_count for item in rejected_queue} == {1}
    assert {item.observation_count for item in accepted_queue} == {2}


def test_equivalence_state_is_profile_scoped_after_reopen_and_import(
    tmp_path: Path,
) -> None:
    repository = SQLiteLearnerProfileRepository(tmp_path / "profiles.sqlite3")
    profile_a = LearnerProfile("profile-a", "A", None, NOW, NOW)
    profile_b = LearnerProfile("profile-b", "B", None, NOW, NOW)
    repository.save_profile(profile_a)
    repository.save_profile(profile_b)
    first = _fingerprint("doc-1", "Shared idea")
    second = _fingerprint("doc-2", "Shared idea")
    candidate_a = ConceptEquivalenceCandidate(first, second, 0.8, NOW, "profile-a")
    candidate_b = replace(candidate_a, profile_id="profile-b")
    link_a = ConceptEquivalenceLink(
        first, second, EquivalenceDecision.ACCEPTED, NOW, "profile-a"
    )
    link_b = replace(
        link_a, decision=EquivalenceDecision.REJECTED, profile_id="profile-b"
    )
    repository.save_candidate(candidate_a)
    repository.save_candidate(candidate_b)
    repository.save_equivalence_link(link_a)
    repository.save_equivalence_link(link_b)
    exported_a = ExportLearnerProfileUseCase(repository).execute("profile-a")
    repository.close()

    repository = SQLiteLearnerProfileRepository(tmp_path / "profiles.sqlite3")
    assert repository.list_candidates("profile-a") == (candidate_a,)
    assert repository.list_candidates("profile-b") == (candidate_b,)
    assert repository.list_equivalence_links("profile-a") == (link_a,)
    assert repository.list_equivalence_links("profile-b") == (link_b,)

    repository.delete_profile("profile-a")
    ImportLearnerProfileUseCase(repository).execute(
        ImportLearnerProfileRequest("profile-a", exported_a)
    )
    assert repository.list_equivalence_links("profile-a") == (link_a,)
    assert repository.list_equivalence_links("profile-b") == (link_b,)
    repository.close()


def test_legacy_equivalence_collision_rehomes_only_valid_payload_owner(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "legacy-profiles.sqlite3"
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
        CREATE TABLE learner_profiles (
            profile_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            target_date TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE learner_observations (
            observation_id TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE TABLE equivalence_candidates (
            candidate_key TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE TABLE equivalence_links (
            link_key TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        """
    )
    profile_a = LearnerProfile("profile-a", "A", None, NOW, NOW)
    profile_b = LearnerProfile("profile-b", "B", None, NOW, NOW)
    for profile in (profile_a, profile_b):
        connection.execute(
            "INSERT INTO learner_profiles VALUES (?, ?, ?, ?, ?)",
            (
                profile.identifier,
                profile.display_name,
                None,
                profile.created_at.isoformat(),
                profile.updated_at.isoformat(),
            ),
        )
    first = _fingerprint("doc-1", "Shared idea")
    second = _fingerprint("doc-2", "Shared idea")
    candidate_b = ConceptEquivalenceCandidate(first, second, 0.8, NOW, "profile-b")
    link_b = ConceptEquivalenceLink(
        first, second, EquivalenceDecision.REJECTED, NOW, "profile-b"
    )
    pair_key = _pair_key(first, second)
    connection.execute(
        "INSERT INTO equivalence_candidates VALUES (?, ?, ?)",
        (pair_key, "profile-a", json.dumps(_candidate_payload(candidate_b))),
    )
    connection.execute(
        "INSERT INTO equivalence_links VALUES (?, ?, ?)",
        (pair_key, "profile-a", json.dumps(_link_payload(link_b))),
    )
    connection.commit()
    connection.close()

    repository = SQLiteLearnerProfileRepository(database_path)
    assert repository.list_candidates("profile-a") == ()
    assert repository.list_equivalence_links("profile-a") == ()
    assert repository.list_candidates("profile-b") == (candidate_b,)
    assert repository.list_equivalence_links("profile-b") == (link_b,)
    repository.close()


def test_profile_deletion_detaches_sessions_after_reopen(tmp_path: Path) -> None:
    session_path = tmp_path / "sessions.sqlite3"
    profile_path = tmp_path / "profiles.sqlite3"
    sessions = SQLiteStudySessionRepository(session_path)
    profiles = SQLiteLearnerProfileRepository(profile_path, sessions)
    profile = profiles.get_or_create_default(NOW)
    sessions.save(replace(_session(), learner_profile_id=profile.identifier))
    assert profiles.delete_profile(profile.identifier) == 1
    profiles.close()
    sessions.close()

    reopened_sessions = SQLiteStudySessionRepository(session_path)
    reopened_profiles = SQLiteLearnerProfileRepository(profile_path, reopened_sessions)
    assert reopened_profiles.get_profile(profile.identifier) is None
    assert reopened_profiles.list_observations(profile.identifier) == ()
    assert reopened_sessions.list()[0].learner_profile_id is None
    reopened_profiles.close()
    reopened_sessions.close()


def test_v3_session_reads_detached_and_next_save_writes_v4(tmp_path: Path) -> None:
    repository = SQLiteStudySessionRepository(tmp_path / "sessions.sqlite3")
    session = replace(_session(), learner_profile_id="legacy-profile")
    payload = _session_payload(session)
    payload["schema_version"] = 3
    with repository._connection:  # type: ignore[attr-defined]
        repository._connection.execute(  # type: ignore[attr-defined]
            "INSERT INTO study_sessions(session_id, document_id, payload, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (
                session.identifier,
                session.document_id.value,
                json.dumps(payload),
                session.updated_at.isoformat(),
            ),
        )

    restored = repository.get(session.identifier)
    assert restored is not None
    assert restored.learner_profile_id is None
    repository.save(restored)
    row = repository._connection.execute(  # type: ignore[attr-defined]
        "SELECT payload FROM study_sessions WHERE session_id = ?",
        (session.identifier,),
    ).fetchone()
    assert json.loads(row[0])["schema_version"] == 4
    repository.close()
