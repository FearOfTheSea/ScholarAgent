"""API contracts for additive Phase 2 profile and review routes."""

import asyncio
from pathlib import Path

from httpx import ASGITransport, AsyncClient

from scholar_agent.config.settings import Settings
from scholar_agent.domain.entities.learner_profile import ConceptFingerprint
from scholar_agent.domain.value_objects.document_id import DocumentId
from scholar_agent.presentation.api.main import create_app


def test_profile_routes_are_in_openapi_and_round_trip(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path
        app = create_app(
            Settings(
                catalog_db_path=root / "catalog.sqlite3",
                learner_profile_db_path=root / "profiles.sqlite3",
                document_library_path=root / "documents",
                vector_db_path=root / "vectors",
            )
        )
        paths = app.openapi()["paths"]
        assert "/learner-profiles" in paths
        assert "/learner-profiles/{profile_id}" in paths
        assert "/learner-profiles/{profile_id}/export" in paths
        assert "/learner-profiles/{profile_id}/import" in paths
        assert "/learner-profiles/{profile_id}/review-queue" in paths
        assert "/learner-profiles/{profile_id}/review-outcomes" in paths
        assert "/learner-profiles/{profile_id}/equivalence-candidates" in paths
        assert "/learner-profiles/{profile_id}/equivalence-decisions" in paths
        assert "/learner-profiles/{profile_id}/review-missions" in paths

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            created = await client.post(
                "/learner-profiles", json={"display_name": "API learner"}
            )
            assert created.status_code == 201
            profile_id = created.json()["profile_id"]
            profile_response = await client.get(f"/learner-profiles/{profile_id}")
            assert profile_response.status_code == 200
            assert (
                await client.get(f"/learner-profiles/{profile_id}/review-queue")
            ).json() == []
            exported = await client.get(f"/learner-profiles/{profile_id}/export")
            assert exported.status_code == 200
            assert "response" not in exported.text
            assert (
                await client.post(
                    f"/learner-profiles/{profile_id}/import",
                    json={"payload": exported.json()},
                )
            ).status_code == 409
            deleted = await client.delete(f"/learner-profiles/{profile_id}")
            assert deleted.status_code == 200
            assert deleted.json()["detached_session_count"] == 0
            restored = await client.post(
                f"/learner-profiles/{profile_id}/import",
                json={"payload": exported.json()},
            )
            assert restored.status_code == 200

    asyncio.run(scenario())


def test_review_outcome_api_rejects_forged_or_unsupported_fingerprint(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        app = create_app(
            Settings(
                catalog_db_path=tmp_path / "catalog.sqlite3",
                learner_profile_db_path=tmp_path / "profiles.sqlite3",
                document_library_path=tmp_path / "documents",
                vector_db_path=tmp_path / "vectors",
            )
        )
        fingerprint = ConceptFingerprint.from_descriptor(
            DocumentId("document-1"), "Memory", "A durable learning concept."
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            created = await client.post(
                "/learner-profiles", json={"display_name": "API learner"}
            )
            profile_id = created.json()["profile_id"]
            base = {
                "algorithm_version": fingerprint.algorithm_version,
                "document_id": fingerprint.document_id.value,
                "normalized_title": fingerprint.normalized_title,
                "normalized_description": fingerprint.normalized_description,
            }
            forged = {**base, "fingerprint": "0" * 64}
            payload = {
                "fingerprint": forged,
                "objective_id": "objective-1",
                "modality": "recall",
                "score": 2,
                "difficulty": 2,
                "citations": [
                    {
                        "document_id": "document-1",
                        "chunk_id": "chunk-1",
                        "page_number": 1,
                    }
                ],
            }
            response = await client.post(
                f"/learner-profiles/{profile_id}/review-outcomes", json=payload
            )
            assert response.status_code == 400

            unsupported = {
                **base,
                "algorithm_version": "future-v2",
                "fingerprint": fingerprint.value,
            }
            response = await client.post(
                f"/learner-profiles/{profile_id}/review-outcomes",
                json={**payload, "fingerprint": unsupported},
            )
            assert response.status_code == 400

        assert (
            app.state.container.learner_profile_repository().list_observations(
                profile_id
            )
            == ()
        )

    asyncio.run(scenario())
