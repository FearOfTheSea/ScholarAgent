"""Tests for safe local server startup defaults."""

from typing import Any

from fastapi import FastAPI
from pytest import MonkeyPatch

from scholar_agent.presentation.api import main


def test_api_run_binds_to_loopback_only(monkeypatch: MonkeyPatch) -> None:
    """The project launcher never exposes the single-laptop API to the network."""
    captured: dict[str, Any] = {}

    def fake_run(application: FastAPI, **kwargs: object) -> None:
        captured["application"] = application
        captured.update(kwargs)

    monkeypatch.setattr(main.uvicorn, "run", fake_run)

    main.run()

    assert captured["application"] is main.app
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8000
