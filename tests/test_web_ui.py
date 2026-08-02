"""Composition checks for the unified Streamlit study interface."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from scholar_agent.presentation.web.app import format_due_label


def test_streamlit_navigation_is_unified_without_count_or_multi_pdf_controls() -> None:
    source = (
        Path(__file__).parents[1]
        / "src"
        / "scholar_agent"
        / "presentation"
        / "web"
        / "app.py"
    ).read_text(encoding="utf-8")

    assert '"Adaptive Tutor"' in source
    assert '"Ask Study Agent"' in source
    assert "st.slider" not in source
    assert "st.multiselect" not in source
    assert '"Compare"' not in source


def test_today_due_formatter_distinguishes_due_and_upcoming() -> None:
    as_of = datetime(2026, 8, 1, 12, tzinfo=UTC)
    assert format_due_label(as_of - timedelta(minutes=1), as_of) == "Due now"
    upcoming = format_due_label(as_of + timedelta(days=2, hours=3), as_of)
    assert upcoming.startswith("Upcoming: ")
    assert "Aug 3" in upcoming
