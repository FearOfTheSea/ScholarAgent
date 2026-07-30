"""Composition checks for the unified Streamlit study interface."""

from pathlib import Path


def test_streamlit_navigation_is_unified_without_count_or_multi_pdf_controls() -> None:
    source = (
        Path(__file__).parents[1]
        / "src"
        / "scholar_agent"
        / "presentation"
        / "web"
        / "app.py"
    ).read_text(encoding="utf-8")

    assert '("Library", "Ask Study Agent")' in source
    assert "st.slider" not in source
    assert "st.multiselect" not in source
    assert '"Compare"' not in source
