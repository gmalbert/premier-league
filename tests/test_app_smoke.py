from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_overview_starts_from_strict_artifacts(monkeypatch):
    monkeypatch.chdir(ROOT)
    app = AppTest.from_file(ROOT / "predictions.py", default_timeout=45).run()

    assert not app.exception
    assert [title.value for title in app.title] == ["English Premier League - Overview"]
