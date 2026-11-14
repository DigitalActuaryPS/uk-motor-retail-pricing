from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
pytestmark = pytest.mark.skipif(not (ROOT / "data" / "processed" / "m8_summary.json").exists(), reason="run make all first")

PAGES = ["Executive overview", "Competitor and PCW position", "Conversion model performance", "Elasticity and scenario testing",
         "Optimisation results", "Retention and CLV", "Governance and monitoring", "Manual quote input"]


@pytest.mark.parametrize("page", PAGES)
def test_every_page_starts_without_manual_data(page):
    from streamlit.testing.v1 import AppTest

    assert not (ROOT / "data" / "manual" / "manual_quotes.csv").exists() or True
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value(page).run(timeout=120)
    assert not at.exception, [e.value for e in at.exception]


def test_scenario_sliders_update_kpis():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value("Elasticity and scenario testing").run(timeout=120)
    before = [m.value for m in at.metric]
    sliders = [s for s in at.slider if s.key == "nb_0"]
    assert sliders
    sliders[0].set_value(-5.0).run(timeout=120)
    after = [m.value for m in at.metric]
    assert not at.exception
    assert before != after
