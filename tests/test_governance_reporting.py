import json
from pathlib import Path

import numpy as np
import pytest

from retailpricing.governance.monitoring import psi, rag

ROOT = Path(__file__).resolve().parents[1]


def test_psi_identical_is_zero():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x.copy()) == pytest.approx(0.0, abs=1e-9)


def test_psi_categorical_hand_calculation():
    e = np.array(["a"] * 50 + ["b"] * 50)
    a = np.array(["a"] * 70 + ["b"] * 30)
    expected = (0.7 - 0.5) * np.log(0.7 / 0.5) + (0.3 - 0.5) * np.log(0.3 / 0.5)
    assert psi(e, a, categorical=True) == pytest.approx(expected)


def test_psi_detects_shift_and_rag():
    rng = np.random.default_rng(1)
    base = rng.normal(size=20000)
    shifted = rng.normal(loc=1.0, size=20000)
    p = psi(base, shifted)
    assert p > 0.25 and rag(p, 0.10, 0.25) == "red"
    assert rag(0.05, 0.10, 0.25) == "green" and rag(0.10, 0.10, 0.25) == "amber"


needs_outputs = pytest.mark.skipif(not (ROOT / "data" / "processed" / "m8_summary.json").exists(), reason="run make all first")


@needs_outputs
def test_stress_outputs_consistent():
    import pandas as pd

    t = pd.read_csv(ROOT / "data" / "processed" / "m8_stress_tests__simulated_data.csv")
    s = json.loads((ROOT / "data" / "processed" / "m8_summary.json").read_text())
    assert len(t) == s["stress_scenarios"]
    assert s["stress_min_gain"] == pytest.approx(t["gain"].min())
    assert s["stress_baseline_below_margin_floor"] == int((~t["baseline_margin_floor_met"]).sum())


@needs_outputs
def test_reports_match_current_outputs():
    from retailpricing.reporting import build

    v = build.collect(build.Registry())
    assert (ROOT / "docs" / "executive_summary.md").read_text(encoding="utf-8") == build.render_exec(v)
    assert (ROOT / "README.md").read_text(encoding="utf-8") == build.render_readme(v)
