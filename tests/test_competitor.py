import numpy as np
import pandas as pd
import pytest

from retailpricing.competitor.analysis import add_position_metrics, flag_segments, run_analysis, recommendations
from retailpricing.core.io import nb_observational


def test_position_metrics_hand_calculation():
    df = pd.DataFrame({"our_premium": [550.0], "comp_median": [500.0], "price_gap_to_cheapest": [0.25],
                       "price_ratio_to_median": [1.10]})
    out = add_position_metrics(df)
    assert out["median_gap"].iloc[0] == pytest.approx(0.10)
    assert out["pci"].iloc[0] == pytest.approx(100 * 500 / 550)
    assert out["ln_ratio"].iloc[0] == pytest.approx(np.log(1.10))


def test_analysis_runs_and_flags(small, cfg):
    df, _ = small
    nb = add_position_metrics(nb_observational(df))
    res = run_analysis(nb, cfg)
    assert res["by_segment"]["position_flag"].isin(["overpriced", "underpriced", "in_line"]).all()
    assert res["by_rank"]["quotes"].sum() == len(nb)
    # conversion should fall with rank on a market with negative price and rank effects
    r = res["by_rank"].set_index("our_rank")["conversion"]
    assert r.loc[1] > r.loc[r.index.max()]
    text = recommendations(res)
    assert "Simulated data" in text


def test_flag_threshold(cfg):
    d = pd.DataFrame({"segment": ["a"] * 2 + ["b"] * 2, "ln_ratio": [0.10, 0.10, -0.10, -0.10]})
    seg = pd.DataFrame({"segment": ["a", "b"]})
    out = flag_segments(seg, d, cfg).set_index("segment")
    assert out.loc["a", "position_flag"] == "overpriced"
    assert out.loc["b", "position_flag"] == "underpriced"
