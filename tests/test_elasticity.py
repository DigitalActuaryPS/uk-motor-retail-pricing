import numpy as np
import pandas as pd
import pytest

from retailpricing.conversion.features import GLMDesign
from retailpricing.conversion.models import fit_glm
from retailpricing.core.features import unit_economics
from retailpricing.core.io import nb_experiment, nb_observational, time_split
from retailpricing.data_gen.truth import true_probability
from retailpricing.elasticity.core import (ShiftedGLM, arc_elasticity, arm_table, calibrate_shifts,
                                           endogeneity_stats, experiment_elasticities, scenario_table)


def test_arc_elasticity_constant_elasticity_hand_case():
    # p = 0.1 * (P / 100) ** -2 has elasticity exactly -2 for any step
    price = np.array([100.0, 150.0, 80.0])
    fn = lambda pr: 0.1 * (pr / 100.0) ** -2
    assert arc_elasticity(fn, price, None, 0.01) == pytest.approx(-2.0, abs=1e-9)
    assert arc_elasticity(fn, price, None, 0.05) == pytest.approx(-2.0, abs=1e-9)


def test_scenario_arithmetic_hand_case(cfg):
    df = pd.DataFrame({"our_premium": [500.0, 700.0], "age_band": ["35-49", "35-49"], "region": ["Midlands_East"] * 2,
                       "pre_existing_risk_band": ["medium"] * 2, "channel": ["PCW", "direct"], "business_type": ["new_business"] * 2})
    econ = unit_economics(cfg, df)
    cost = econ["unit_cost"] + econ["expense"] + econ["acquisition"]
    fn = lambda pr: np.array([0.10, 0.20])
    t = scenario_table(fn, df, cfg, [0.0, 0.05]).set_index("price_change")
    assert t.loc[0.0, "expected_policies"] == pytest.approx(0.30)
    assert t.loc[0.0, "gwp"] == pytest.approx(0.1 * 500 + 0.2 * 700)
    assert t.loc[0.05, "gwp"] == pytest.approx(0.1 * 525 + 0.2 * 735)
    assert t.loc[0.05, "contribution"] == pytest.approx(0.1 * (525 - cost[0]) + 0.2 * (735 - cost[1]))
    assert t.loc[0.05, "average_premium"] == pytest.approx((0.1 * 525 + 0.2 * 735) / 0.3)
    assert t.loc[0.0, "margin"] == pytest.approx(t.loc[0.0, "contribution"] / t.loc[0.0, "gwp"])


def test_shifted_glm_baseline_unchanged_and_hits_target(small, cfg):
    k = cfg["market.n_competitors"]
    nb = nb_observational(small[0])
    train, _, test = time_split(nb, cfg)
    glm = fit_glm(train, k)
    d = GLMDesign(test, k)
    p0 = test["our_premium"].to_numpy()
    delta = np.where(test["channel"].astype(str) == "PCW", -1.0, 0.0)
    sh = ShiftedGLM(glm, d, delta, p0)
    assert np.allclose(sh.predict(p0), glm.predict(d, p0))
    shifts = calibrate_shifts(glm, d, test, p0, {"PCW": -8.0}, 0.01)
    dr = test["channel"].astype(str).map(shifts).fillna(0.0).to_numpy()
    mask = (test["channel"].astype(str) == "PCW").to_numpy()
    assert arc_elasticity(ShiftedGLM(glm, d, dr, p0).predict, p0, mask, 0.01) == pytest.approx(-8.0, abs=1e-3)


def test_randomised_test_recovers_truth_better_than_observational(cfg):
    from retailpricing.data_gen.generator import generate

    df, truth = generate(cfg, seed=3, n_quotes=60_000)
    exp = nb_experiment(df)
    ex = experiment_elasticities(exp, 20, 1).set_index("group")
    u = truth.set_index("quote_id").loc[exp["quote_id"], "latent_intent"].to_numpy()
    fn = lambda pr: true_probability(exp, u, cfg, pr)
    true_e = arc_elasticity(fn, exp["reference_premium"].to_numpy(), None, 0.01)
    assert ex.loc["overall", "experiment"] < 0
    assert abs(ex.loc["overall", "experiment"] - true_e) < 2.0
    arms = arm_table(exp)
    assert set(arms["treatment_group"]) == {"control", "T-5", "T-2", "T+2", "T+5"}


def test_endogeneity_visible(small):
    df, truth = small
    nb = nb_observational(df)
    u = truth.set_index("quote_id").loc[nb["quote_id"], "latent_intent"].to_numpy()
    s = endogeneity_stats(nb, u)
    assert s["corr_ln_ratio_intent"] > 0.1
