import numpy as np
import pytest

from retailpricing.core.io import renewals, time_split
from retailpricing.retention.clv import clv_per_policy
from retailpricing.retention.features import RenewalDesign
from retailpricing.retention.models import fit_renewal_glm


def const_ret(r):
    return lambda price_t, prior_t, t, comp_t: np.full(len(price_t), r)


def test_clv_hand_calculation_three_years():
    # -60 + (500 - 450) + 0.8*50/1.05 + 0.64*50/1.05^2
    price, cost, acq, comp = np.array([500.0]), np.array([450.0]), np.array([60.0]), np.array([[400.0]])
    got = clv_per_policy(price, cost, acq, comp, const_ret(0.8), horizon=3, discount_rate=0.05,
                         premium_growth=0.0, cost_growth=0.0, market_growth=0.0)
    expected = -60 + 50 + 0.8 * 50 / 1.05 + 0.64 * 50 / 1.05**2
    assert got[0] == pytest.approx(expected)
    assert expected == pytest.approx(57.1202, abs=1e-3)


def test_clv_with_growth_hand_calculation():
    price, cost, acq, comp = np.array([100.0]), np.array([80.0]), np.array([10.0]), np.array([[90.0]])
    got = clv_per_policy(price, cost, acq, comp, const_ret(0.5), horizon=2, discount_rate=0.10,
                         premium_growth=0.10, cost_growth=0.05, market_growth=0.0)
    expected = -10 + (100 - 80) + (1 / 1.10) * 0.5 * (110 - 80 * 1.05)
    assert got[0] == pytest.approx(expected)


def test_clv_horizon_one_is_first_year_contribution():
    got = clv_per_policy(np.array([300.0]), np.array([250.0]), np.array([20.0]), np.array([[280.0]]), const_ret(0.9),
                         horizon=1, discount_rate=0.05, premium_growth=0.03, cost_growth=0.02, market_growth=0.03)
    assert got[0] == pytest.approx(30.0)


def test_clv_retention_shift_and_discount_monotonic():
    args = dict(price=np.array([500.0]), cost=np.array([420.0]), acq=np.array([50.0]), comp=np.array([[450.0]]), horizon=5,
                premium_growth=0.05, cost_growth=0.03, market_growth=0.05)
    lo = clv_per_policy(retention_fn=const_ret(0.7), discount_rate=0.05, **args)[0]
    hi = clv_per_policy(retention_fn=const_ret(0.8), discount_rate=0.05, **args)[0]
    hd = clv_per_policy(retention_fn=const_ret(0.7), discount_rate=0.10, **args)[0]
    assert hi > lo > hd
    sh = clv_per_policy(retention_fn=const_ret(0.7), discount_rate=0.05, retention_shift=0.1, **args)[0]
    assert sh == pytest.approx(hi)


def test_renewal_glm_price_change_sign_and_design(small, cfg):
    k = cfg["market.n_competitors"]
    ren = renewals(small[0])
    train, _, test = time_split(ren, cfg)
    glm = fit_renewal_glm(train, k)
    c = dict(zip(glm.columns, glm.params))
    assert c["ln_change"] < 0
    d = RenewalDesign(test, k)
    X = d.matrix()
    j = d.columns.index("ln_change")
    assert np.allclose(X[:, j], np.log(test["our_premium"] / test["prior_year_premium"]))
