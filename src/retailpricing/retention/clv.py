"""Customer lifetime value for a new business policy. Simulated data only.

CLV = -acquisition + sum_{t=1..H} d^(t-1) * S_t * (P_t - cost_t),  S_1 = 1,  S_{t+1} = S_t * r_t
r_t is the modelled probability of renewing at the end of year t. Year 1 is not discounted.
Cost per policy is the fixed value from configuration, grown at a configured rate. It is never fitted.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.special import expit

from ..core.config import Config
from ..core.features import unit_economics
from .features import RenewalDesign

RetentionFn = Callable[[np.ndarray, np.ndarray, int, np.ndarray], np.ndarray]


def clv_per_policy(price: np.ndarray, cost: np.ndarray, acq: np.ndarray, comp: np.ndarray, retention_fn: RetentionFn,
                   horizon: int, discount_rate: float, premium_growth: float, cost_growth: float, market_growth: float,
                   retention_shift: float = 0.0, return_detail: bool = False):
    """Per-policy CLV. `cost` is unit cost plus expense per year at year 1 prices (excludes acquisition)."""
    d = 1.0 / (1.0 + discount_rate)
    total = -acq + (price - cost)
    surv = np.ones_like(price, dtype=float)
    detail = {"survival": [surv.copy()], "ret": []}
    for t in range(1, horizon):
        prior_t = price * (1 + premium_growth) ** (t - 1)
        price_t = price * (1 + premium_growth) ** t
        comp_t = comp * (1 + market_growth) ** t
        r = np.clip(retention_fn(price_t, prior_t, t, comp_t) + retention_shift, 0.0, 1.0)
        surv = surv * r
        total = total + d**t * surv * (price_t - cost * (1 + cost_growth) ** t)
        detail["survival"].append(surv.copy())
        detail["ret"].append(r)
    return (total, detail) if return_detail else total


def make_retention_fn(renewal_params: np.ndarray, design: RenewalDesign) -> RetentionFn:
    def fn(price_t, prior_t, t, comp_t):
        tenure = np.full(design.n, float(t))
        return expit(design.matrix(price=price_t, prior=prior_t, tenure=tenure, comp=comp_t) @ renewal_params)
    return fn


def clv_settings(cfg: Config, **over) -> dict:
    s = {"horizon": int(cfg["clv.horizon_years"]), "discount_rate": cfg["clv.discount_rate"],
         "premium_growth": cfg["clv.renewal_premium_growth"], "cost_growth": cfg["clv.unit_cost_growth"],
         "market_growth": cfg["market.annual_premium_trend"], "retention_shift": 0.0, "cost_shift": 0.0}
    s.update(over)
    return s


def nb_clv(nb: pd.DataFrame, price: np.ndarray, retention_fn: RetentionFn, cfg: Config, return_detail: bool = False, **over):
    from ..core.features import comp_columns

    s = clv_settings(cfg, **over)
    econ = unit_economics(cfg, nb)
    cost = (econ["unit_cost"] + econ["expense"]) * (1 + s["cost_shift"])
    comp = nb[comp_columns(cfg["market.n_competitors"])].to_numpy()
    return clv_per_policy(price, cost, econ["acquisition"], comp, retention_fn, s["horizon"], s["discount_rate"],
                          s["premium_growth"], s["cost_growth"], s["market_growth"], s["retention_shift"], return_detail)


def make_clv_objective(book, retention_params: np.ndarray, cfg: Config, **over):
    """Objective for the M5 solver: NB policies valued at CLV plus one-year renewal contribution, per segment."""
    design = RenewalDesign(book.nb, cfg["market.n_competitors"], smooth=cfg["optimisation.rank_smoothing_width"])
    rfn = make_retention_fn(retention_params, design)

    def objective(book_, x, y, ev) -> tuple:
        clv = nb_clv(book_.nb, ev["nb_price"], rfn, cfg, **over)
        nb_val = np.bincount(book_.nb_seg, weights=ev["nb_prob"] * clv, minlength=book_.S)
        return nb_val, ev["ren_contrib_seg"]

    return objective, rfn
