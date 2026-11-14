"""Planning book for the optimiser: test-period quotes, response models and aggregate evaluation.

Decision variables: x_s (new business price change) and y_s (renewal price change) for each of the 15 segments.
Simulated data only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..core.config import Config
from ..core.features import market_position, smooth_rank, unit_economics

Resp = Callable[[np.ndarray], np.ndarray]


@dataclass
class PlanningBook:
    cfg: Config
    nb: pd.DataFrame
    ren: pd.DataFrame
    segments: list = field(default_factory=list)

    def __post_init__(self):
        self.segments = sorted(self.nb["segment"].astype(str).unique())
        idx = {s: i for i, s in enumerate(self.segments)}
        self.S = len(self.segments)
        self.nb_seg = self.nb["segment"].astype(str).map(idx).to_numpy()
        self.ren_seg = self.ren["segment"].astype(str).map(idx).to_numpy()
        self.nb_p0 = self.nb["our_premium"].to_numpy(dtype=float)
        self.ren_p0 = self.ren["our_premium"].to_numpy(dtype=float)
        self.ren_prior = self.ren["prior_year_premium"].to_numpy(dtype=float)
        e_nb, e_rn = unit_economics(self.cfg, self.nb), unit_economics(self.cfg, self.ren)
        self.nb_cost = e_nb["unit_cost"] + e_nb["expense"] + e_nb["acquisition"]
        self.ren_cost = e_rn["unit_cost"] + e_rn["expense"] + e_rn["acquisition"]
        k = self.cfg["market.n_competitors"]
        from ..core.features import comp_columns

        self.nb_comp = self.nb[comp_columns(k)].to_numpy()
        self.ren_comp = self.ren[comp_columns(k)].to_numpy()
        self.width = self.cfg["optimisation.rank_smoothing_width"]
        self.nb_count = np.bincount(self.nb_seg, minlength=self.S).astype(float)
        self.ren_count = np.bincount(self.ren_seg, minlength=self.S).astype(float)

    def _rank(self, price, comp, smooth: bool):
        return smooth_rank(price, comp, self.width) if smooth else market_position(price, comp)["rank"].astype(float)

    def evaluate(self, x: np.ndarray, y: np.ndarray, nb_resp: Resp, ren_resp: Resp, smooth: bool = True) -> dict:
        S = self.S
        pn = self.nb_p0 * (1 + x[self.nb_seg])
        pr = self.ren_p0 * (1 + y[self.ren_seg])
        qn, qr = nb_resp(pn), ren_resp(pr)
        seg = lambda idx, w: np.bincount(idx, weights=w, minlength=S)
        rk_nb = seg(self.nb_seg, self._rank(pn, self.nb_comp, smooth)) / self.nb_count
        rk_rn = seg(self.ren_seg, self._rank(pr, self.ren_comp, smooth)) / self.ren_count
        ev = {
            "nb_prob": qn, "nb_price": pn,
            "nb_policies_seg": seg(self.nb_seg, qn), "nb_gwp_seg": seg(self.nb_seg, qn * pn),
            "nb_contrib_seg": seg(self.nb_seg, qn * (pn - self.nb_cost)),
            "ren_policies_seg": seg(self.ren_seg, qr), "ren_gwp_seg": seg(self.ren_seg, qr * pr),
            "ren_contrib_seg": seg(self.ren_seg, qr * (pr - self.ren_cost)),
            "nb_rank_seg": rk_nb, "ren_rank_seg": rk_rn,
            "ren_change_seg": seg(self.ren_seg, pr / self.ren_prior - 1.0) / self.ren_count,
        }
        for k_ in ["nb_policies", "nb_gwp", "nb_contrib", "ren_policies", "ren_gwp", "ren_contrib"]:
            ev[k_] = float(ev[k_ + "_seg"].sum())
        ev["gwp"] = ev["nb_gwp"] + ev["ren_gwp"]
        ev["contrib"] = ev["nb_contrib"] + ev["ren_contrib"]
        ev["margin"] = ev["contrib"] / ev["gwp"]
        ev["policies"] = ev["nb_policies"] + ev["ren_policies"]
        return ev

    def constraints(self, ev: dict, base: dict, offset: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
        """All entries must be >= 0. Scaled to be of order one."""
        c = self.cfg
        vals, names = [], []
        vals.append((ev["margin"] - c["optimisation.min_portfolio_margin"]) * 10.0); names.append("portfolio_margin_floor")
        vals.append(ev["nb_policies"] / base["nb_policies"] - c["optimisation.min_volume_ratio"]); names.append("nb_volume_floor")
        vals.append(ev["ren_policies"] / base["ren_policies"] - c["optimisation.min_renewal_volume_ratio"]); names.append("renewal_volume_floor")
        d = c["optimisation.max_rank_deterioration"]
        for i, s in enumerate(self.segments):
            vals.append(d - (ev["nb_rank_seg"][i] - base["nb_rank_seg"][i])); names.append(f"nb_rank_cap:{s}")
        for i, s in enumerate(self.segments):
            vals.append(d - (ev["ren_rank_seg"][i] - base["ren_rank_seg"][i])); names.append(f"renewal_rank_cap:{s}")
        cap = c["optimisation.renewal_price_change_cap"]
        for i, s in enumerate(self.segments):
            vals.append((cap - ev["ren_change_seg"][i]) * 10.0); names.append(f"renewal_change_cap:{s}")
        vals = np.array(vals)
        if offset is not None:
            vals = vals - offset
        return vals, names
