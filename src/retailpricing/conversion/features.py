"""Feature construction for the conversion models.

Leakage controls (kept in code on purpose):
  * Only quote-time information is used: our premium, competitor prices, channel, segment inputs, month.
  * Outcome-derived columns are never features: converted, written_premium, unit_cost,
    acquisition_cost, fixed_cost_contribution, renewed_flag, renewal_outcome.
  * Identifiers (quote_id, customer_id), treatment_group and the hidden truth file are never features.
  * Absolute premium levels are excluded. Only relative market position enters, so premium inflation
    over time does not leak a time trend into out-of-time scoring.
  * Splits are by quote date, never random. The randomised price test rows are excluded from fitting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.features import AGE_BANDS, CHANNELS, REGIONS, RISK_BANDS, comp_columns, market_position, smooth_rank, smooth_top1

FORBIDDEN_FEATURES = {
    "converted", "written_premium", "unit_cost", "acquisition_cost", "fixed_cost_contribution",
    "renewed_flag", "renewal_outcome", "quote_id", "customer_id", "treatment_group", "quote_date",
    "latent_intent", "true_probability",
}

GBM_FEATURES = ["ln_ratio", "ratio", "rank", "cheapest_gap", "ln_comp_spread", "channel", "age_band",
                "region", "risk_band", "month"]
GBM_CATEGORICAL = ["channel", "age_band", "region", "risk_band"]
GBM_MONOTONE = {"ln_ratio": -1, "ratio": -1, "rank": -1, "cheapest_gap": -1}


class GLMDesign:
    """Design matrix for the new business GLM. Price-dependent columns are recomputed for any price vector."""

    def __init__(self, df: pd.DataFrame, k: int, smooth: float | None = None):
        self.k = k
        self.smooth = smooth  # only the optimiser sets this, to make rank differentiable
        self.comp = df[comp_columns(k)].to_numpy()
        self.base_price = df["our_premium"].to_numpy(dtype=float)
        ch = df["channel"].astype(str).to_numpy()
        age = df["age_band"].astype(str).to_numpy()
        reg = df["region"].astype(str).to_numpy()
        risk = df["pre_existing_risk_band"].astype(str).to_numpy()
        month = pd.DatetimeIndex(df["quote_date"]).month.to_numpy()
        self.is_direct = (ch == "direct").astype(float)
        self.is_broker = (ch == "broker").astype(float)
        self.age_d = [(age == a).astype(float) for a in AGE_BANDS[1:]]
        self.reg_d = [(reg == r).astype(float) for r in REGIONS[1:]]
        self.risk_d = [(risk == r).astype(float) for r in RISK_BANDS[1:]]
        self.sin_m = np.sin(2 * np.pi * month / 12.0)
        self.cos_m = np.cos(2 * np.pi * month / 12.0)
        self.n = len(df)
        self.columns = (
            ["const", "ch_direct", "ch_broker"] + [f"age_{a}" for a in AGE_BANDS[1:]]
            + [f"reg_{r}" for r in REGIONS[1:]] + [f"risk_{r}" for r in RISK_BANDS[1:]]
            + ["sin_month", "cos_month", "ln_ratio", "ln_ratio:direct", "ln_ratio:broker"]
            + [f"ln_ratio:age_{a}" for a in AGE_BANDS[1:]]
            + ["ln_rank", "ln_rank:direct", "ln_rank:broker", "top1", "top1:direct", "top1:broker"]
        )

    def matrix(self, price: np.ndarray | None = None) -> np.ndarray:
        price = self.base_price if price is None else np.asarray(price, dtype=float)
        pos = market_position(price, self.comp)
        lr = np.log(pos["ratio"])
        if self.smooth:
            lk = np.log(smooth_rank(price, self.comp, self.smooth))
            top1 = smooth_top1(price, self.comp, self.smooth)
        else:
            lk = np.log(pos["rank"])
            top1 = (pos["rank"] == 1).astype(float)
        cols = [np.ones(self.n), self.is_direct, self.is_broker, *self.age_d, *self.reg_d, *self.risk_d,
                self.sin_m, self.cos_m, lr, lr * self.is_direct, lr * self.is_broker,
                *[lr * a for a in self.age_d], lk, lk * self.is_direct, lk * self.is_broker,
                top1, top1 * self.is_direct, top1 * self.is_broker]
        return np.column_stack(cols)


def gbm_frame(df: pd.DataFrame, k: int, price: np.ndarray | None = None) -> pd.DataFrame:
    comp = df[comp_columns(k)].to_numpy()
    price = df["our_premium"].to_numpy(dtype=float) if price is None else np.asarray(price, dtype=float)
    pos = market_position(price, comp)
    out = pd.DataFrame(
        {
            "ln_ratio": np.log(pos["ratio"]),
            "ratio": pos["ratio"],
            "rank": pos["rank"].astype(float),
            "cheapest_gap": pos["gap_cheapest"],
            "ln_comp_spread": np.log(np.median(comp, axis=1) / comp[:, 0]),
            "channel": pd.Categorical(df["channel"].astype(str), categories=CHANNELS),
            "age_band": pd.Categorical(df["age_band"].astype(str), categories=AGE_BANDS),
            "region": pd.Categorical(df["region"].astype(str), categories=REGIONS),
            "risk_band": pd.Categorical(df["pre_existing_risk_band"].astype(str), categories=RISK_BANDS),
            "month": pd.DatetimeIndex(df["quote_date"]).month.to_numpy(),
        },
        index=df.index,
    )
    assert not (set(out.columns) & FORBIDDEN_FEATURES), "leakage guard"
    return out[GBM_FEATURES]


from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class GBMSpec:
    name: str
    features: list
    categorical: list
    monotone: dict
    frame_fn: Callable


NB_SPEC = GBMSpec("nb", GBM_FEATURES, GBM_CATEGORICAL, GBM_MONOTONE, gbm_frame)
