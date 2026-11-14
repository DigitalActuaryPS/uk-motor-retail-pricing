"""Renewal model features. Same leakage rules as the conversion models (see conversion.features)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..conversion.features import GBMSpec
from ..core.features import (AGE_BANDS, CHANNELS, REGIONS, RISK_BANDS, comp_columns, market_position, smooth_rank)


class RenewalDesign:
    """Design matrix for the renewal GLM. Price, prior premium, tenure and competitor prices can be overridden,
    which lets the CLV code score a future renewal for a customer won today."""

    def __init__(self, df: pd.DataFrame, k: int, smooth: float | None = None):
        self.k, self.smooth = k, smooth
        self.comp = df[comp_columns(k)].to_numpy()
        self.base_price = df["our_premium"].to_numpy(dtype=float)
        self.base_prior = df["prior_year_premium"].to_numpy(dtype=float) if "prior_year_premium" in df else None
        self.base_tenure = df["tenure"].to_numpy(dtype=float)
        ch = df["channel"].astype(str).to_numpy()
        age = df["age_band"].astype(str).to_numpy()
        reg = df["region"].astype(str).to_numpy()
        risk = df["pre_existing_risk_band"].astype(str).to_numpy()
        self.is_direct = (ch == "direct").astype(float)
        self.is_broker = (ch == "broker").astype(float)
        self.age_d = [(age == a).astype(float) for a in AGE_BANDS[1:]]
        self.reg_d = [(reg == r).astype(float) for r in REGIONS[1:]]
        self.risk_d = [(risk == r).astype(float) for r in RISK_BANDS[1:]]
        self.n = len(df)
        self.columns = (["const", "ch_direct", "ch_broker"] + [f"age_{a}" for a in AGE_BANDS[1:]]
                        + [f"reg_{r}" for r in REGIONS[1:]] + [f"risk_{r}" for r in RISK_BANDS[1:]]
                        + ["ln_tenure", "ln_change", "ln_ratio", "ln_ratio:direct", "ln_ratio:broker", "ln_rank"])

    def matrix(self, price=None, prior=None, tenure=None, comp=None) -> np.ndarray:
        price = self.base_price if price is None else np.asarray(price, dtype=float)
        prior = self.base_prior if prior is None else np.asarray(prior, dtype=float)
        tenure = self.base_tenure if tenure is None else np.asarray(tenure, dtype=float)
        comp = self.comp if comp is None else comp
        pos = market_position(price, comp)
        lr = np.log(pos["ratio"])
        lk = np.log(smooth_rank(price, comp, self.smooth)) if self.smooth else np.log(pos["rank"])
        cols = [np.ones(self.n), self.is_direct, self.is_broker, *self.age_d, *self.reg_d, *self.risk_d,
                np.log(np.maximum(tenure, 1.0)), np.log(price / prior), lr, lr * self.is_direct, lr * self.is_broker, lk]
        return np.column_stack(cols)


def renewal_frame(df: pd.DataFrame, k: int, price: np.ndarray | None = None) -> pd.DataFrame:
    comp = df[comp_columns(k)].to_numpy()
    price = df["our_premium"].to_numpy(dtype=float) if price is None else np.asarray(price, dtype=float)
    pos = market_position(price, comp)
    return pd.DataFrame(
        {
            "ln_change": np.log(price / df["prior_year_premium"].to_numpy(dtype=float)),
            "ln_ratio": np.log(pos["ratio"]),
            "rank": pos["rank"].astype(float),
            "tenure": df["tenure"].to_numpy(dtype=float),
            "channel": pd.Categorical(df["channel"].astype(str), categories=CHANNELS),
            "age_band": pd.Categorical(df["age_band"].astype(str), categories=AGE_BANDS),
            "region": pd.Categorical(df["region"].astype(str), categories=REGIONS),
            "risk_band": pd.Categorical(df["pre_existing_risk_band"].astype(str), categories=RISK_BANDS),
        },
        index=df.index,
    )


REN_FEATURES = ["ln_change", "ln_ratio", "rank", "tenure", "channel", "age_band", "region", "risk_band"]
REN_SPEC = GBMSpec("renewal", REN_FEATURES, ["channel", "age_band", "region", "risk_band"],
                   {"ln_change": -1, "ln_ratio": -1, "rank": -1, "tenure": 1}, renewal_frame)
