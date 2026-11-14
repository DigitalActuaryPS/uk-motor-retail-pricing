"""Shared market-position arithmetic and fixed cost lookups.

Used by the data generator (truth), the models and the optimiser so that rank,
ratio and gap are defined once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

AGE_BANDS = ["17-24", "25-34", "35-49", "50-64", "65+"]
REGIONS = ["London_SE", "Midlands_East", "North_West_Rest"]
RISK_BANDS = ["low", "medium", "high"]
CHANNELS = ["PCW", "direct", "broker"]
NB = "new_business"
REN = "renewal"


def comp_columns(k: int) -> list[str]:
    return [f"comp_p{j}" for j in range(1, k + 1)]


def market_position(price: np.ndarray, comp_sorted: np.ndarray) -> dict[str, np.ndarray]:
    """Rank (1 + number of cheaper competitors), ratio to median and gap to cheapest."""
    median = np.median(comp_sorted, axis=1)
    rank = 1 + (comp_sorted < price[:, None]).sum(axis=1)
    return {
        "rank": rank.astype(int),
        "ratio": price / median,
        "gap_cheapest": price / comp_sorted[:, 0] - 1.0,
    }


def smooth_rank(price: np.ndarray, comp_sorted: np.ndarray, width: float) -> np.ndarray:
    """Differentiable rank used only inside the optimiser: 1 + sum of logistic steps."""
    z = (price[:, None] - comp_sorted) / (comp_sorted * width)
    return 1.0 + (1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))).sum(axis=1)


def smooth_top1(price: np.ndarray, comp_sorted: np.ndarray, width: float) -> np.ndarray:
    """Differentiable probability that we are cheapest: product of logistic steps over competitors."""
    z = (comp_sorted - price[:, None]) / (comp_sorted * width)
    return np.prod(1.0 / (1.0 + np.exp(-np.clip(z, -50, 50))), axis=1)


def _normalised(rel: dict, mix: dict) -> dict:
    mean = sum(rel[k] * mix[k] for k in rel)
    return {k: v / mean for k, v in rel.items()}


def premium_relativities(cfg: Config) -> dict[str, dict]:
    return {
        "age": _normalised(cfg["market.age_band_premium_relativity"], cfg["volumes.age_band_mix"]),
        "region": _normalised(cfg["market.region_premium_relativity"], cfg["volumes.region_mix"]),
        "risk": _normalised(cfg["market.risk_band_premium_relativity"], cfg["volumes.risk_band_mix"]),
    }


def unit_cost(cfg: Config, age: pd.Series, region: pd.Series, risk: pd.Series) -> np.ndarray:
    """Fixed cost per policy read from configuration. Never fitted."""
    a = _normalised(cfg["costs.age_band_cost_relativity"], cfg["volumes.age_band_mix"])
    r = _normalised(cfg["costs.region_cost_relativity"], cfg["volumes.region_mix"])
    k = _normalised(cfg["costs.risk_band_cost_relativity"], cfg["volumes.risk_band_mix"])
    base = cfg["costs.unit_cost_base"]
    return base * age.map(a).to_numpy() * region.map(r).to_numpy() * risk.map(k).to_numpy()


def acquisition_cost(cfg: Config, channel: pd.Series, business_type: pd.Series) -> np.ndarray:
    m = channel.map(cfg["costs.acquisition_cost_by_channel"]).to_numpy(dtype=float)
    return np.where(business_type.to_numpy() == NB, m, 0.0)


def unit_economics(cfg: Config, df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Fixed per-policy economics from configuration: cost, expense and acquisition cost."""
    uc = unit_cost(cfg, df["age_band"].astype(str), df["region"].astype(str), df["pre_existing_risk_band"].astype(str))
    exp = np.full(len(df), cfg["costs.expense_per_policy"], dtype=float)
    acq = acquisition_cost(cfg, df["channel"].astype(str), df["business_type"].astype(str))
    return {"unit_cost": uc, "expense": exp, "acquisition": acq}
