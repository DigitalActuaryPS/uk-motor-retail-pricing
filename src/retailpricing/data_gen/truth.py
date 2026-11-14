"""True conversion and retention probabilities of the simulated market.

These functions embody the data generating process. They are used by the generator
and, afterwards, only to validate estimates against the known truth. They are never
used as model features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.config import Config
from ..core.features import NB, market_position, comp_columns


def _season(cfg: Config, month: np.ndarray) -> np.ndarray:
    amp = cfg["conversion_response.seasonality_amplitude"]
    peak = cfg["conversion_response.seasonality_peak_month"]
    return amp * np.cos(2 * np.pi * (month - peak) / 12.0)


def true_probability(df: pd.DataFrame, u: np.ndarray, cfg: Config, price: np.ndarray | None = None) -> np.ndarray:
    """Probability of conversion (NB) or renewal (renewal rows) at `price`."""
    k = cfg["market.n_competitors"]
    comp = df[comp_columns(k)].to_numpy()
    price = df["our_premium"].to_numpy() if price is None else np.asarray(price, dtype=float)
    pos = market_position(price, comp)
    ln_r = np.log(pos["ratio"])
    rank = pos["rank"]
    ch = df["channel"].astype(str)
    age = df["age_band"].astype(str)
    reg = df["region"].astype(str)
    risk = df["pre_existing_risk_band"].astype(str)
    month = pd.DatetimeIndex(df["quote_date"]).month.to_numpy()
    is_nb = (df["business_type"].astype(str) == NB).to_numpy()

    cr = cfg["conversion_response.channel_effect"]
    # New business
    nb_eta = (
        cfg["conversion_response.intercept"]
        + ch.map(cr).to_numpy()
        + age.map(cfg["conversion_response.age_band_effect"]).to_numpy()
        + reg.map(cfg["conversion_response.region_effect"]).to_numpy()
        + risk.map(cfg["conversion_response.risk_band_effect"]).to_numpy()
        + _season(cfg, month)
        + ch.map(cfg["conversion_response.true_price_coefficient_by_channel"]).to_numpy()
        * age.map(cfg["conversion_response.true_price_multiplier_by_age_band"]).to_numpy()
        * ln_r
        + ch.map(cfg["conversion_response.rank_sensitivity_by_channel"]).to_numpy()
        * (cfg["conversion_response.rank_log_coefficient"] * np.log(rank) + cfg["conversion_response.top_rank_bonus"] * (rank == 1))
        + cfg["conversion_response.latent_intent_coefficient"] * u
    )
    # Renewal
    prior = df["prior_year_premium"].to_numpy(dtype=float)
    tenure = df["tenure"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ln_chg = np.log(price / prior)
        ln_ten = np.log(np.maximum(tenure, 1.0))
    ren_eta = (
        cfg["renewal_response.intercept"]
        + ch.map(cfg["renewal_response.channel_effect"]).to_numpy()
        + age.map(cfg["renewal_response.age_band_effect"]).to_numpy()
        + cfg["renewal_response.tenure_log_coefficient"] * ln_ten
        + cfg["renewal_response.price_change_coefficient"] * ln_chg
        + ch.map(cfg["renewal_response.market_ratio_coefficient_by_channel"]).to_numpy() * ln_r
        + cfg["renewal_response.rank_log_coefficient"] * np.log(rank)
        + cfg["renewal_response.latent_intent_coefficient"] * u
    )
    eta = np.where(is_nb, nb_eta, np.nan_to_num(ren_eta, nan=0.0))
    return 1.0 / (1.0 + np.exp(-eta))
