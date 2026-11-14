"""Seeded generator for the simulated quote market. Simulated data only."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .. import DATA_LABEL
from ..core.config import Config
from ..core.features import (
    AGE_BANDS, CHANNELS, NB, REGIONS, REN, RISK_BANDS, acquisition_cost, comp_columns,
    market_position, premium_relativities, unit_cost,
)
from .truth import true_probability


def _choice(rng: np.random.Generator, mix: dict, n: int) -> np.ndarray:
    keys = list(mix.keys())
    p = np.array([mix[k] for k in keys], dtype=float)
    return np.array(keys, dtype=object)[rng.choice(len(keys), size=n, p=p / p.sum())]


def _market_shocks(rng: np.random.Generator, cfg: Config) -> np.ndarray:
    m = cfg["volumes.n_months"]
    sd, rho = cfg["market.market_shock_sd"], cfg["market.market_shock_persistence"]
    shock = np.zeros(m)
    for t in range(1, m):
        shock[t] = rho * shock[t - 1] + rng.normal(0, sd)
    return shock


def generate(cfg: Config, seed: int | None = None, n_quotes: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (quotes, truth). Same seed and config give identical output."""
    seed = cfg["modelling.seed"] if seed is None else seed
    n = int(cfg["volumes.n_quotes"] if n_quotes is None else n_quotes)
    rng = np.random.default_rng(seed)
    K = cfg["market.n_competitors"]
    n_months = cfg["volumes.n_months"]

    # Dates with mild volume seasonality
    start = pd.Timestamp(cfg["volumes.start_date"])
    month_starts = pd.date_range(start, periods=n_months, freq="MS")
    cal_month = month_starts.month.to_numpy()
    w = 1.0 + cfg["volumes.volume_seasonality_amplitude"] * np.cos(2 * np.pi * (cal_month - 1) / 12.0)
    midx = rng.choice(n_months, size=n, p=w / w.sum())
    dim = month_starts.days_in_month.to_numpy()[midx]
    day_frac = rng.random(n)
    day = np.floor(day_frac * dim).astype(int)
    quote_date = month_starts.to_numpy()[midx] + day.astype("timedelta64[D]")

    business = np.where(rng.random(n) < cfg["volumes.new_business_share"], NB, REN)
    is_nb = business == NB
    channel = np.where(
        is_nb,
        _choice(rng, cfg["volumes.channel_mix_new_business"], n),
        _choice(rng, cfg["volumes.channel_mix_renewal"], n),
    )
    age = _choice(rng, cfg["volumes.age_band_mix"], n)
    region = _choice(rng, cfg["volumes.region_mix"], n)
    risk = _choice(rng, cfg["volumes.risk_band_mix"], n)

    ten_p = np.array(cfg["volumes.renewal_tenure_probabilities"], dtype=float)
    tenure = np.where(is_nb, 0, rng.choice(np.arange(1, len(ten_p) + 1), size=n, p=ten_p / ten_p.sum()))

    rel = premium_relativities(cfg)
    ln_rel = (
        np.log(cfg["market.average_premium_level"])
        + np.log(pd.Series(age).map(rel["age"]).to_numpy())
        + np.log(pd.Series(region).map(rel["region"]).to_numpy())
        + np.log(pd.Series(risk).map(rel["risk"]).to_numpy())
    )
    g = np.log(1.0 + cfg["market.annual_premium_trend"])
    shocks = _market_shocks(rng, cfg)
    t_now = (midx + day_frac) / 12.0
    profile = rng.normal(0, cfg["market.profile_noise_sd"], n)
    ln_market = ln_rel + g * t_now + shocks[midx] + profile

    # Competitors
    mu = np.array(cfg["market.competitor_mean_offsets"], dtype=float)
    z = rng.normal(0, cfg["market.competitor_price_noise_sd"], (n, K))
    comp = np.sort(np.exp(ln_market[:, None] + mu[None, :] + z), axis=1)

    # Our price. u is unobserved intent that raises conversion and is loaded into price.
    u = rng.normal(0, 1, n)
    off = (
        pd.Series(age).map(cfg["our_pricing.age_band_position_offset"]).to_numpy()
        + pd.Series(region).map(cfg["our_pricing.region_position_offset"]).to_numpy()
    )
    interval = cfg["our_pricing.repricing_interval_months"]
    q_idx = (midx // interval) * interval
    ln_ours_nb = (
        ln_rel + g * (q_idx / 12.0) + shocks[q_idx] + profile + off
        + rng.normal(0, cfg["our_pricing.own_price_noise_sd"], n)
        + cfg["our_pricing.intent_price_loading"] * u
    )
    ln_prior = (
        ln_market - g + off + rng.normal(0, cfg["renewal_response.prior_premium_noise_sd"], n)
    )
    ln_change = (
        np.log(1.0 + cfg["market.annual_premium_trend"] * cfg["renewal_response.trend_passthrough"])
        + pd.Series(age).map(cfg["renewal_response.rate_action_by_age_band"]).to_numpy()
        + rng.normal(0, cfg["renewal_response.change_noise_sd"], n)
        + cfg["renewal_response.intent_price_loading"] * u
    )
    reference = np.where(is_nb, np.exp(ln_ours_nb), np.exp(ln_prior + ln_change))
    prior_premium = np.where(is_nb, np.nan, np.exp(ln_prior))

    # Randomised price test on a share of new business quotes
    arms = cfg["volumes.test_arm_probabilities"]
    arm_chg = cfg["volumes.test_arm_price_changes"]
    in_test = is_nb & (rng.random(n) < cfg["volumes.test_share_of_new_business"])
    group = np.where(in_test, _choice(rng, arms, n), "none").astype(object)
    mult = 1.0 + np.array([arm_chg.get(gp, 0.0) for gp in group], dtype=float)
    price = reference * mult

    pos = market_position(price, comp)
    df = pd.DataFrame(
        {
            "quote_date": pd.to_datetime(quote_date),
            "business_type": business,
            "channel": channel,
            "age_band": age,
            "region": region,
            "pre_existing_risk_band": risk,
            "tenure": tenure.astype(int),
            "prior_year_premium": prior_premium,
            "our_premium": price,
            "reference_premium": reference,
            "comp_min": comp[:, 0],
            "comp_median": np.median(comp, axis=1),
            "comp_mean": comp.mean(axis=1),
        }
    )
    for j, c in enumerate(comp_columns(K)):
        df[c] = comp[:, j]
    df["our_rank"] = pos["rank"]
    df["price_ratio_to_median"] = pos["ratio"]
    df["price_gap_to_cheapest"] = pos["gap_cheapest"]
    df["treatment_group"] = group
    df["segment"] = df["age_band"].astype(str) + "|" + df["region"].astype(str)

    p_true = true_probability(df, u, cfg)
    converted = (rng.random(n) < p_true).astype(int)
    df["converted"] = converted
    is_ren = ~is_nb
    df["renewed_flag"] = pd.array(np.where(is_ren, converted, -1), dtype="Int64").astype("Int64")
    df.loc[~is_ren, "renewed_flag"] = pd.NA
    df["renewal_outcome"] = np.where(is_ren, np.where(converted == 1, "renewed", "lapsed"), "not_applicable")

    uc = unit_cost(cfg, df["age_band"], df["region"], df["pre_existing_risk_band"])
    acq = acquisition_cost(cfg, df["channel"], df["business_type"])
    expense = cfg["costs.expense_per_policy"]
    df["written_premium"] = df["our_premium"] * converted
    df["unit_cost"] = uc * converted
    df["acquisition_cost"] = acq * converted
    df["fixed_cost_contribution"] = converted * (df["our_premium"] - uc - expense - acq)

    # Order, keys, types
    df = df.sort_values("quote_date", kind="stable").reset_index(drop=True)
    perm = rng.permutation(n)
    df.insert(0, "quote_id", ["Q" + str(i + 1).zfill(8) for i in range(n)])
    df.insert(2, "customer_id", ["C" + str(p).zfill(8) for p in perm])
    df["data_label"] = DATA_LABEL

    # Truth must follow the same row order as df. Recompute u and p on the sorted frame.
    order = np.argsort(quote_date.astype("datetime64[ns]"), kind="stable")
    u_sorted = u[order]
    truth = pd.DataFrame(
        {"quote_id": df["quote_id"], "latent_intent": u_sorted, "true_probability": p_true[order]}
    )
    for col in ["business_type", "channel", "age_band", "region", "pre_existing_risk_band", "treatment_group", "segment", "renewal_outcome"]:
        df[col] = df[col].astype("category")
    return df, truth


def frame_hash(df: pd.DataFrame) -> str:
    """Content hash independent of file format."""
    h = hashlib.sha256()
    h.update(",".join(df.columns).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return h.hexdigest()
