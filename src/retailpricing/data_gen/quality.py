"""Data quality checks on the simulated quote table."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.config import Config
from ..core.features import AGE_BANDS, CHANNELS, NB, REGIONS, REN, RISK_BANDS, comp_columns, market_position

REQUIRED = [
    "quote_id", "quote_date", "customer_id", "business_type", "channel", "our_premium", "comp_min",
    "comp_median", "comp_mean", "our_rank", "price_ratio_to_median", "price_gap_to_cheapest", "tenure",
    "segment", "pre_existing_risk_band", "treatment_group", "converted", "written_premium",
    "fixed_cost_contribution",
]


def run_quality_checks(df: pd.DataFrame, cfg: Config, truth: pd.DataFrame | None = None,
                       min_rows: int = 150_000, max_rows: int = 250_000) -> pd.DataFrame:
    K = cfg["market.n_competitors"]
    comp = df[comp_columns(K)].to_numpy()
    pos = market_position(df["our_premium"].to_numpy(), comp)
    nb = df["business_type"].astype(str) == NB
    ren = ~nb
    checks: list[tuple[str, bool, str]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    add("required columns present", all(c in df.columns for c in REQUIRED), f"{len(REQUIRED)} required")
    add(f"row count within {min_rows:,} to {max_rows:,}", min_rows <= len(df) <= max_rows, f"{len(df):,} rows")
    add("quote_id unique", df["quote_id"].is_unique)
    add("customer_id unique", df["customer_id"].is_unique)
    add("no nulls in required fields", int(df[REQUIRED].isna().sum().sum()) == 0)
    add("our_premium positive", (df["our_premium"] > 0).all())
    add("competitor prices sorted ascending", (np.diff(comp, axis=1) >= 0).all())
    add("comp_min <= comp_median <= comp_max", ((df["comp_min"] <= df["comp_median"]) & (df["comp_median"] <= comp[:, -1])).all())
    add("comp_mean between min and max", ((df["comp_mean"] >= df["comp_min"]) & (df["comp_mean"] <= comp[:, -1])).all())
    add("rank recomputes from competitor prices", (pos["rank"] == df["our_rank"].to_numpy()).all())
    add("rank between 1 and n_competitors + 1", df["our_rank"].between(1, K + 1).all())
    add("price ratio recomputes", np.allclose(pos["ratio"], df["price_ratio_to_median"]))
    add("gap to cheapest recomputes", np.allclose(pos["gap_cheapest"], df["price_gap_to_cheapest"]))
    add("converted is 0 or 1", df["converted"].isin([0, 1]).all())
    add("written premium equals premium when converted", np.allclose(df["written_premium"], df["our_premium"] * df["converted"]))
    add("contribution zero when not converted", (df.loc[df["converted"] == 0, "fixed_cost_contribution"] == 0).all())
    add("NB rows have tenure 0 and no prior premium", ((df.loc[nb, "tenure"] == 0).all()) and df.loc[nb, "prior_year_premium"].isna().all())
    add("renewal rows have tenure >= 1 and prior premium", ((df.loc[ren, "tenure"] >= 1).all()) and df.loc[ren, "prior_year_premium"].notna().all())
    add("renewal flag only on renewal rows", df.loc[nb, "renewed_flag"].isna().all() and df.loc[ren, "renewed_flag"].notna().all())
    add("treatment only on new business", (df.loc[ren, "treatment_group"].astype(str) == "none").all())
    add("reference equals quoted premium outside test", np.allclose(
        df.loc[df["treatment_group"].astype(str) == "none", "reference_premium"],
        df.loc[df["treatment_group"].astype(str) == "none", "our_premium"]))
    add("valid age bands", df["age_band"].astype(str).isin(AGE_BANDS).all())
    add("valid regions", df["region"].astype(str).isin(REGIONS).all())
    add("valid channels", df["channel"].astype(str).isin(CHANNELS).all())
    add("valid risk bands", df["pre_existing_risk_band"].astype(str).isin(RISK_BANDS).all())
    start = pd.Timestamp(cfg["volumes.start_date"])
    end = start + pd.DateOffset(months=cfg["volumes.n_months"])
    add("quote dates inside configured window", ((df["quote_date"] >= start) & (df["quote_date"] < end)).all())
    nb_obs = df[nb & (df["treatment_group"].astype(str) == "none")]
    rate = nb_obs["converted"].mean()
    add("NB conversion rate plausible (1% to 40%)", 0.01 < rate < 0.40, f"{rate:.3%}")
    rr = df.loc[ren, "converted"].mean()
    add("renewal rate plausible (40% to 98%)", 0.40 < rr < 0.98, f"{rr:.3%}")
    if truth is not None:
        add("true probabilities within [0, 1]", truth["true_probability"].between(0, 1).all())
        add("truth aligned to quotes", (truth["quote_id"].to_numpy() == df["quote_id"].to_numpy()).all())
    add("data_label is 'Simulated data'", (df["data_label"] == "Simulated data").all())
    out = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    out.attrs["data_label"] = "Simulated data"
    return out
