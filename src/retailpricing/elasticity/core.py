"""Module 4: elasticity, scenarios, randomised price test. Simulated data only."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit

from ..core.config import Config
from ..core.features import AGE_BANDS, REGIONS, unit_economics
from ..conversion.features import GLMDesign
from ..conversion.models import GLMModel
from ..data_gen.truth import true_probability

ProbFn = Callable[[np.ndarray], np.ndarray]


def arc_elasticity(fn: ProbFn, price0: np.ndarray, mask: np.ndarray | None = None, step: float = 0.01) -> float:
    """[ln E(P(1+s)) - ln E(P(1-s))] / [ln(1+s) - ln(1-s)], E = expected policies. Rank is recomputed inside fn."""
    up, dn = fn(price0 * (1 + step)), fn(price0 * (1 - step))
    if mask is not None:
        up, dn = up[mask], dn[mask]
    return float((np.log(up.sum()) - np.log(dn.sum())) / (np.log1p(step) - np.log1p(-step)))


@dataclass
class ScaledGLM:
    """GLM whose price response (change in linear predictor from baseline) is scaled by channel.

    Baseline predictions are unchanged. Used to align an observational GLM with randomised-test elasticity.
    """

    glm: GLMModel
    design: GLMDesign
    scale: np.ndarray  # per row
    base_price: np.ndarray

    def __post_init__(self):
        self.eta0 = self.glm.eta(self.design, self.base_price)

    def predict(self, price: np.ndarray) -> np.ndarray:
        eta = self.glm.eta(self.design, price)
        return expit(self.eta0 + self.scale * (eta - self.eta0))


@dataclass
class ShiftedGLM:
    """GLM with an added log-price term delta_c * ln(P'/P0). Baseline predictions are unchanged.

    delta_c is solved so the arc elasticity of channel c equals a target taken from the randomised test.
    Unlike a pure scale factor it also repairs a wrong-signed observational coefficient.
    """

    glm: GLMModel
    design: GLMDesign
    delta: np.ndarray  # per row
    base_price: np.ndarray

    def predict(self, price: np.ndarray) -> np.ndarray:
        eta = self.glm.eta(self.design, price) + self.delta * np.log(np.asarray(price) / self.base_price)
        return expit(eta)


def calibrate_shifts(glm: GLMModel, design: GLMDesign, df: pd.DataFrame, price0: np.ndarray,
                     targets: dict[str, float], step: float) -> dict[str, float]:
    from scipy.optimize import brentq

    out = {}
    ch = df["channel"].astype(str).to_numpy()
    for c, target in targets.items():
        mask = ch == c

        def f(d: float) -> float:
            delta = np.where(mask, d, 0.0)
            return arc_elasticity(ShiftedGLM(glm, design, delta, price0).predict, price0, mask, step) - target

        out[c] = float(brentq(f, -40.0, 40.0, xtol=1e-6))
    return out


def scale_vector(df: pd.DataFrame, by_channel: dict[str, float]) -> np.ndarray:
    return df["channel"].astype(str).map(by_channel).to_numpy(dtype=float)


def elasticity_table(sources: dict[str, ProbFn], df: pd.DataFrame, price0: np.ndarray, step: float) -> pd.DataFrame:
    rows = []
    groups = {"overall": np.ones(len(df), dtype=bool)}
    for c in ["PCW", "direct", "broker"]:
        groups[f"channel:{c}"] = (df["channel"].astype(str) == c).to_numpy()
    for a in AGE_BANDS:
        groups[f"age:{a}"] = (df["age_band"].astype(str) == a).to_numpy()
    for r in REGIONS:
        groups[f"region:{r}"] = (df["region"].astype(str) == r).to_numpy()
    for s in sorted(df["segment"].astype(str).unique()):
        groups[f"segment:{s}"] = (df["segment"].astype(str) == s).to_numpy()
    for g, m in groups.items():
        row = {"group": g, "quotes": int(m.sum())}
        for name, fn in sources.items():
            row[name] = arc_elasticity(fn, price0, m, step)
        rows.append(row)
    return pd.DataFrame(rows)


def scenario_table(fn: ProbFn, df: pd.DataFrame, cfg: Config, changes: list[float]) -> pd.DataFrame:
    econ = unit_economics(cfg, df)
    cost = econ["unit_cost"] + econ["expense"] + econ["acquisition"]
    price0 = df["our_premium"].to_numpy(dtype=float)
    rows = []
    for x in changes:
        price = price0 * (1 + x)
        p = fn(price)
        policies = p.sum()
        gwp = float((p * price).sum())
        contrib = float((p * (price - cost)).sum())
        rows.append({"price_change": x, "conversion_probability": float(p.mean()), "expected_policies": float(policies),
                     "gwp": gwp, "average_premium": gwp / policies, "contribution": contrib, "margin": contrib / gwp})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- randomised test

def _exp_design(df: pd.DataFrame, group_cols: bool = True) -> pd.DataFrame:
    X = pd.DataFrame({"const": 1.0, "ln_price_change": np.log1p(df["our_premium"] / df["reference_premium"] - 1.0)}, index=df.index)
    if group_cols:
        ch = df["channel"].astype(str)
        X["ch_direct"] = (ch == "direct").astype(float)
        X["ch_broker"] = (ch == "broker").astype(float)
    for a in AGE_BANDS[1:]:
        X[f"age_{a}"] = (df["age_band"].astype(str) == a).astype(float)
    for r in REGIONS[1:]:
        X[f"reg_{r}"] = (df["region"].astype(str) == r).astype(float)
    return X


def _exp_elasticity(df: pd.DataFrame, group_cols: bool = True, drop: list[str] | None = None) -> float:
    X = _exp_design(df, group_cols)
    if drop:
        X = X.drop(columns=[c for c in drop if c in X.columns])
        X = X.loc[:, X.nunique() > 1].assign(const=1.0)
    y = df["converted"].to_numpy(dtype=float)
    res = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    p = res.predict(X)
    beta = res.params["ln_price_change"]
    return float(beta * (p * (1 - p)).sum() / p.sum())


def experiment_elasticities(exp: pd.DataFrame, B: int, seed: int) -> pd.DataFrame:
    """Elasticity from the randomised test: overall, by channel, by age band, with bootstrap intervals."""
    rng = np.random.default_rng(seed)
    specs = {"overall": (exp, True, None)}
    for c in ["PCW", "direct", "broker"]:
        specs[f"channel:{c}"] = (exp[exp["channel"].astype(str) == c], False, None)
    for a in AGE_BANDS:
        specs[f"age:{a}"] = (exp[exp["age_band"].astype(str) == a], True, ["age_" + b for b in AGE_BANDS[1:]])
    specs["channel:off_PCW"] = (exp[exp["channel"].astype(str) != "PCW"], True, None)
    rows = []
    for name, (d, gc, drop) in specs.items():
        pt = _exp_elasticity(d, gc, drop)
        boots = []
        idx = np.arange(len(d))
        for _ in range(B):
            s = rng.choice(idx, size=len(idx), replace=True)
            try:
                boots.append(_exp_elasticity(d.iloc[s], gc, drop))
            except Exception:
                continue
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"group": name, "experiment": pt, "ci_low": float(lo), "ci_high": float(hi), "n": len(d),
                     "conversions": int(d["converted"].sum())})
    return pd.DataFrame(rows)


def arm_table(exp: pd.DataFrame) -> pd.DataFrame:
    g = exp.groupby("treatment_group", observed=True).agg(
        quotes=("converted", "size"), conversion=("converted", "mean"),
        mean_price_change=("our_premium", lambda s: 0.0)).reset_index()
    chg = (exp["our_premium"] / exp["reference_premium"] - 1).groupby(exp["treatment_group"], observed=True).mean()
    g["mean_price_change"] = g["treatment_group"].map(chg)
    ctrl = float(g.loc[g["treatment_group"] == "control", "conversion"].iloc[0])
    g["arc_elasticity_vs_control"] = np.where(g["mean_price_change"].abs() > 1e-9,
                                              np.log(g["conversion"] / ctrl) / np.log1p(g["mean_price_change"]), np.nan)
    order = {"T-5": 0, "T-2": 1, "control": 2, "T+2": 3, "T+5": 4}
    return g.sort_values("treatment_group", key=lambda s: s.map(order)).reset_index(drop=True)


# ----------------------------------------------------------------------------- endogeneity

def oracle_glm_elasticity(train: pd.DataFrame, u_train: np.ndarray, test: pd.DataFrame, u_test: np.ndarray, k: int, step: float) -> dict:
    """Refit the GLM with the latent intent included. Shows the bias is omitted-variable bias."""
    dtr = GLMDesign(train, k)
    X = np.column_stack([dtr.matrix(), u_train])
    res = sm.GLM(train["converted"].to_numpy(dtype=float), X, family=sm.families.Binomial()).fit()
    dte = GLMDesign(test, k)

    def fn(price):
        return expit(np.column_stack([dte.matrix(price), u_test]) @ res.params)

    price0 = test["our_premium"].to_numpy(dtype=float)
    out = {"overall": arc_elasticity(fn, price0, None, step)}
    for c in ["PCW", "direct", "broker"]:
        out[f"channel:{c}"] = arc_elasticity(fn, price0, (test["channel"].astype(str) == c).to_numpy(), step)
    out["intent_coefficient"] = float(res.params[-1])
    return out


def endogeneity_stats(df: pd.DataFrame, u: np.ndarray) -> dict:
    ln_r = np.log(df["price_ratio_to_median"].to_numpy())
    cov = float(np.cov(ln_r, u)[0, 1])
    var = float(np.var(ln_r, ddof=1))
    return {"corr_ln_ratio_intent": float(np.corrcoef(ln_r, u)[0, 1]), "slope_intent_on_ln_ratio": cov / var,
            "sd_ln_ratio": float(np.sqrt(var))}


def true_fn(df: pd.DataFrame, u: np.ndarray, cfg: Config, base_price: np.ndarray | None = None) -> ProbFn:
    return lambda price: true_probability(df, u, cfg, price)
