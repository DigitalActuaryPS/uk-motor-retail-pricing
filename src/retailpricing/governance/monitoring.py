"""Module 8: out-of-time tests, PSI, stress tests. Simulated data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from ..conversion.features import gbm_frame


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10, categorical: bool = False) -> float:
    """Population stability index: sum (a - e) ln(a / e). Bin edges come from the expected (training) sample."""
    eps = 1e-4
    if categorical:
        cats = sorted(set(pd.unique(expected)) | set(pd.unique(actual)))
        e = np.array([(expected == c).mean() for c in cats])
        a = np.array([(actual == c).mean() for c in cats])
    else:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, edges)[0] / len(expected)
        a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(((a - e) * np.log(a / e)).sum())


def rag(value: float, amber: float, red: float) -> str:
    return "red" if value >= red else ("amber" if value >= amber else "green")


def monthly_out_of_time(frame: pd.DataFrame, preds: dict[str, np.ndarray], target: str = "converted") -> pd.DataFrame:
    rows = []
    month = frame["quote_date"].dt.to_period("M").astype(str).to_numpy()
    y = frame[target].to_numpy()
    for m in sorted(set(month)):
        idx = month == m
        if idx.sum() < 200 or len(np.unique(y[idx])) < 2:
            continue
        for name, p in preds.items():
            pm = np.clip(p[idx], 1e-9, 1 - 1e-9)
            rows.append({"month": m, "model": name, "rows": int(idx.sum()), "roc_auc": roc_auc_score(y[idx], pm),
                         "log_loss": log_loss(y[idx], pm), "brier": brier_score_loss(y[idx], pm),
                         "actual_to_expected": float(y[idx].mean() / pm.mean())})
    return pd.DataFrame(rows)


def feature_psi_table(train: pd.DataFrame, test: pd.DataFrame, k: int, score_train: np.ndarray, score_test: np.ndarray,
                      amber: float, red: float) -> pd.DataFrame:
    a, b = gbm_frame(train, k), gbm_frame(test, k)
    rows = []
    for c in ["ln_ratio", "rank", "cheapest_gap", "ln_comp_spread"]:
        v = psi(a[c].to_numpy(), b[c].to_numpy())
        rows.append({"item": c, "type": "feature", "psi": v, "rag": rag(v, amber, red)})
    for c in ["channel", "age_band", "region", "risk_band"]:
        v = psi(a[c].astype(str).to_numpy(), b[c].astype(str).to_numpy(), categorical=True)
        rows.append({"item": c, "type": "feature", "psi": v, "rag": rag(v, amber, red)})
    v = psi(score_train, score_test)
    rows.append({"item": "model_score", "type": "score", "psi": v, "rag": rag(v, amber, red)})
    return pd.DataFrame(rows)


def elasticity_stress(resp, base_price: np.ndarray, multiplier: float):
    """Scale the response of the linear predictor to a price change away from the baseline."""
    p_base = np.clip(resp(base_price), 1e-9, 1 - 1e-9)
    l0 = logit(p_base)

    def stressed(price):
        p = np.clip(resp(price), 1e-9, 1 - 1e-9)
        return expit(l0 + multiplier * (logit(p) - l0))

    return stressed
