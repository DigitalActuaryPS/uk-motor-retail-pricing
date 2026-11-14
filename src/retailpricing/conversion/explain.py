"""SHAP, partial dependence and price-response curves. Simulated data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..core.config import Config
from ..data_gen.truth import true_probability
from .features import GLMDesign, gbm_frame
from .models import GBMModel, GLMModel


def shap_importance(gbm: GBMModel, df: pd.DataFrame, k: int, n: int = 4000, seed: int = 1):
    import shap

    sample = df.sample(min(n, len(df)), random_state=seed)
    X = gbm_frame(sample, k)
    explainer = shap.TreeExplainer(gbm.model.booster)
    sv = explainer.shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(sv).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    return imp, sv, X


def save_shap_beeswarm(sv, X, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    Xn = X.copy()
    for c in Xn.columns:
        if str(Xn[c].dtype) == "category":
            Xn[c] = Xn[c].cat.codes
    plt.figure(figsize=(8, 5))
    shap.summary_plot(sv, Xn, show=False)
    plt.title("SHAP summary, LightGBM conversion model | Simulated data")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def price_response(df: pd.DataFrame, k: int, glm: GLMModel, gbm: GBMModel, u: np.ndarray, cfg: Config,
                   changes=np.linspace(-0.10, 0.10, 21), by_channel: bool = True) -> pd.DataFrame:
    """Average conversion probability across price changes, for the GLM, GBM and the true process."""
    rows = []
    groups = [("all", np.ones(len(df), dtype=bool))]
    if by_channel:
        groups += [(c, (df["channel"].astype(str) == c).to_numpy()) for c in ["PCW", "direct", "broker"]]
    design = GLMDesign(df, k)
    for name, mask in groups:
        sub = df[mask]
        for x in changes:
            price = df["our_premium"].to_numpy() * (1 + x)
            p_glm = glm.predict(design, price)[mask].mean()
            p_gbm = gbm.predict(df, price)[mask].mean()
            p_true = true_probability(df, u, cfg, price)[mask].mean()
            rows.append({"group": name, "price_change": float(x), "glm": p_glm, "gbm": p_gbm, "true": p_true})
    return pd.DataFrame(rows)


def partial_dependence_ln_ratio(df: pd.DataFrame, k: int, glm: GLMModel, gbm: GBMModel, grid=None) -> pd.DataFrame:
    """Partial dependence on the price ratio with rank held at its observed value (feature-level view)."""
    grid = np.linspace(0.85, 1.20, 15) if grid is None else grid
    X = gbm_frame(df, k)
    design = GLMDesign(df, k)
    rows = []
    for r in grid:
        Xr = X.copy()
        Xr["ratio"] = r
        Xr["ln_ratio"] = np.log(r)
        rows.append({"ratio": float(r), "gbm_pd": float(gbm._predict_frame(Xr).mean())})
    out = pd.DataFrame(rows)
    # GLM partial dependence: shift ln_ratio only (rank terms fixed)
    base = design.matrix()
    cols = design.columns
    pd_glm = []
    for r in grid:
        Xm = base.copy()
        lr_new = np.log(r)
        for j, name in enumerate(cols):
            if name == "ln_ratio":
                Xm[:, j] = lr_new
            elif name == "ln_ratio:direct":
                Xm[:, j] = lr_new * design.is_direct
            elif name == "ln_ratio:broker":
                Xm[:, j] = lr_new * design.is_broker
            elif name.startswith("ln_ratio:age_"):
                a = design.age_d[[f"age_{x}" for x in ["25-34", "35-49", "50-64", "65+"]].index(name.split(":")[1])]
                Xm[:, j] = lr_new * a
        pd_glm.append(float((1 / (1 + np.exp(-(Xm @ glm.params)))).mean()))
    out["glm_pd"] = pd_glm
    return out


def plot_price_response(pr: pd.DataFrame) -> go.Figure:
    from ..core.plotting import finalise

    fig = go.Figure()
    d = pr[pr["group"] == "all"]
    fig.add_scatter(x=d["price_change"], y=d["true"], name="True process", line=dict(color="black", dash="dash"))
    fig.add_scatter(x=d["price_change"], y=d["glm"], name="GLM")
    fig.add_scatter(x=d["price_change"], y=d["gbm"], name="LightGBM")
    fig.update_xaxes(tickformat=".0%", title="Price change")
    fig.update_yaxes(tickformat=".1%", title="Average conversion probability")
    return finalise(fig, "Price-response curves, test period, all channels")


def plot_calibration(tables: dict[str, pd.DataFrame]) -> go.Figure:
    from ..core.plotting import finalise

    fig = go.Figure()
    mx = max(t[["mean_predicted", "observed"]].to_numpy().max() for t in tables.values())
    fig.add_scatter(x=[0, mx], y=[0, mx], name="Perfect", line=dict(color="grey", dash="dot"))
    for name, t in tables.items():
        fig.add_scatter(x=t["mean_predicted"], y=t["observed"], mode="lines+markers", name=name)
    fig.update_xaxes(title="Mean predicted conversion", tickformat=".0%")
    fig.update_yaxes(title="Observed conversion", tickformat=".0%")
    return finalise(fig, "Calibration on the test period (deciles)")
