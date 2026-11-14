"""Module 3 models: binomial GLM, gradient boosting, naive benchmark."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from ..core.config import Config
from .features import NB_SPEC, GBMSpec, GLMDesign

try:  # LightGBM is preferred. HistGradientBoosting is the documented fallback.
    import lightgbm as lgb

    HAVE_LGBM = True
except Exception:  # pragma: no cover
    HAVE_LGBM = False


@dataclass
class GLMModel:
    columns: list[str]
    params: np.ndarray
    bse: np.ndarray
    pvalues: np.ndarray
    k: int

    def predict(self, design: GLMDesign, price: np.ndarray | None = None) -> np.ndarray:
        return expit(design.matrix(price) @ self.params)

    def eta(self, design: GLMDesign, price: np.ndarray | None = None) -> np.ndarray:
        return design.matrix(price) @ self.params

    def coef_table(self) -> pd.DataFrame:
        return pd.DataFrame({"term": self.columns, "coef": self.params, "std_err": self.bse, "p_value": self.pvalues})

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"columns": self.columns, "params": self.params.tolist(),
                                    "bse": self.bse.tolist(), "pvalues": self.pvalues.tolist(), "k": self.k}))

    @staticmethod
    def load(path: Path) -> "GLMModel":
        d = json.loads(path.read_text())
        return GLMModel(d["columns"], np.array(d["params"]), np.array(d["bse"]), np.array(d["pvalues"]), d["k"])


def fit_glm(df: pd.DataFrame, k: int, target: str = "converted") -> GLMModel:
    design = GLMDesign(df, k)
    res = sm.GLM(df[target].to_numpy(dtype=float), design.matrix(), family=sm.families.Binomial()).fit()
    return GLMModel(design.columns, res.params, res.bse, res.pvalues, k)


class GBMModel:
    """LightGBM wrapper with a HistGradientBoosting fallback. Same predict interface."""

    def __init__(self, model, backend: str, k: int, params: dict, spec: GBMSpec = NB_SPEC):
        self.model, self.backend, self.k, self.params, self.spec = model, backend, k, params, spec

    def predict(self, df: pd.DataFrame, price: np.ndarray | None = None, **kw) -> np.ndarray:
        X = self.spec.frame_fn(df, self.k, price, **kw)
        return self._predict_frame(X)

    def _predict_frame(self, X: pd.DataFrame) -> np.ndarray:
        if self.backend == "lightgbm":
            return self.model.predict(X)
        return self.model.predict_proba(self._sk_matrix(X, self.spec.categorical))[:, 1]

    @staticmethod
    def _sk_matrix(X: pd.DataFrame, categorical: list) -> pd.DataFrame:
        Z = X.copy()
        for c in categorical:
            Z[c] = Z[c].cat.codes
        return Z

    def save(self, directory: Path, name: str = "conversion_gbm") -> None:
        if self.backend == "lightgbm":
            self.model.save_model(str(directory / f"{name}.txt"))
        (directory / f"{name}_meta.json").write_text(json.dumps({"backend": self.backend, "k": self.k, "params": self.params}))

    @staticmethod
    def load(directory: Path, name: str = "conversion_gbm", spec: GBMSpec = NB_SPEC) -> "GBMModel":
        meta = json.loads((directory / f"{name}_meta.json").read_text())
        if meta["backend"] != "lightgbm":
            raise RuntimeError("Fallback models are refitted, not loaded")
        booster = lgb.Booster(model_file=str(directory / f"{name}.txt"))
        return GBMModel(_BoosterWrap(booster), "lightgbm", meta["k"], meta["params"], spec)


class _BoosterWrap:
    def __init__(self, booster):
        self.booster = booster

    def predict(self, X):
        return self.booster.predict(X)

    def save_model(self, path: str) -> None:
        self.booster.save_model(path)


def _search_space(rng: np.random.Generator, n_iter: int) -> list[dict]:
    out = []
    for _ in range(n_iter):
        out.append(
            {
                "learning_rate": float(rng.choice([0.03, 0.05, 0.08])),
                "num_leaves": int(rng.choice([7, 15, 31])),
                "min_child_samples": int(rng.choice([50, 100, 200, 400])),
                "subsample": float(rng.choice([0.7, 0.85, 1.0])),
                "colsample_bytree": float(rng.choice([0.7, 0.85, 1.0])),
                "reg_lambda": float(rng.choice([0.0, 1.0, 5.0, 20.0])),
            }
        )
    return out


def fit_gbm(train: pd.DataFrame, val: pd.DataFrame, k: int, cfg: Config, backend: str | None = None,
            n_iter: int | None = None, target: str = "converted", seed: int | None = None,
            spec: GBMSpec = NB_SPEC) -> tuple[GBMModel, pd.DataFrame]:
    """Random search scored on the validation set only. The test set is never touched here."""
    backend = backend or ("lightgbm" if HAVE_LGBM else "sklearn")
    seed = cfg["modelling.seed"] if seed is None else seed
    n_iter = cfg["modelling.gbm_search_iterations"] if n_iter is None else n_iter
    rng = np.random.default_rng(seed)
    Xt, yt = spec.frame_fn(train, k), train[target].to_numpy()
    Xv, yv = spec.frame_fn(val, k), val[target].to_numpy()
    log, best, best_ll = [], None, np.inf
    mono = [spec.monotone.get(c, 0) for c in spec.features]
    for i, hp in enumerate(_search_space(rng, n_iter)):
        if backend == "lightgbm":
            m = lgb.LGBMClassifier(n_estimators=1500, random_state=seed, n_jobs=2, verbose=-1, subsample_freq=1,
                                   monotone_constraints=mono, **hp)
            m.fit(Xt, yt, eval_set=[(Xv, yv)], eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            pv = m.predict_proba(Xv)[:, 1]
            n_trees = int(m.best_iteration_ or m.n_estimators)
            model = m
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier

            m = HistGradientBoostingClassifier(
                learning_rate=hp["learning_rate"], max_leaf_nodes=hp["num_leaves"], min_samples_leaf=hp["min_child_samples"],
                l2_regularization=hp["reg_lambda"], max_iter=400, early_stopping=False, random_state=seed,
                categorical_features=[spec.features.index(c) for c in spec.categorical],
            )
            m.fit(GBMModel._sk_matrix(Xt, spec.categorical), yt)
            pv = m.predict_proba(GBMModel._sk_matrix(Xv, spec.categorical))[:, 1]
            n_trees = 400
            model = m
        ll = log_loss(yv, pv)
        log.append({"iteration": i, **hp, "n_trees": n_trees, "val_log_loss": ll, "val_auc": roc_auc_score(yv, pv)})
        if ll < best_ll:
            best_ll, best = ll, (model, hp, n_trees)
    model, hp, n_trees = best
    params = {**hp, "n_trees": n_trees, "backend": backend}
    if backend == "lightgbm":
        return GBMModel(_BoosterWrap(model.booster_), backend, k, params, spec), pd.DataFrame(log)
    return GBMModel(model, backend, k, params, spec), pd.DataFrame(log)


class NaiveBenchmark:
    """Segment-by-channel conversion rate from the training period."""

    def __init__(self, train: pd.DataFrame, target: str = "converted"):
        key = train["segment"].astype(str) + "|" + train["channel"].astype(str)
        self.rate = train.groupby(key)[target].mean()
        self.channel_rate = train.groupby(train["channel"].astype(str))[target].mean()
        self.overall = float(train[target].mean())

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        key = df["segment"].astype(str) + "|" + df["channel"].astype(str)
        p = key.map(self.rate)
        p = p.fillna(df["channel"].astype(str).map(self.channel_rate)).fillna(self.overall)
        return p.to_numpy(dtype=float)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return {"roc_auc": float(roc_auc_score(y, p)), "log_loss": float(log_loss(y, p)),
            "brier": float(brier_score_loss(y, p)), "mean_prediction": float(p.mean()), "observed_rate": float(np.mean(y))}


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": y, "p": p})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), bins, labels=False)
    return d.groupby("bin").agg(mean_predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index()
