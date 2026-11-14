"""Renewal models (GLM and gradient boosting). Target: renewed (1) or lapsed (0). Simulated data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit

from ..conversion.models import GBMModel, GLMModel, NaiveBenchmark, calibration_table, fit_gbm, metrics
from .features import REN_SPEC, RenewalDesign


class RenewalGLM(GLMModel):
    @staticmethod
    def load(path) -> "RenewalGLM":
        base = GLMModel.load(path)
        return RenewalGLM(base.columns, base.params, base.bse, base.pvalues, base.k)

    def predict_df(self, df: pd.DataFrame, price=None) -> np.ndarray:
        return expit(RenewalDesign(df, self.k).matrix(price) @ self.params)


def fit_renewal_glm(df: pd.DataFrame, k: int) -> RenewalGLM:
    d = RenewalDesign(df, k)
    res = sm.GLM(df["converted"].to_numpy(dtype=float), d.matrix(), family=sm.families.Binomial()).fit()
    return RenewalGLM(d.columns, res.params, res.bse, res.pvalues, k)


def fit_renewal_gbm(train, val, k, cfg, **kw):
    return fit_gbm(train, val, k, cfg, spec=REN_SPEC, **kw)


def load_renewal_gbm(directory) -> GBMModel:
    return GBMModel.load(directory, "renewal_gbm", REN_SPEC)
