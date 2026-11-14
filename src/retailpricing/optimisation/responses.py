"""Response functions used by the optimiser and its evaluation. Simulated data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from ..conversion.features import GLMDesign
from ..conversion.models import GLMModel
from ..core import paths
from ..core.config import Config
from ..core.io import read_json
from ..data_gen.truth import true_probability
from ..elasticity.core import ShiftedGLM
from ..retention.features import RenewalDesign
from ..retention.models import RenewalGLM


def build_responses(cfg: Config, nb: pd.DataFrame, ren: pd.DataFrame, u_nb: np.ndarray, u_ren: np.ndarray, smooth: bool) -> dict:
    """Return callables price -> probability for new business and renewal rows.

    smooth=True uses differentiable rank inside the models (optimiser). smooth=False is exact (evaluation).
    The 'true' responses use the data generating process and are for validation only.
    """
    k = cfg["market.n_competitors"]
    width = cfg["optimisation.rank_smoothing_width"] if smooth else None
    mdir = paths.data_processed() / "models"
    glm = GLMModel.load(mdir / "conversion_glm.json")
    rglm = RenewalGLM.load(mdir / "renewal_glm.json")
    shifts = read_json("m4_calibration_shift.json")["shift_by_channel"]
    dn = GLMDesign(nb, k, smooth=width)
    p0 = nb["our_premium"].to_numpy(dtype=float)
    delta = nb["channel"].astype(str).map(shifts).to_numpy(dtype=float)
    cal = ShiftedGLM(glm, dn, delta, p0)
    rd = RenewalDesign(ren, k, smooth=width)
    return {
        "nb_observational": lambda pr: glm.predict(dn, pr),
        "nb_calibrated": cal.predict,
        "nb_true": lambda pr: true_probability(nb, u_nb, cfg, pr),
        "ren_model": lambda pr: expit(rd.matrix(pr) @ rglm.params),
        "ren_true": lambda pr: true_probability(ren, u_ren, cfg, pr),
    }
