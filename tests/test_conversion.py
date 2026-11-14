import numpy as np
import pandas as pd
import pytest

from retailpricing.conversion.features import FORBIDDEN_FEATURES, GBM_FEATURES, GLMDesign, gbm_frame
from retailpricing.conversion.models import NaiveBenchmark, calibration_table, fit_gbm, fit_glm, metrics
from retailpricing.core.io import nb_observational, time_split


@pytest.fixture(scope="module")
def nb_small(small):
    return nb_observational(small[0])


def test_no_forbidden_features():
    assert not (set(GBM_FEATURES) & FORBIDDEN_FEATURES)


def test_time_split_is_chronological(nb_small, cfg):
    tr, va, te = time_split(nb_small, cfg)
    assert tr["quote_date"].max() < va["quote_date"].min()
    assert va["quote_date"].max() < te["quote_date"].min()
    assert len(tr) + len(va) + len(te) == len(nb_small)


def test_glm_design_reproduces_stored_position(nb_small, cfg):
    k = cfg["market.n_competitors"]
    d = GLMDesign(nb_small, k)
    X = d.matrix()
    j = d.columns.index("ln_ratio")
    assert np.allclose(X[:, j], np.log(nb_small["price_ratio_to_median"].to_numpy()))
    X2 = d.matrix(nb_small["our_premium"].to_numpy() * 1.05)
    assert (X2[:, j] > X[:, j]).all()


def test_glm_price_coefficient_negative(nb_small, cfg):
    k = cfg["market.n_competitors"]
    train, _, _ = time_split(nb_small, cfg)
    glm = fit_glm(train, k)
    coefs = dict(zip(glm.columns, glm.params))
    assert coefs["ln_ratio"] < 0
    p = glm.predict(GLMDesign(train, k))
    assert ((p > 0) & (p < 1)).all()


def test_naive_benchmark_hand_case():
    tr = pd.DataFrame({"segment": ["a", "a", "b", "b"], "channel": ["PCW"] * 4, "converted": [1, 0, 0, 0]})
    nb = NaiveBenchmark(tr)
    te = pd.DataFrame({"segment": ["a", "b", "z"], "channel": ["PCW"] * 3})
    assert nb.predict(te).tolist() == pytest.approx([0.5, 0.0, 0.25])


def test_gbm_both_backends(nb_small, cfg):
    k = cfg["market.n_competitors"]
    train, val, test = time_split(nb_small, cfg)
    for backend in ["lightgbm", "sklearn"]:
        m, log = fit_gbm(train, val, k, cfg, backend=backend, n_iter=2)
        p = m.predict(test)
        assert len(p) == len(test) and ((p >= 0) & (p <= 1)).all()
        assert len(log) == 2
        mt = metrics(test["converted"].to_numpy(), p)
        assert 0.5 < mt["roc_auc"] < 1.0


def test_gbm_price_override_changes_prediction(nb_small, cfg):
    k = cfg["market.n_competitors"]
    train, val, test = time_split(nb_small, cfg)
    m, _ = fit_gbm(train, val, k, cfg, backend="lightgbm", n_iter=1)
    p0 = m.predict(test)
    p1 = m.predict(test, test["our_premium"].to_numpy() * 1.10)
    assert p1.mean() <= p0.mean()


def test_calibration_table_shape():
    y = np.array([0, 1] * 50)
    p = np.linspace(0.01, 0.99, 100)
    t = calibration_table(y, p)
    assert len(t) == 10 and t["n"].sum() == 100


def test_gbm_frame_has_only_declared_features(nb_small, cfg):
    X = gbm_frame(nb_small, cfg["market.n_competitors"])
    assert list(X.columns) == GBM_FEATURES
