import numpy as np
import pandas as pd
import pytest

from retailpricing.core.config import Config, load_config
from retailpricing.data_gen.generator import frame_hash, generate
from retailpricing.data_gen.quality import run_quality_checks
from retailpricing.data_gen.truth import true_probability
from retailpricing.core.features import comp_columns, market_position


def test_determinism_fixed_seed(cfg):
    a, ta = generate(cfg, seed=5, n_quotes=3000)
    b, tb = generate(cfg, seed=5, n_quotes=3000)
    assert frame_hash(a) == frame_hash(b)
    assert frame_hash(ta) == frame_hash(tb)


def test_different_seed_differs(cfg):
    a, _ = generate(cfg, seed=5, n_quotes=3000)
    b, _ = generate(cfg, seed=6, n_quotes=3000)
    assert frame_hash(a) != frame_hash(b)


REQUIRED = ["quote_date", "customer_id", "business_type", "channel", "our_premium", "comp_min", "comp_median",
            "comp_mean", "our_rank", "price_ratio_to_median", "price_gap_to_cheapest", "tenure",
            "prior_year_premium", "segment", "pre_existing_risk_band", "treatment_group", "converted",
            "renewed_flag", "written_premium", "fixed_cost_contribution"]


def test_schema(small):
    df, _ = small
    for c in REQUIRED:
        assert c in df.columns
    assert set(df["channel"].astype(str)) == {"PCW", "direct", "broker"}
    assert df["segment"].nunique() == 15
    assert (df["data_label"] == "Simulated data").all()


def test_probability_bounds(small, cfg):
    df, truth = small
    assert truth["true_probability"].between(0, 1).all()
    assert df["converted"].isin([0, 1]).all()
    assert set(df["our_rank"].unique()) <= set(range(1, cfg["market.n_competitors"] + 2))


def test_truth_recomputes_from_frame(small, cfg):
    df, truth = small
    p = true_probability(df, truth["latent_intent"].to_numpy(), cfg)
    assert np.allclose(p, truth["true_probability"].to_numpy())


def test_quality_checks_all_pass(small, cfg):
    df, truth = small
    qc = run_quality_checks(df, cfg, truth, min_rows=1000)
    assert qc["passed"].all(), qc[~qc["passed"]]


def test_price_increase_reduces_true_probability(small, cfg):
    df, truth = small
    u = truth["latent_intent"].to_numpy()
    base = true_probability(df, u, cfg)
    up = true_probability(df, u, cfg, price=df["our_premium"].to_numpy() * 1.05)
    assert up.mean() < base.mean()


def test_rank_matches_manual_calculation():
    comp = np.array([[100.0, 110.0, 120.0, 130.0]])
    pos = market_position(np.array([115.0]), comp)
    assert pos["rank"][0] == 3
    assert pos["ratio"][0] == pytest.approx(115.0 / 115.0)
    assert pos["gap_cheapest"][0] == pytest.approx(0.15)


def test_config_every_parameter_has_status(cfg):
    n = 0
    for path, node in cfg.iter_params():
        assert set(node) >= {"value", "status", "source_note"}, path
        assert node["status"] in ("illustrative_assumption", "user_supplied_public")
        n += 1
    assert n > 60


def test_config_defaults_have_empty_source_note(cfg):
    assert (cfg.status_table()["status"] == "illustrative_assumption").all()
    assert (cfg.status_table()["source_note"] == "").all()


def test_config_rejects_missing_status():
    raw = {"s": {"p": {"value": 1, "source_note": ""}}}
    assert Config(raw).validate()
    raw = {"s": {"p": {"value": 1, "status": "illustrative_assumption", "source_note": ""}}}
    assert not Config(raw).validate()
    raw = {"s": {"p": {"value": 1, "status": "user_supplied_public", "source_note": ""}}}
    assert Config(raw).validate()
