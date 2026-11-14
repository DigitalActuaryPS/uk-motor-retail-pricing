import copy

import numpy as np
import pandas as pd
import pytest

from retailpricing.core.config import Config
from retailpricing.core.io import nb_observational, renewals, time_split
from retailpricing.optimisation.book import PlanningBook
from retailpricing.optimisation.solver import optimise


def make_book(small, cfg, **overrides):
    raw = copy.deepcopy(cfg.raw)
    for k, v in overrides.items():
        sec, name = k.split(".")
        raw[sec][name]["value"] = v
    c = Config(raw)
    df, _ = small
    _, _, nbt = time_split(nb_observational(df), c)
    _, _, rnt = time_split(renewals(df), c)
    return PlanningBook(c, nbt, rnt), c


def responses(book, e_nb=-6.0, e_ren=-1.5):
    nb_p0, ren_p0 = book.nb_p0, book.ren_p0
    nb_resp = lambda pr: np.clip(0.09 * (pr / nb_p0) ** e_nb, 0, 1)
    ren_resp = lambda pr: np.clip(0.80 * (pr / ren_p0) ** e_ren, 0, 1)
    return nb_resp, ren_resp


def test_evaluate_hand_case(small, cfg):
    book, _ = make_book(small, cfg)
    nb_resp, ren_resp = responses(book)
    z = np.zeros(book.S)
    ev = book.evaluate(z, z, nb_resp, ren_resp, smooth=False)
    assert ev["nb_policies"] == pytest.approx(0.09 * len(book.nb_p0))
    assert ev["nb_gwp"] == pytest.approx((0.09 * book.nb_p0).sum())
    assert ev["nb_contrib"] == pytest.approx((0.09 * (book.nb_p0 - book.nb_cost)).sum())
    assert ev["margin"] == pytest.approx(ev["contrib"] / ev["gwp"])
    x = np.full(book.S, 0.02)
    ev2 = book.evaluate(x, z, nb_resp, ren_resp, smooth=False)
    assert ev2["nb_gwp"] == pytest.approx((0.09 * (1.02) ** -6.0 * book.nb_p0 * 1.02).sum())


def test_baseline_is_feasible_and_solution_respects_all_constraints(small, cfg):
    book, c = make_book(small, cfg, **{"optimisation.min_portfolio_margin": 0.10})
    nb_resp, ren_resp = responses(book)
    z = np.zeros(book.S)
    base = book.evaluate(z, z, nb_resp, ren_resp)
    assert (book.constraints(base, base)[0] >= -1e-9).all()
    r = optimise(book, nb_resp, ren_resp, maxiter=80)
    lo, hi = c["optimisation.price_change_lower"], c["optimisation.price_change_upper"]
    assert (r["x"] >= lo - 1e-9).all() and (r["x"] <= hi + 1e-9).all()
    assert (r["y"] >= lo - 1e-9).all() and (r["y"] <= hi + 1e-9).all()
    assert r["min_constraint_slack"] >= -1e-5
    ev = book.evaluate(r["x"], r["y"], nb_resp, ren_resp)
    assert ev["contrib"] >= base["contrib"] - 1e-6
    assert ev["nb_policies"] / base["nb_policies"] >= c["optimisation.min_volume_ratio"] - 1e-5
    assert ev["ren_policies"] / base["ren_policies"] >= c["optimisation.min_renewal_volume_ratio"] - 1e-5
    assert (ev["ren_change_seg"] <= c["optimisation.renewal_price_change_cap"] + 1e-5).all()
    d = c["optimisation.max_rank_deterioration"]
    assert ((ev["nb_rank_seg"] - base["nb_rank_seg"]) <= d + 1e-5).all()
    assert ((ev["ren_rank_seg"] - base["ren_rank_seg"]) <= d + 1e-5).all()


def test_tight_renewal_cap_binds(small, cfg):
    book0, _ = make_book(small, cfg)
    nb_resp0, ren_resp0 = responses(book0, e_ren=-0.2)
    z0 = np.zeros(book0.S)
    cap = float(book0.evaluate(z0, z0, nb_resp0, ren_resp0)["ren_change_seg"].max() + 0.004)
    book, c = make_book(small, cfg, **{"optimisation.renewal_price_change_cap": cap, "optimisation.min_portfolio_margin": 0.05,
                                      "optimisation.min_renewal_volume_ratio": 0.5, "optimisation.min_volume_ratio": 0.5,
                                      "optimisation.max_rank_deterioration": 5.0})
    nb_resp, ren_resp = responses(book, e_ren=-0.2)
    r = optimise(book, nb_resp, ren_resp, maxiter=80)
    ev = book.evaluate(r["x"], r["y"], nb_resp, ren_resp)
    assert (ev["ren_change_seg"] <= cap + 1e-5).all()
    # with an inelastic renewal book the cap should be what stops further increases
    assert ev["ren_change_seg"].max() >= cap - 1e-3


def test_local_optimality(small, cfg):
    book, c = make_book(small, cfg, **{"optimisation.min_portfolio_margin": 0.10})
    nb_resp, ren_resp = responses(book)
    r = optimise(book, nb_resp, ren_resp, maxiter=120)
    z = np.zeros(book.S)
    base = book.evaluate(z, z, nb_resp, ren_resp)
    best = book.evaluate(r["x"], r["y"], nb_resp, ren_resp)["contrib"]
    rng = np.random.default_rng(0)
    lo, hi = -0.05, 0.05
    for _ in range(40):
        x = np.clip(r["x"] + rng.normal(0, 0.004, book.S), lo, hi)
        y = np.clip(r["y"] + rng.normal(0, 0.004, book.S), lo, hi)
        ev = book.evaluate(x, y, nb_resp, ren_resp)
        vals, _ = book.constraints(ev, base)
        if (vals >= 0).all():
            assert ev["contrib"] <= best * (1 + 2e-3)


def test_analytic_jacobians_match_brute_force(small, cfg):
    from retailpricing.optimisation.solver import _jacobian_blocks, constraint_jacobian

    book, c = make_book(small, cfg, **{"optimisation.min_portfolio_margin": 0.10})
    nb_resp, ren_resp = responses(book)
    S = book.S
    rng = np.random.default_rng(2)
    x, y = rng.uniform(-0.02, 0.02, S), rng.uniform(-0.02, 0.02, S)
    z0 = np.zeros(S)
    base = book.evaluate(z0, z0, nb_resp, ren_resp)
    h = 1e-4
    ev0 = book.evaluate(x, y, nb_resp, ren_resp)
    ev1 = book.evaluate(x + h, y + h, nb_resp, ren_resp)
    J = constraint_jacobian(book, ev0, _jacobian_blocks(book, ev0, ev1, base, h), base)
    g0 = book.constraints(ev0, base)[0]
    z = np.concatenate([x, y]) * 100
    for j in [0, 3, S - 1, S, S + 4, 2 * S - 1]:
        zz = z.copy(); zz[j] += 0.01
        ev = book.evaluate(zz[:S] / 100, zz[S:] / 100, nb_resp, ren_resp)
        fd = (book.constraints(ev, base)[0] - g0) / 0.01
        assert np.allclose(J[:, j], fd, rtol=2e-2, atol=1e-6), j
