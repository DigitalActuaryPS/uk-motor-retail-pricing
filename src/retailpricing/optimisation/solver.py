"""Constrained optimisation with scipy.optimize (SLSQP). Simulated data only.

Maximise expected value over x (new business change per segment) and y (renewal change per segment).
Variables are expressed in percentage points inside the solver for conditioning.

Speed: segment aggregates depend only on their own segment's variable, so one forward-difference evaluation
with every variable perturbed together yields the full Jacobian. Tests check these Jacobians against
brute-force finite differences.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import minimize

from .book import PlanningBook, Resp

# objective(book, x, y, ev) -> (new business value per segment, renewal value per segment)
Objective = Callable[[PlanningBook, np.ndarray, np.ndarray, dict], tuple]


def first_year_objective(book: PlanningBook, x: np.ndarray, y: np.ndarray, ev: dict) -> tuple:
    return ev["nb_contrib_seg"], ev["ren_contrib_seg"]


def _jacobian_blocks(book: PlanningBook, ev0: dict, ev1: dict, base: dict, h: float):
    """Derivatives of segment aggregates with respect to their own variable (per unit of x)."""
    d = lambda k: (ev1[k] - ev0[k]) / h
    return {k: d(k) for k in ["nb_policies_seg", "nb_gwp_seg", "nb_contrib_seg", "ren_policies_seg", "ren_gwp_seg",
                              "ren_contrib_seg", "nb_rank_seg", "ren_rank_seg", "ren_change_seg"]}


def constraint_jacobian(book: PlanningBook, ev0: dict, blocks: dict, base: dict) -> np.ndarray:
    """Jacobian of book.constraints with respect to z = 100 * [x, y]."""
    S = book.S
    rows = []
    C, G = ev0["contrib"], ev0["gwp"]

    def margin_row():
        r = np.zeros(2 * S)
        dC_nb, dG_nb = blocks["nb_contrib_seg"], blocks["nb_gwp_seg"]
        dC_rn, dG_rn = blocks["ren_contrib_seg"], blocks["ren_gwp_seg"]
        r[:S] = (dC_nb * G - C * dG_nb) / G**2 * 10.0
        r[S:] = (dC_rn * G - C * dG_rn) / G**2 * 10.0
        return r

    rows.append(margin_row())
    r = np.zeros(2 * S); r[:S] = blocks["nb_policies_seg"] / base["nb_policies"]; rows.append(r)
    r = np.zeros(2 * S); r[S:] = blocks["ren_policies_seg"] / base["ren_policies"]; rows.append(r)
    for i in range(S):
        r = np.zeros(2 * S); r[i] = -blocks["nb_rank_seg"][i]; rows.append(r)
    for i in range(S):
        r = np.zeros(2 * S); r[S + i] = -blocks["ren_rank_seg"][i]; rows.append(r)
    for i in range(S):
        r = np.zeros(2 * S); r[S + i] = -blocks["ren_change_seg"][i] * 10.0; rows.append(r)
    return np.array(rows) / 100.0


def optimise(book: PlanningBook, nb_resp: Resp, ren_resp: Resp, objective: Objective = first_year_objective,
             z0: np.ndarray | None = None, maxiter: int = 400, h: float = 0.001, exact: tuple | None = None,
             exact_rounds: int = 4) -> dict:
    """Solve. If `exact=(nb_resp_exact, ren_resp_exact)` is given, constraints are re-checked with exact rank
    after each solve and tightened by any shortfall, so the returned solution is feasible on exact rank."""
    offset = np.zeros(len(book.constraints(book.evaluate(np.zeros(book.S), np.zeros(book.S), nb_resp, ren_resp),
                                           book.evaluate(np.zeros(book.S), np.zeros(book.S), nb_resp, ren_resp))[0]))
    rounds = exact_rounds if exact is not None else 0
    res = None
    for _ in range(rounds + 1):
        res = _solve(book, nb_resp, ren_resp, objective, z0, maxiter, h, offset)
        if exact is None:
            break
        S = book.S
        zero = np.zeros(S)
        base_e = book.evaluate(zero, zero, exact[0], exact[1], smooth=False)
        ev_e = book.evaluate(res["x"], res["y"], exact[0], exact[1], smooth=False)
        ve, _ = book.constraints(ev_e, base_e)
        res["exact_min_slack"] = float(ve.min())
        if ve.min() >= -1e-9:
            break
        offset = offset + np.maximum(0.0, -ve + 1e-3)
        z0 = np.concatenate([res["x"], res["y"]]) * 100.0
    res["constraint_offset_max"] = float(offset.max())
    return res


def _solve(book: PlanningBook, nb_resp: Resp, ren_resp: Resp, objective: Objective, z0, maxiter: int, h: float, offset: np.ndarray) -> dict:
    S = book.S
    cfg = book.cfg
    lo, hi = cfg["optimisation.price_change_lower"], cfg["optimisation.price_change_upper"]
    zero = np.zeros(S)
    base = book.evaluate(zero, zero, nb_resp, ren_resp, smooth=True)
    nb0, rn0 = objective(book, zero, zero, base)
    scale = float(nb0.sum() + rn0.sum())
    cache: dict = {}

    def point(z):
        key = z.tobytes()
        if key not in cache:
            if len(cache) > 50:
                cache.clear()
            x, y = z[:S] / 100.0, z[S:] / 100.0
            ev0 = book.evaluate(x, y, nb_resp, ren_resp, smooth=True)
            cache[key] = {"ev0": ev0, "x": x, "y": y, "obj": objective(book, x, y, ev0)}
        return cache[key]

    def with_jac(z):
        p = point(z)
        if "blocks" not in p:
            x1, y1 = p["x"] + h, p["y"] + h
            ev1 = book.evaluate(x1, y1, nb_resp, ren_resp, smooth=True)
            obj1 = objective(book, x1, y1, ev1)
            p["blocks"] = _jacobian_blocks(book, p["ev0"], ev1, base, h)
            p["dobj"] = ((obj1[0] - p["obj"][0]) / h, (obj1[1] - p["obj"][1]) / h)
        return p

    f = lambda z: -(point(z)["obj"][0].sum() + point(z)["obj"][1].sum()) / scale
    fprime = lambda z: -np.concatenate(with_jac(z)["dobj"]) / scale / 100.0
    g = lambda z: book.constraints(point(z)["ev0"], base, offset)[0]
    gjac = lambda z: constraint_jacobian(book, point(z)["ev0"], with_jac(z)["blocks"], base)

    z0 = np.zeros(2 * S) if z0 is None else z0
    res = minimize(f, z0, jac=fprime, method="SLSQP", bounds=[(lo * 100, hi * 100)] * (2 * S),
                   constraints=[{"type": "ineq", "fun": g, "jac": gjac}], options={"maxiter": maxiter, "ftol": 1e-10})
    z = res.x
    x, y = z[:S] / 100.0, z[S:] / 100.0
    ev_s = book.evaluate(x, y, nb_resp, ren_resp, smooth=True)
    vals, names = book.constraints(ev_s, base, offset)
    return {"x": x, "y": y, "success": bool(res.success), "message": str(res.message), "nit": int(res.nit),
            "objective_scaled": float(-res.fun), "min_constraint_slack": float(vals.min()),
            "constraint_names": names, "constraint_values": vals, "baseline_smooth": base}
