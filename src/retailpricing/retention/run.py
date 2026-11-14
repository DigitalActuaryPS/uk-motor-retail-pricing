"""Module 6 runner: CLV, first-year versus CLV decisions, sensitivities. Simulated data only."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..core import paths
from ..core.config import load_config
from ..core.io import write_json
from ..core.plotting import finalise, save_fig
from ..optimisation.book import PlanningBook
from ..optimisation.responses import build_responses
from ..optimisation.run import ACTION_THRESHOLD, planning_data
from ..optimisation.solver import optimise
from ..retention.models import RenewalGLM
from .clv import make_clv_objective, make_retention_fn, nb_clv
from .features import RenewalDesign


def main() -> dict:
    warnings.filterwarnings("ignore")
    cfg = load_config()
    k = cfg["market.n_competitors"]
    proc = paths.data_processed()
    nbt, rnt, u_nb, u_rn = planning_data(cfg)
    book = PlanningBook(cfg, nbt, rnt)
    sm = build_responses(cfg, nbt, rnt, u_nb, u_rn, True)
    ex = build_responses(cfg, nbt, rnt, u_nb, u_rn, False)
    rglm = RenewalGLM.load(proc / "models" / "renewal_glm.json")
    S = book.S
    zero = np.zeros(S)

    # Baseline CLV by segment
    width = cfg["optimisation.rank_smoothing_width"]
    d_exact = RenewalDesign(nbt, k)
    rfn_exact = make_retention_fn(rglm.params, d_exact)
    price0 = nbt["our_premium"].to_numpy(dtype=float)
    clv0, detail = nb_clv(nbt, price0, rfn_exact, cfg, return_detail=True)
    econ_first = price0 - book.nb_cost  # first-year contribution per policy, after acquisition cost
    seg_idx = book.nb_seg
    base_tbl = pd.DataFrame({
        "segment": book.segments,
        "first_year_contribution_per_policy": np.bincount(seg_idx, weights=econ_first, minlength=S) / book.nb_count,
        "clv_per_policy": np.bincount(seg_idx, weights=clv0, minlength=S) / book.nb_count,
        "retention_year1_renewal": np.bincount(seg_idx, weights=detail["ret"][0], minlength=S) / book.nb_count,
        "expected_policy_years": np.bincount(seg_idx, weights=sum(detail["survival"]), minlength=S) / book.nb_count,
    })
    base_tbl["clv_to_first_year_multiple"] = base_tbl["clv_per_policy"] / base_tbl["first_year_contribution_per_policy"].where(base_tbl["first_year_contribution_per_policy"].abs() > 1e-9)
    base_tbl.to_csv(proc / "m6_clv_by_segment__simulated_data.csv", index=False)

    # Optimise: first-year (read M5 solution) versus CLV objective
    sol = np.load(proc / "m5_solution_calibrated.npy")
    x_fy, y_fy = sol[:S], sol[S:]
    objective, rfn_s = make_clv_objective(book, rglm.params, cfg)
    r_clv = optimise(book, sm["nb_calibrated"], sm["ren_model"], objective=objective, exact=(ex["nb_calibrated"], ex["ren_model"]))
    x_clv, y_clv = r_clv["x"], r_clv["y"]

    def total_clv(x, y, over=None, exact=True):
        resp = ex["nb_calibrated"]
        pn = book.nb_p0 * (1 + x[book.nb_seg])
        q = resp(pn)
        clv = nb_clv(nbt, pn, rfn_exact, cfg, **(over or {}))
        return float((q * clv).sum()), float(q.sum())

    cmp = pd.DataFrame({
        "segment": book.segments, "first_year_nb_change": x_fy, "clv_nb_change": x_clv,
        "first_year_renewal_change": y_fy, "clv_renewal_change": y_clv,
    })
    lab = lambda v: np.where(v > ACTION_THRESHOLD, "increase", np.where(v < -ACTION_THRESHOLD, "discount", "hold"))
    cmp["first_year_action"], cmp["clv_action"] = lab(x_fy), lab(x_clv)
    cmp["actions_differ"] = cmp["first_year_action"] != cmp["clv_action"]
    cmp = cmp.merge(base_tbl[["segment", "first_year_contribution_per_policy", "clv_per_policy"]], on="segment")
    cmp.to_csv(proc / "m6_first_year_vs_clv_decisions__simulated_data.csv", index=False)

    rows = []
    for lab_, (x, y) in {"baseline": (zero, zero), "first_year_optimum": (x_fy, y_fy), "clv_optimum": (x_clv, y_clv)}.items():
        v, pol = total_clv(x, y)
        ev = book.evaluate(x, y, ex["nb_calibrated"], ex["ren_model"], smooth=False)
        rows.append({"decision_set": lab_, "nb_policies": pol, "nb_total_clv": v, "nb_first_year_contribution": ev["nb_contrib"],
                     "total_first_year_contribution": ev["contrib"], "margin": ev["margin"]})
    pd.DataFrame(rows).to_csv(proc / "m6_decision_set_comparison__simulated_data.csv", index=False)

    # Sensitivities on CLV per policy (baseline prices)
    sens = []
    gov = cfg
    for dr in gov["governance.clv_discount_rates"]:
        sens.append({"case": f"discount_rate={dr:.0%}", "mean_clv_per_policy": float(nb_clv(nbt, price0, rfn_exact, cfg, discount_rate=dr).mean())})
    for rs in gov["governance.clv_retention_shifts"]:
        sens.append({"case": f"retention_shift={rs:+.0%}", "mean_clv_per_policy": float(nb_clv(nbt, price0, rfn_exact, cfg, retention_shift=rs).mean())})
    for cs in [0.0] + gov["governance.stress_cost_shifts"]:
        sens.append({"case": f"cost_shift={cs:+.0%}", "mean_clv_per_policy": float(nb_clv(nbt, price0, rfn_exact, cfg, cost_shift=cs).mean())})
    pd.DataFrame(sens).to_csv(proc / "m6_clv_sensitivity__simulated_data.csv", index=False)

    # Re-optimise under changed assumptions: do segment actions flip?
    cases = {"discount_rate=2%": {"discount_rate": 0.02}, "discount_rate=8%": {"discount_rate": 0.08},
             "retention_shift=-5%": {"retention_shift": -0.05}, "cost_shift=+10%": {"cost_shift": 0.10}}
    reopt = []
    for name, over in cases.items():
        obj, _ = make_clv_objective(book, rglm.params, cfg, **over)
        r = optimise(book, sm["nb_calibrated"], sm["ren_model"], objective=obj, maxiter=200, exact=(ex["nb_calibrated"], ex["ren_model"]))
        a = lab(r["x"])
        reopt.append({"case": name, "converged": r["success"], "segments_discount": int((a == "discount").sum()),
                      "segments_increase": int((a == "increase").sum()), "segments_hold": int((a == "hold").sum()),
                      "segments_changed_vs_clv_base": int((a != lab(x_clv)).sum()),
                      "mean_nb_change": float(r["x"].mean())})
    pd.DataFrame(reopt).to_csv(proc / "m6_clv_reoptimisation_sensitivity__simulated_data.csv", index=False)

    # Illustrative project control: renewal price against equivalent new business price (not a compliance test)
    seg_nb = pd.Series(book.nb_p0 * (1 + x_fy[book.nb_seg])).groupby(book.nb_seg).mean()
    seg_rn = pd.Series(book.ren_p0 * (1 + y_fy[book.ren_seg])).groupby(book.ren_seg).mean()
    ctl = pd.DataFrame({"segment": book.segments, "avg_new_business_price": seg_nb.to_numpy(), "avg_renewal_price": seg_rn.to_numpy()})
    ctl["renewal_to_new_business_ratio"] = ctl["avg_renewal_price"] / ctl["avg_new_business_price"]
    ctl["illustrative_flag"] = np.where(ctl["renewal_to_new_business_ratio"] > 1.0, "renewal above NB average", "ok")
    ctl.to_csv(proc / "m6_illustrative_renewal_vs_nb_control__simulated_data.csv", index=False)

    fig = go.Figure()
    fig.add_bar(y=cmp["segment"], x=cmp["first_year_nb_change"], orientation="h", name="First-year objective")
    fig.add_bar(y=cmp["segment"], x=cmp["clv_nb_change"], orientation="h", name="CLV objective")
    fig.update_xaxes(tickformat=".1%", title="New business price change")
    save_fig(finalise(fig, "Segment actions: first-year margin versus CLV", 560), "m6_first_year_vs_clv")

    dc = pd.DataFrame(rows).set_index("decision_set")
    out = {
        "mean_clv_per_policy": float(clv0.mean()), "mean_first_year_contribution_per_policy": float(econ_first.mean()),
        "clv_converged": bool(r_clv["success"]), "segments_with_different_action": int(cmp["actions_differ"].sum()),
        "segments_different": cmp.loc[cmp["actions_differ"], "segment"].tolist(),
        "clv_discount_segments": cmp.loc[cmp["clv_action"] == "discount", "segment"].tolist(),
        "clv_increase_segments": cmp.loc[cmp["clv_action"] == "increase", "segment"].tolist(),
        "nb_total_clv": {k_: float(v) for k_, v in dc["nb_total_clv"].items()},
        "total_first_year_contribution": {k_: float(v) for k_, v in dc["total_first_year_contribution"].items()},
        "renewal_above_nb_segments": int((ctl["illustrative_flag"] != "ok").sum()),
    }
    np.save(proc / "m6_solution_clv.npy", np.concatenate([x_clv, y_clv]))
    write_json(out, "m6_summary.json")
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2))
