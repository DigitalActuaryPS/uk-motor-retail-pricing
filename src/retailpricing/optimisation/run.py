"""Module 5 runner. Simulated data only."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..core import paths
from ..core.config import load_config
from ..core.io import load_quotes, load_truth, nb_observational, renewals, time_split, write_json
from ..core.plotting import finalise, save_fig
from .book import PlanningBook
from .responses import build_responses
from .solver import optimise

ACTION_THRESHOLD = 0.005


def planning_data(cfg):
    df = load_quotes()
    truth = load_truth().set_index("quote_id")
    _, _, nbt = time_split(nb_observational(df), cfg)
    _, _, rnt = time_split(renewals(df), cfg)
    u_nb = truth.loc[nbt["quote_id"], "latent_intent"].to_numpy()
    u_rn = truth.loc[rnt["quote_id"], "latent_intent"].to_numpy()
    return nbt, rnt, u_nb, u_rn


def summarise_eval(ev: dict, label: str) -> dict:
    return {"scenario": label, "nb_conversion": ev["nb_policies"] / len(ev["nb_prob"]) if "nb_prob" in ev else np.nan,
            "nb_policies": ev["nb_policies"], "nb_gwp": ev["nb_gwp"], "nb_contribution": ev["nb_contrib"],
            "renewal_policies": ev["ren_policies"], "renewal_gwp": ev["ren_gwp"], "renewal_contribution": ev["ren_contrib"],
            "total_policies": ev["policies"], "total_gwp": ev["gwp"], "total_contribution": ev["contrib"], "margin": ev["margin"]}


def segment_table(book: PlanningBook, x, y, ev_base, ev_opt) -> pd.DataFrame:
    t = pd.DataFrame({
        "segment": book.segments, "nb_price_change": x, "renewal_price_change": y,
        "nb_quotes": book.nb_count,
        "nb_conversion_base": ev_base["nb_policies_seg"] / book.nb_count,
        "nb_conversion_opt": ev_opt["nb_policies_seg"] / book.nb_count,
        "nb_policies_base": ev_base["nb_policies_seg"], "nb_policies_opt": ev_opt["nb_policies_seg"],
        "nb_contribution_base": ev_base["nb_contrib_seg"], "nb_contribution_opt": ev_opt["nb_contrib_seg"],
        "nb_rank_base": ev_base["nb_rank_seg"], "nb_rank_opt": ev_opt["nb_rank_seg"],
        "renewal_policies_base": ev_base["ren_policies_seg"], "renewal_policies_opt": ev_opt["ren_policies_seg"],
        "renewal_contribution_base": ev_base["ren_contrib_seg"], "renewal_contribution_opt": ev_opt["ren_contrib_seg"],
        "renewal_avg_change_opt": ev_opt["ren_change_seg"],
    })
    t["nb_action"] = np.where(t["nb_price_change"] > ACTION_THRESHOLD, "increase", np.where(t["nb_price_change"] < -ACTION_THRESHOLD, "discount", "hold"))
    t["renewal_action"] = np.where(t["renewal_price_change"] > ACTION_THRESHOLD, "increase", np.where(t["renewal_price_change"] < -ACTION_THRESHOLD, "discount", "hold"))
    return t


def main() -> dict:
    warnings.filterwarnings("ignore")
    cfg = load_config()
    nbt, rnt, u_nb, u_rn = planning_data(cfg)
    book = PlanningBook(cfg, nbt, rnt)
    sm, ex = build_responses(cfg, nbt, rnt, u_nb, u_rn, True), build_responses(cfg, nbt, rnt, u_nb, u_rn, False)
    zero = np.zeros(book.S)
    proc = paths.data_processed()

    runs = {}
    for name, nb_key in [("calibrated", "nb_calibrated"), ("observational", "nb_observational")]:
        runs[name] = optimise(book, sm[nb_key], sm["ren_model"], exact=(ex[nb_key], ex["ren_model"]))
        runs[name]["nb_key"] = nb_key

    rows = []
    seg_tables = {}
    for name, r in runs.items():
        x, y = r["x"], r["y"]
        for basis, nb_r, rn_r in [("model", ex[r["nb_key"]], ex["ren_model"]), ("true_process", ex["nb_true"], ex["ren_true"])]:
            b = book.evaluate(zero, zero, nb_r, rn_r, smooth=False)
            o = book.evaluate(x, y, nb_r, rn_r, smooth=False)
            if basis == "model":
                seg_tables[name] = segment_table(book, x, y, b, o)
            for lab, ev in [("baseline", b), ("optimised", o)]:
                s = summarise_eval(ev, lab)
                s["nb_conversion"] = ev["nb_policies"] / len(nbt)
                rows.append({"optimised_with": name, "evaluated_on": basis, **s})
    res = pd.DataFrame(rows)
    res.to_csv(proc / "m5_baseline_vs_optimised__simulated_data.csv", index=False)
    for n, t in seg_tables.items():
        t.to_csv(proc / f"m5_segment_decisions_{n}__simulated_data.csv", index=False)

    primary = runs["calibrated"]
    cons = pd.DataFrame({"constraint": primary["constraint_names"], "slack": primary["constraint_values"]})
    cons["binding"] = cons["slack"].abs() < 1e-3
    cons.to_csv(proc / "m5_constraints__simulated_data.csv", index=False)

    seg = seg_tables["calibrated"]
    fig = go.Figure()
    fig.add_bar(y=seg["segment"], x=seg["nb_price_change"], orientation="h", name="New business")
    fig.add_bar(y=seg["segment"], x=seg["renewal_price_change"], orientation="h", name="Renewal")
    fig.update_xaxes(tickformat=".1%", title="Optimised price change")
    save_fig(finalise(fig, "Optimised price change by segment", 560), "m5_segment_price_changes")

    t = res[(res["optimised_with"] == "calibrated")].set_index(["evaluated_on", "scenario"])
    gain = {b: float(t.loc[(b, "optimised"), "total_contribution"] / t.loc[(b, "baseline"), "total_contribution"] - 1) for b in ["model", "true_process"]}
    t2 = res[(res["optimised_with"] == "observational")].set_index(["evaluated_on", "scenario"])
    gain_obs = {b: float(t2.loc[(b, "optimised"), "total_contribution"] / t2.loc[(b, "baseline"), "total_contribution"] - 1) for b in ["model", "true_process"]}
    out = {
        "converged": {n: bool(r["success"]) for n, r in runs.items()},
        "iterations": {n: r["nit"] for n, r in runs.items()},
        "min_constraint_slack": {n: r["min_constraint_slack"] for n, r in runs.items()},
        "exact_min_constraint_slack": {n: r["exact_min_slack"] for n, r in runs.items()},
        "contribution_gain_calibrated": gain, "contribution_gain_observational": gain_obs,
        "binding_constraints": cons.loc[cons["binding"], "constraint"].tolist(),
        "nb_increase_segments": seg.loc[seg["nb_action"] == "increase", "segment"].tolist(),
        "nb_discount_segments": seg.loc[seg["nb_action"] == "discount", "segment"].tolist(),
        "nb_hold_segments": seg.loc[seg["nb_action"] == "hold", "segment"].tolist(),
        "planning_book_nb_quotes": int(len(nbt)), "planning_book_renewal_quotes": int(len(rnt)),
    }
    np.save(proc / "m5_solution_calibrated.npy", np.concatenate([primary["x"], primary["y"]]))
    np.save(proc / "m5_solution_observational.npy", np.concatenate([runs["observational"]["x"], runs["observational"]["y"]]))
    write_json(out, "m5_summary.json")
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2))
