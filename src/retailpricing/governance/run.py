"""Module 8 runner: data quality, out-of-time, PSI, stress tests, approval template, risk register."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..conversion.features import GLMDesign
from ..conversion.models import GBMModel, GLMModel
from ..core import paths
from ..core.config import load_config
from ..core.features import comp_columns, unit_economics
from ..core.io import (load_quotes, load_truth, nb_observational, read_json, renewals, time_split, write_json)
from ..data_gen.quality import run_quality_checks
from ..optimisation.book import PlanningBook
from ..optimisation.responses import build_responses
from ..optimisation.run import planning_data
from ..retention.features import RenewalDesign
from ..retention.models import RenewalGLM, load_renewal_gbm
from .monitoring import elasticity_stress, feature_psi_table, monthly_out_of_time
from .register import approval_template, risk_register_markdown, risk_register_table


def stressed_book(cfg, nbt, rnt, u_nb, u_rn, comp_shift: float = 0.0, cost_shift: float = 0.0, elast_mult: float = 1.0):
    k = cfg["market.n_competitors"]
    cols = comp_columns(k)
    nb, rn = nbt.copy(), rnt.copy()
    for d in (nb, rn):
        d[cols] = d[cols].to_numpy() * (1 + comp_shift)
        d["comp_median"] = d["comp_median"] * (1 + comp_shift)
        d["comp_min"] = d["comp_min"] * (1 + comp_shift)
    book = PlanningBook(cfg, nb, rn)
    if cost_shift:
        for d, attr in ((nb, "nb_cost"), (rn, "ren_cost")):
            e = unit_economics(cfg, d)
            setattr(book, attr, getattr(book, attr) + cost_shift * (e["unit_cost"] + e["expense"]))
    ex = build_responses(cfg, nb, rn, u_nb, u_rn, False)
    nb_r, rn_r = ex["nb_calibrated"], ex["ren_model"]
    if elast_mult != 1.0:
        nb_r = elasticity_stress(nb_r, book.nb_p0, elast_mult)
        rn_r = elasticity_stress(rn_r, book.ren_p0, elast_mult)
    return book, nb_r, rn_r


def main() -> dict:
    warnings.filterwarnings("ignore")
    cfg = load_config()
    k = cfg["market.n_competitors"]
    proc = paths.data_processed()
    df = load_quotes()
    truth = load_truth()
    qc = run_quality_checks(df, cfg, truth)
    qc.to_csv(proc / "m8_data_quality__simulated_data.csv", index=False)

    # Out-of-time performance (validation and test months) and PSI, new business
    nb = nb_observational(df)
    train, val, test = time_split(nb, cfg)
    mdir = proc / "models"
    glm, gbm = GLMModel.load(mdir / "conversion_glm.json"), GBMModel.load(mdir)
    vt = pd.concat([val, test])
    preds = {"glm": glm.predict(GLMDesign(vt, k)), "gbm": gbm.predict(vt)}
    oot = monthly_out_of_time(vt, preds).assign(book="new_business")

    ren = renewals(df)
    rtr, rva, rte = time_split(ren, cfg)
    rglm, rgbm = RenewalGLM.load(mdir / "renewal_glm.json"), load_renewal_gbm(mdir)
    rvt = pd.concat([rva, rte])
    oot_r = monthly_out_of_time(rvt, {"glm": rglm.predict_df(rvt), "gbm": rgbm.predict(rvt)}).assign(book="renewal")
    pd.concat([oot, oot_r]).to_csv(proc / "m8_out_of_time_by_month__simulated_data.csv", index=False)

    amber, red = cfg["governance.psi_amber"], cfg["governance.psi_red"]
    psi_nb = feature_psi_table(train, test, k, glm.predict(GLMDesign(train, k)), glm.predict(GLMDesign(test, k)), amber, red).assign(book="new_business")
    from ..retention.features import renewal_frame
    from .monitoring import psi, rag

    rows = []
    for c in ["ln_change", "ln_ratio", "rank", "tenure"]:
        v = psi(renewal_frame(rtr, k)[c].to_numpy(), renewal_frame(rte, k)[c].to_numpy())
        rows.append({"item": c, "type": "feature", "psi": v, "rag": rag(v, amber, red), "book": "renewal"})
    v = psi(rglm.predict_df(rtr), rglm.predict_df(rte))
    rows.append({"item": "model_score", "type": "score", "psi": v, "rag": rag(v, amber, red), "book": "renewal"})
    psi_all = pd.concat([psi_nb, pd.DataFrame(rows)])
    psi_all.to_csv(proc / "m8_psi__simulated_data.csv", index=False)

    # Stress tests on the optimised decisions
    nbt, rnt, u_nb, u_rn = planning_data(cfg)
    sol = np.load(proc / "m5_solution_calibrated.npy")
    S = len(sol) // 2
    x, y = sol[:S], sol[S:]
    scenarios = [("central", 0.0, 0.0, 1.0)]
    for m in cfg["governance.stress_elasticity_multipliers"]:
        scenarios.append((f"elasticity x{m}", 0.0, 0.0, m))
    for s in cfg["governance.stress_competitor_price_shifts"]:
        scenarios.append((f"competitor prices {s:+.0%}", s, 0.0, 1.0))
    for s in cfg["governance.stress_cost_shifts"]:
        scenarios.append((f"cost per policy {s:+.0%}", 0.0, s, 1.0))
    mmax = max(cfg["governance.stress_elasticity_multipliers"])
    scenarios.append((f"combined: elasticity x{mmax}, competitors -5%, cost +10%", -0.05, 0.10, mmax))
    out_rows = []
    z = np.zeros(S)
    for name, cs, cost, em in scenarios:
        book, nb_r, rn_r = stressed_book(cfg, nbt, rnt, u_nb, u_rn, cs, cost, em)
        b = book.evaluate(z, z, nb_r, rn_r, smooth=False)
        o = book.evaluate(x, y, nb_r, rn_r, smooth=False)
        out_rows.append({
            "scenario": name, "baseline_contribution": b["contrib"], "optimised_contribution": o["contrib"],
            "gain": o["contrib"] / b["contrib"] - 1, "baseline_margin": b["margin"], "nb_policies_vs_baseline": o["nb_policies"] / b["nb_policies"],
            "renewal_policies_vs_baseline": o["ren_policies"] / b["ren_policies"], "optimised_margin": o["margin"],
            "baseline_margin_floor_met": bool(b["margin"] >= cfg["optimisation.min_portfolio_margin"]),
            "margin_floor_met": bool(o["margin"] >= cfg["optimisation.min_portfolio_margin"]),
            "nb_volume_floor_met": bool(o["nb_policies"] / b["nb_policies"] >= cfg["optimisation.min_volume_ratio"]),
            "max_nb_rank_deterioration": float((o["nb_rank_seg"] - b["nb_rank_seg"]).max()),
            "max_renewal_change": float(o["ren_change_seg"].max()),
        })
    stress = pd.DataFrame(out_rows)
    stress.to_csv(proc / "m8_stress_tests__simulated_data.csv", index=False)

    # Documents
    docs = paths.docs()
    seg = pd.read_csv(proc / "m5_segment_decisions_calibrated__simulated_data.csv")
    (docs / "price_change_approval_template.md").write_text(approval_template(seg, stress, read_json("m5_summary.json")), encoding="utf-8")
    reg = risk_register_table()
    reg.to_csv(proc / "m8_model_risk_register__simulated_data.csv", index=False)
    (docs / "model_risk_register.md").write_text(risk_register_markdown(reg), encoding="utf-8")

    last = oot[oot["model"] == "glm"]
    summary = {
        "quality_checks_passed": int(qc["passed"].sum()), "quality_checks_total": int(len(qc)),
        "oot_months": int(last["month"].nunique()),
        "oot_glm_auc_min": float(last["roc_auc"].min()), "oot_glm_auc_max": float(last["roc_auc"].max()),
        "oot_glm_actual_to_expected_min": float(last["actual_to_expected"].min()), "oot_glm_actual_to_expected_max": float(last["actual_to_expected"].max()),
        "psi_max": float(psi_all["psi"].max()), "psi_max_item": str(psi_all.loc[psi_all["psi"].idxmax(), "item"]),
        "psi_red_count": int((psi_all["rag"] == "red").sum()), "psi_amber_count": int((psi_all["rag"] == "amber").sum()),
        "stress_min_gain": float(stress["gain"].min()), "stress_min_gain_scenario": str(stress.loc[stress["gain"].idxmin(), "scenario"]),
        "stress_scenarios": int(len(stress)), "stress_all_gains_positive": bool((stress["gain"] > 0).all()),
        "stress_floor_breaches_caused_by_decision": int(((~stress["nb_volume_floor_met"]) | ((~stress["margin_floor_met"]) & stress["baseline_margin_floor_met"])).sum()),
        "stress_baseline_below_margin_floor": int((~stress["baseline_margin_floor_met"]).sum()),
    }
    write_json(summary, "m8_summary.json")
    return summary


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2))
