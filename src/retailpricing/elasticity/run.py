"""Module 4 runner. Simulated data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..conversion.features import GLMDesign
from ..conversion.models import GBMModel, GLMModel
from ..core import paths
from ..core.config import load_config
from ..core.features import unit_economics
from ..core.io import load_quotes, load_truth, nb_experiment, nb_observational, time_split, write_json
from ..core.plotting import finalise, save_fig
from .core import (ScaledGLM, ShiftedGLM, arc_elasticity, arm_table, calibrate_shifts, elasticity_table, endogeneity_stats,
                   experiment_elasticities, oracle_glm_elasticity, scenario_table, true_fn)


def main() -> dict:
    import warnings

    warnings.filterwarnings("ignore")
    cfg = load_config()
    k = cfg["market.n_competitors"]
    step = cfg["modelling.elasticity_step"]
    df = load_quotes()
    truth = load_truth().set_index("quote_id")
    proc, mdir = paths.data_processed(), paths.data_processed() / "models"
    glm = GLMModel.load(mdir / "conversion_glm.json")
    gbm = GBMModel.load(mdir)

    nb = nb_observational(df)
    train, _, test = time_split(nb, cfg)
    exp = nb_experiment(df)
    u_test = truth.loc[test["quote_id"], "latent_intent"].to_numpy()
    u_train = truth.loc[train["quote_id"], "latent_intent"].to_numpy()
    u_exp = truth.loc[exp["quote_id"], "latent_intent"].to_numpy()

    d_test = GLMDesign(test, k)
    p0 = test["our_premium"].to_numpy(dtype=float)
    src_obs = {
        "true": true_fn(test, u_test, cfg),
        "glm_observational": lambda p: glm.predict(d_test, p),
        "gbm_observational": lambda p: gbm.predict(test, p),
    }
    obs_tbl = elasticity_table(src_obs, test, p0, step)

    # randomised test
    ex = experiment_elasticities(exp, cfg["modelling.bootstrap_resamples"], cfg["modelling.seed"])
    arms = arm_table(exp)
    ref0 = exp["reference_premium"].to_numpy(dtype=float)
    true_exp = {"overall": arc_elasticity(true_fn(exp, u_exp, cfg), ref0, None, step)}
    for c in ["PCW", "direct", "broker"]:
        true_exp[f"channel:{c}"] = arc_elasticity(true_fn(exp, u_exp, cfg), ref0, (exp["channel"].astype(str) == c).to_numpy(), step)
    for a in ["17-24", "25-34", "35-49", "50-64", "65+"]:
        true_exp[f"age:{a}"] = arc_elasticity(true_fn(exp, u_exp, cfg), ref0, (exp["age_band"].astype(str) == a).to_numpy(), step)
    ex["true_on_test_rows"] = ex["group"].map(true_exp)

    cmp = obs_tbl[obs_tbl["group"].isin(["overall", "channel:PCW", "channel:direct", "channel:broker"] + [f"age:{a}" for a in ["17-24", "25-34", "35-49", "50-64", "65+"]])]
    cmp = cmp.merge(ex[["group", "experiment", "ci_low", "ci_high", "true_on_test_rows"]], on="group", how="left")
    for m in ["glm_observational", "gbm_observational", "experiment"]:
        cmp[f"bias_{m}"] = cmp[m] - cmp["true"]
    oracle = oracle_glm_elasticity(train, u_train, test, u_test, k, step)
    cmp["glm_with_latent_intent"] = cmp["group"].map(oracle)
    cmp.to_csv(proc / "m4_elasticity_comparison__simulated_data.csv", index=False)
    obs_tbl.to_csv(proc / "m4_elasticity_by_group__simulated_data.csv", index=False)
    ex.to_csv(proc / "m4_experiment_elasticity__simulated_data.csv", index=False)
    arms.to_csv(proc / "m4_experiment_arms__simulated_data.csv", index=False)
    endo = endogeneity_stats(test, u_test)
    endo.update({"oracle_intent_coefficient": oracle["intent_coefficient"], "true_intent_coefficient": cfg["conversion_response.latent_intent_coefficient"]})

    # Calibrate the observational GLM to the randomised test. Direct and broker rows are pooled
    # (off-PCW) because the broker arm alone is too thin to resolve an elasticity.
    exi = ex.set_index("group")
    targets = {"PCW": float(exi.loc["channel:PCW", "experiment"]),
               "direct": float(exi.loc["channel:off_PCW", "experiment"]),
               "broker": float(exi.loc["channel:off_PCW", "experiment"])}
    shifts = calibrate_shifts(glm, d_test, test, p0, targets, step)
    delta_row = test["channel"].astype(str).map(shifts).to_numpy(dtype=float)
    scaled = ShiftedGLM(glm, d_test, delta_row, p0)
    write_json({"shift_by_channel": shifts, "target_elasticity_by_channel": targets}, "m4_calibration_shift.json")

    # scenarios
    changes = cfg["modelling.scenario_price_changes"]
    sc_sources = {"true": src_obs["true"], "glm_observational": src_obs["glm_observational"],
                  "gbm_observational": src_obs["gbm_observational"], "glm_experiment_calibrated": scaled.predict}
    sc = pd.concat([scenario_table(fn, test, cfg, changes).assign(source=n) for n, fn in sc_sources.items()])
    sc.to_csv(proc / "m4_scenarios__simulated_data.csv", index=False)

    # sensitivity tables
    sens_step = pd.DataFrame([{"step": s, **{n: arc_elasticity(fn, p0, None, s) for n, fn in sc_sources.items()}} for s in [0.005, 0.01, 0.02, 0.05]])
    sens_step.to_csv(proc / "m4_sensitivity_step__simulated_data.csv", index=False)
    mult_rows = []
    for mlt in cfg["governance.stress_elasticity_multipliers"] + [1.0]:
        sm_ = ScaledGLM(glm, d_test, np.full(len(test), mlt), p0)
        t = scenario_table(sm_.predict, test, cfg, [-0.02, 0.0, 0.02, 0.05]).assign(elasticity_multiplier=mlt)
        mult_rows.append(t)
    pd.concat(mult_rows).to_csv(proc / "m4_sensitivity_elasticity_multiplier__simulated_data.csv", index=False)
    cost_rows = []
    for shift in [0.0] + cfg["governance.stress_cost_shifts"]:
        econ = unit_economics(cfg, test)
        base = econ["unit_cost"] + econ["expense"] + econ["acquisition"]
        for x in [-0.05, 0.0, 0.05]:
            price = p0 * (1 + x)
            p = scaled.predict(price)
            cost_rows.append({"cost_shift": shift, "price_change": x, "contribution": float((p * (price - base * (1 + shift))).sum())})
    pd.DataFrame(cost_rows).to_csv(proc / "m4_sensitivity_cost__simulated_data.csv", index=False)

    # charts
    c = cmp[cmp["group"].isin(["overall", "channel:PCW", "channel:direct", "channel:broker"])]
    fig = go.Figure()
    for col, lab in [("true", "True"), ("glm_observational", "GLM (observational)"), ("gbm_observational", "LightGBM (observational)"), ("experiment", "Randomised test")]:
        kw = {}
        if col == "experiment":
            kw = dict(error_y=dict(type="data", symmetric=False, array=c["ci_high"] - c["experiment"], arrayminus=c["experiment"] - c["ci_low"]))
        fig.add_bar(x=c["group"], y=c[col], name=lab, **kw)
    fig.update_yaxes(title="Price elasticity of expected policies")
    save_fig(finalise(fig, "Elasticity: truth, observational models and randomised test"), "m4_elasticity_comparison")
    f2 = go.Figure()
    for n, g in sc.groupby("source"):
        f2.add_scatter(x=g["price_change"], y=g["contribution"], mode="lines+markers", name=n)
    f2.update_xaxes(tickformat=".0%", title="Uniform price change")
    f2.update_yaxes(title="Expected NB contribution, test period")
    save_fig(finalise(f2, "Scenario contribution by response source"), "m4_scenario_contribution")

    ov = cmp.set_index("group").loc["overall"]
    out = {
        "elasticity_overall": {k_: float(ov[k_]) for k_ in ["true", "glm_observational", "gbm_observational", "experiment", "ci_low", "ci_high", "glm_with_latent_intent"]},
        "elasticity_by_channel": {c_: {k_: float(cmp.set_index("group").loc[f"channel:{c_}", k_]) for k_ in ["true", "glm_observational", "gbm_observational", "experiment", "ci_low", "ci_high", "glm_with_latent_intent"]} for c_ in ["PCW", "direct", "broker"]},
        "endogeneity": endo, "calibration_shift": shifts, "calibration_targets": targets, "experiment_rows": int(len(exp)),
    }
    write_json(out, "m4_summary.json")
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2))
