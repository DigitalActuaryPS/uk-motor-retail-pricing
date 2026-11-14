"""Build README.md and docs/executive_summary.md from executed outputs.

Every number inserted into prose goes through Registry.put, which records the formatted
string with its source. scripts/verify.py re-derives the values and checks the text.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from ..core import paths
from ..core.config import load_config

SIM = "simulated_data"


class Registry:
    def __init__(self) -> None:
        self.items: dict[str, dict] = {}

    def put(self, key: str, value: float, fmt: str, source: str) -> str:
        text = format(value, fmt)
        self.items[key] = {"text": text, "value": float(value), "source": source}
        return text


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(paths.data_processed() / name)


def _js(name: str) -> dict:
    return json.loads((paths.data_processed() / name).read_text())


def test_counts() -> dict:
    p = paths.data_processed() / "pytest_junit.xml"
    if not p.exists():
        return {"tests": None, "failures": None, "errors": None, "skipped": None}
    root = ET.parse(p).getroot()
    s = root if root.tag == "testsuite" else root.find("testsuite")
    return {k: int(s.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}


def md_table(df: pd.DataFrame, fmts: dict[str, str] | None = None) -> str:
    fmts = fmts or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [format(r[c], fmts[c]) if c in fmts else str(r[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def collect(reg: Registry) -> dict[str, str]:
    """Return every formatted figure used in the prose, registering each one."""
    cfg = load_config()
    m1, m3, m4, m5, m6, m8 = (_js(f"m{i}_summary.json") for i in (1, 3, 4, 5, 6, 8))
    imp = _csv("m5_baseline_vs_optimised__simulated_data.csv")
    cal = imp[imp["optimised_with"] == "calibrated"]
    mb = cal[(cal["evaluated_on"] == "model") & (cal["scenario"] == "baseline")].iloc[0]
    mo = cal[(cal["evaluated_on"] == "model") & (cal["scenario"] == "optimised")].iloc[0]
    tb = cal[(cal["evaluated_on"] == "true_process") & (cal["scenario"] == "baseline")].iloc[0]
    to = cal[(cal["evaluated_on"] == "true_process") & (cal["scenario"] == "optimised")].iloc[0]
    seg = _csv("m5_segment_decisions_calibrated__simulated_data.csv")
    cap = cfg["optimisation.renewal_price_change_cap"]
    ds = _csv("m6_decision_set_comparison__simulated_data.csv").set_index("decision_set")
    m2 = _js("m2_summary.json")
    t = test_counts()
    P = reg.put
    v: dict[str, str] = {}
    v["rows"] = P("rows", m1["rows"], ",.0f", "m1_summary.json")
    v["nb_rows"] = P("nb_rows", m1["new_business_rows"], ",.0f", "m1_summary.json")
    v["ren_rows"] = P("ren_rows", m1["renewal_rows"], ",.0f", "m1_summary.json")
    v["test_rows"] = P("test_rows", m1["randomised_test_rows"], ",.0f", "m1_summary.json")
    v["nb_conv"] = P("nb_conv", m1["nb_conversion_rate"], ".1%", "m1_summary.json")
    v["ren_rate"] = P("ren_rate", m1["renewal_rate"], ".1%", "m1_summary.json")
    v["dq_pass"] = P("dq_pass", m1["quality_checks_passed"], ".0f", "m1_summary.json")
    v["dq_total"] = P("dq_total", m1["quality_checks_total"], ".0f", "m1_summary.json")
    v["sha"] = m1["sha256"][:16]
    reg.items["sha"] = {"text": v["sha"], "value": 0.0, "source": "m1_summary.json"}
    v["conv_r1"] = P("conv_r1", m2["conversion_rank1"], ".1%", "m2_summary.json")
    v["conv_r9"] = P("conv_r9", m2["conversion_rank9"], ".1%", "m2_summary.json")
    for k in ("glm", "gbm", "naive"):
        v[f"auc_{k}"] = P(f"auc_{k}", m3["test_auc"][k], ".3f", "m3_summary.json")
        v[f"ll_{k}"] = P(f"ll_{k}", m3["test_log_loss"][k], ".3f", "m3_summary.json")
    el = m4["elasticity_overall"]
    for k in ("true", "glm_observational", "gbm_observational", "experiment", "ci_low", "ci_high", "glm_with_latent_intent"):
        v[f"el_{k}"] = P(f"el_{k}", el[k], ".2f", "m4_summary.json")
    v["corr_u"] = P("corr_u", m4["endogeneity"]["corr_ln_ratio_intent"], ".2f", "m4_summary.json")
    br = m4["elasticity_by_channel"]["broker"]
    v["br_lo"] = P("br_lo", br["ci_low"], ".2f", "m4_summary.json")
    v["br_hi"] = P("br_hi", br["ci_high"], ".2f", "m4_summary.json")
    v["nb_pol_chg"] = P("nb_pol_chg", mo["nb_policies"] / mb["nb_policies"] - 1, "+.1%", "m5 model")
    v["nb_conv_b"] = P("nb_conv_b", mb["nb_conversion"], ".2%", "m5 model")
    v["nb_conv_o"] = P("nb_conv_o", mo["nb_conversion"], ".2%", "m5 model")
    v["gwp_chg"] = P("gwp_chg", mo["total_gwp"] / mb["total_gwp"] - 1, "+.2%", "m5 model")
    v["contrib_chg"] = P("contrib_chg", mo["total_contribution"] / mb["total_contribution"] - 1, "+.1%", "m5 model")
    v["contrib_chg_true"] = P("contrib_chg_true", to["total_contribution"] / tb["total_contribution"] - 1, "+.1%", "m5 true")
    v["gwp_chg_true"] = P("gwp_chg_true", to["total_gwp"] / tb["total_gwp"] - 1, "+.2%", "m5 true")
    v["nb_pol_chg_true"] = P("nb_pol_chg_true", to["nb_policies"] / tb["nb_policies"] - 1, "+.1%", "m5 true")
    v["margin_b"] = P("margin_b", mb["margin"], ".1%", "m5 model")
    v["margin_o"] = P("margin_o", mo["margin"], ".1%", "m5 model")
    v["contrib_abs"] = P("contrib_abs", mo["total_contribution"] - mb["total_contribution"], ",.0f", "m5 model")
    v["n_disc"] = P("n_disc", (seg["nb_action"] == "discount").sum(), ".0f", "m5 segments")
    v["disc_list"] = ", ".join(seg.loc[seg["nb_action"] == "discount", "segment"].str.replace("_", " "))
    reg.items["disc_list"] = {"text": v["disc_list"], "value": 0.0, "source": "m5 segments"}
    v["n_inc"] = P("n_inc", (seg["nb_action"] == "increase").sum(), ".0f", "m5 segments")
    v["n_hold"] = P("n_hold", (seg["nb_action"] == "hold").sum(), ".0f", "m5 segments")
    v["n_ren_cap"] = P("n_ren_cap", (seg["renewal_avg_change_opt"] >= cap - 0.002).sum(), ".0f", "m5 segments")
    v["ren_cap"] = P("ren_cap", cap, ".0%", "config")
    v["ren_y_min"] = P("ren_y_min", seg["renewal_price_change"].min(), "+.1%", "m5 segments")
    v["ren_y_max"] = P("ren_y_max", seg["renewal_price_change"].max(), "+.1%", "m5 segments")
    v["ren_avg_min"] = P("ren_avg_min", seg["renewal_avg_change_opt"].min(), ".1%", "m5 segments")
    v["ren_avg_max"] = P("ren_avg_max", seg["renewal_avg_change_opt"].max(), ".1%", "m5 segments")
    v["n_bind"] = P("n_bind", len(m5["binding_constraints"]), ".0f", "m5_summary.json")
    v["opt_iters"] = P("opt_iters", m5["iterations"]["calibrated"], ".0f", "m5_summary.json")
    v["gain_obs_true"] = P("gain_obs_true", m5["contribution_gain_observational"]["true_process"], ".1%", "m5_summary.json")
    v["clv_mean"] = P("clv_mean", m6["mean_clv_per_policy"], ",.0f", "m6_summary.json")
    v["fy_mean"] = P("fy_mean", m6["mean_first_year_contribution_per_policy"], ",.0f", "m6_summary.json")
    v["clv_diff"] = P("clv_diff", m6["segments_with_different_action"], ".0f", "m6_summary.json")
    v["clv_nb_pol"] = P("clv_nb_pol", ds.loc["clv_optimum", "nb_policies"] / ds.loc["baseline", "nb_policies"] - 1, "+.1%", "m6 decision sets")
    v["clv_total_chg"] = P("clv_total_chg", ds.loc["clv_optimum", "nb_total_clv"] / ds.loc["baseline", "nb_total_clv"] - 1, "+.1%", "m6 decision sets")
    v["fy_total_chg"] = P("fy_total_chg", ds.loc["first_year_optimum", "nb_total_clv"] / ds.loc["baseline", "nb_total_clv"] - 1, "+.1%", "m6 decision sets")
    v["clv_fy_contrib_chg"] = P("clv_fy_contrib_chg", ds.loc["clv_optimum", "total_first_year_contribution"] / ds.loc["baseline", "total_first_year_contribution"] - 1, "+.1%", "m6 decision sets")
    v["oot_lo"] = P("oot_lo", m8["oot_glm_auc_min"], ".3f", "m8_summary.json")
    v["oot_hi"] = P("oot_hi", m8["oot_glm_auc_max"], ".3f", "m8_summary.json")
    v["psi_max"] = P("psi_max", m8["psi_max"], ".3f", "m8_summary.json")
    v["stress_min"] = P("stress_min", m8["stress_min_gain"], ".1%", "m8_summary.json")
    v["stress_n"] = P("stress_n", m8["stress_scenarios"], ".0f", "m8_summary.json")
    v["stress_base_below"] = P("stress_base_below", m8["stress_baseline_below_margin_floor"], ".0f", "m8_summary.json")
    v["tests"] = P("tests", t["tests"] or 0, ".0f", "pytest_junit.xml")
    v["test_fail"] = P("test_fail", (t["failures"] or 0) + (t["errors"] or 0), ".0f", "pytest_junit.xml")
    v["test_skip"] = P("test_skip", t["skipped"] or 0, ".0f", "pytest_junit.xml")
    return v


def decisions_table() -> str:
    seg = _csv("m5_segment_decisions_calibrated__simulated_data.csv")
    df = pd.DataFrame({
        "Segment": seg["segment"].str.replace("_", " "),
        "NB price": seg["nb_price_change"],
        "NB action": seg["nb_action"],
        "NB conversion before": seg["nb_conversion_base"],
        "NB conversion after": seg["nb_conversion_opt"],
        "Renewal avg change": seg["renewal_avg_change_opt"],
    })
    return md_table(df, {"NB price": "+.1%", "NB conversion before": ".1%", "NB conversion after": ".1%", "Renewal avg change": "+.1%"})


def render_exec(v: dict[str, str]) -> str:
    return f"""# Executive summary for the Pricing Director

**Simulated data.** All figures come from a simulated UK motor quote market built from illustrative assumptions. They show how the method behaves. They are not real insurer experience and not a forecast.

## Decision

Move new business and renewal prices by segment within plus or minus 5% of baseline. The optimiser discounts some segments and raises the others slightly. Renewal prices rise by {v['ren_y_min']} to {v['ren_y_max']} above baseline, and the average renewal change against last year's premium stays within the {v['ren_cap']} cap. On the simulated test-period book, the recommended set of moves:

| Effect (versus baseline prices) | Model view | Simulated true process |
|---|---|---|
| New business policies | {v['nb_pol_chg']} | {v['nb_pol_chg_true']} |
| Total GWP (new business and renewal) | {v['gwp_chg']} | {v['gwp_chg_true']} |
| Total contribution | {v['contrib_chg']} (about {v['contrib_abs']} in the simulated book) | {v['contrib_chg_true']} |
| Portfolio margin | {v['margin_b']} to {v['margin_o']} | not tabulated |

New business conversion falls from {v['nb_conv_b']} to {v['nb_conv_o']} on the model view. The plan buys margin with volume, inside the volume floor.

By segment: {v['n_disc']} segments get a new business discount ({v['disc_list']}), {v['n_inc']} get an increase and {v['n_hold']} hold. Average renewal change against last year's premium runs from {v['ren_avg_min']} to {v['ren_avg_max']} across segments, and {v['n_ren_cap']} segments sit at the cap, so the cap rather than the economics limits those. Full table below.

## Why the answer is credible and where it is not

- Observational elasticity is understated. The overall elasticity is {v['el_true']} in the simulated truth. A conversion GLM fitted to observational quotes gives {v['el_glm_observational']} and the GBM gives {v['el_gbm_observational']}, because our price correlates with unobserved customer intent (correlation {v['corr_u']}). The randomised price test recovers {v['el_experiment']} (95% interval {v['el_ci_low']} to {v['el_ci_high']}). Optimising on the uncalibrated observational GLM instead would have given a true-process contribution gain of {v['gain_obs_true']}, against {v['contrib_chg_true']} for the calibrated model. The difference is small here, because the rank and margin constraints bind in both cases.
- The optimiser uses the experiment-calibrated model. The truth-based gain ({v['contrib_chg_true']}) is lower than the model-based gain ({v['contrib_chg']}), which is the honest range.
- The broker experiment is too small to estimate a broker elasticity (95% interval {v['br_lo']} to {v['br_hi']}). Direct and broker were pooled for calibration.
- Cost per policy is a fixed configuration input and never fitted. Margin and contribution depend on it directly.
- {v['n_bind']} constraints bind at the solution, mostly rank-deterioration caps. Changing the cap changes the answer.

## Does a lifetime view change the decision?

Yes. Mean CLV per new policy is {v['clv_mean']} against a first-year contribution of {v['fy_mean']}. {v['clv_diff']} of 15 segments take a different new business action when the objective is CLV instead of first-year contribution. The CLV-optimal set raises new business volume by {v['clv_nb_pol']} and new business CLV by {v['clv_total_chg']} (first-year set: {v['fy_total_chg']}), at a first-year contribution change of {v['clv_fy_contrib_chg']}. The trade is short-term profit for volume that is expected to renew. It rests on renewal retention assumptions and on premium and cost growth set in the configuration.

## Robustness

- Out-of-time monthly ROC-AUC of the conversion GLM ranges from {v['oot_lo']} to {v['oot_hi']}.
- Largest population stability index across checked features and the score: {v['psi_max']} (below the amber threshold of 0.10).
- Across {v['stress_n']} stress scenarios the optimised plan beats baseline contribution in every case, with a smallest gain of {v['stress_min']}. In {v['stress_base_below']} stress scenarios (the cost shifts and the combined case) the baseline is already below the margin floor, so a floor flag there is not caused by the plan.
- {v['dq_pass']} of {v['dq_total']} data quality checks pass on the simulated data.

## Segment decisions

{decisions_table()}

## Not in scope

No claims, frequency, severity or loss-cost modelling. No risk rating. No compliance demonstration: the renewal cap and the renewal versus new business comparison are illustrative controls and are not an ENBP test. See `docs/regulatory_notes.md`.
"""


def render_readme(v: dict[str, str]) -> str:
    m3 = _csv("m3_metrics__simulated_data.csv")
    m3 = m3[m3["split"] == "test"][["model", "roc_auc", "log_loss", "brier"]]
    scen = _csv("m4_scenarios__simulated_data.csv")
    scen = scen[scen["source"] == "glm_experiment_calibrated"][["price_change", "conversion_probability", "gwp", "contribution", "margin"]]
    cmp_ = _csv("m4_elasticity_comparison__simulated_data.csv")
    cmp_ = cmp_[cmp_["group"].isin(["overall", "channel:PCW", "channel:direct", "channel:broker"])][["group", "true", "glm_observational", "gbm_observational", "experiment", "ci_low", "ci_high"]]
    return f"""# UK Motor Insurance: Retail Pricing, Customer Behaviour and Optimisation

> **Simulated data.** Every customer-level record is generated by code from illustrative assumptions in `config/calibration.yaml`. Nothing here is real insurer experience, and nothing here demonstrates regulatory compliance. Parameters carry a status (`illustrative_assumption` by default) and an empty source note unless a public source was supplied.

A retail pricing portfolio project for a UK motor insurer selling through price comparison websites (PCW), direct and broker. It builds a simulated quote market with a known truth, then covers competitor position, conversion modelling, elasticity with a randomised price test, constrained price optimisation, renewal and customer lifetime value, SQL analytics, an eight page Streamlit app and governance material.

**Out of scope on purpose:** claims frequency, severity, pure premium and loss-cost models, risk rating factors, rate tables, reserving and capital. Cost per policy is a fixed configuration value and is never fitted. No scraping or automated calls to comparison sites.

## Quick start

```bash
pip install -r requirements.txt
make all          # data, models, optimisation, outputs, tests, notebooks, reports, checks
make app          # streamlit run app/streamlit_app.py
```

Run order (also the Makefile order): `data`, `competitor`, `conversion`, `renewal-fit`, `elasticity`, `optimisation`, `retention`, `governance`, `sql`, `test`, `reports`, `notebooks`, `verify`. Each step reads the outputs of the earlier ones. Seed and sizes are in the configuration. The app starts without any manual data.

## Executed results

All numbers in this section are read from files in `data/processed/` by `src/retailpricing/reporting/build.py` and checked by `scripts/verify.py`.

**Data (M1).** {v['rows']} simulated quotes: {v['nb_rows']} new business and {v['ren_rows']} renewal. {v['test_rows']} new business rows are a randomised price test and are excluded from model fitting. New business conversion {v['nb_conv']}, renewal rate {v['ren_rate']}. {v['dq_pass']} of {v['dq_total']} data quality checks pass. Frame hash prefix `{v['sha']}`; a second run with the same seed gives the same hash.

**Competitor position (M2).** Conversion falls from {v['conv_r1']} at rank 1 to {v['conv_r9']} at rank 9.

**Conversion models (M3), test period.**

{md_table(m3, {"roc_auc": ".3f", "log_loss": ".3f", "brier": ".4f"})}

The GLM ({v['auc_glm']} AUC) and GBM ({v['auc_gbm']}) both beat the naive benchmark ({v['auc_naive']}). Time-based splits, tuning on validation only, SHAP, partial dependence and calibration are in `notebooks/03_conversion_models.ipynb`.

**Elasticity and endogeneity (M4).**

{md_table(cmp_, {c: ".2f" for c in ["true", "glm_observational", "gbm_observational", "experiment", "ci_low", "ci_high"]})}

Observational models understate elasticity because our price correlates with unobserved intent (correlation {v['corr_u']}). The randomised test recovers {v['el_experiment']} against a truth of {v['el_true']}. Scenario results from the experiment-calibrated GLM (new business test-period book):

{md_table(scen, {"price_change": "+.0%", "conversion_probability": ".2%", "gwp": ",.0f", "contribution": ",.0f", "margin": ".1%"})}

Adding the latent intent to the GLM gives {v['el_glm_with_latent_intent']}. The broker estimate is too noisy to use on its own (interval {v['br_lo']} to {v['br_hi']}).

**Optimisation (M5).** SLSQP, 30 variables, analytic Jacobians, {v['opt_iters']} iterations. On the test-period book: new business policies {v['nb_pol_chg']}, total GWP {v['gwp_chg']}, contribution {v['contrib_chg']} on the model and {v['contrib_chg_true']} against the simulated true process. Using the uncalibrated observational GLM gives a true-process gain of {v['gain_obs_true']}. {v['n_bind']} constraints bind.

**Retention and CLV (M6).** Mean CLV per new policy {v['clv_mean']} versus first-year contribution {v['fy_mean']}. {v['clv_diff']} of 15 segments change action under CLV.

**Governance (M8).** Out-of-time ROC-AUC {v['oot_lo']} to {v['oot_hi']}. Maximum PSI {v['psi_max']}. {v['stress_n']} stress scenarios, smallest contribution gain {v['stress_min']}.

**Tests.** {v['tests']} tests, {v['test_fail']} failed, {v['test_skip']} skipped.

See `docs/executive_summary.md` for the segment table and the decision write-up.

## Repository layout

| Path | Contents |
|---|---|
| `config/calibration.yaml` | Every parameter with `value`, `status`, `source_note` |
| `src/retailpricing/` | `data_gen`, `competitor`, `conversion`, `elasticity`, `optimisation`, `retention`, `analytics_sql`, `governance`, `manual_quotes`, `reporting`, `core` |
| `app/` | Streamlit app, eight pages including manual quote input |
| `notebooks/` | Eight executed notebooks, including the endogeneity explanation and the optimisation formulation in LaTeX |
| `docs/` | Architecture, data dictionary, regulatory notes, risk register, approval template, executive summary |
| `data/manual/` | Header-only quote template. The filled file is git-ignored and never used in modelling |
| `tests/` | pytest suite |

## Manual quote input

The app and `retailpricing.manual_quotes` accept quotes you collected yourself from a live comparison site session. Fewer than 30 quotes is reported as anecdotal. Manual quotes are never used for training, calibration or optimisation, and a test enforces the import isolation. Do not enter personal data.

## Limitations

- All data and every conclusion depend on illustrative assumptions. Read the results as a demonstration of method.
- The optimiser acts on segment averages of a single test-period book, not on individual quotes.
- Rank caps and the renewal cap bind in many segments, so those parameters drive the decision.
- The broker price test is too small for a stand-alone elasticity.
- Renewal retention beyond year one and the CLV growth rates are configuration inputs.
- Controls in `governance` and the renewal comparison are illustrative. They are not an ICOBS 6B equivalent new business price test. See `docs/regulatory_notes.md` for what was verified and what was not.
"""


def main() -> dict:
    reg = Registry()
    v = collect(reg)
    (paths.docs() / "executive_summary.md").write_text(render_exec(v), encoding="utf-8")
    (paths.root() / "README.md").write_text(render_readme(v), encoding="utf-8")
    (paths.docs() / "number_registry.json").write_text(json.dumps(reg.items, indent=1), encoding="utf-8")
    return {"numbers_registered": len(reg.items)}


if __name__ == "__main__":
    print(main())
