"""Price-change approval template and model-risk register. Simulated data only.

Both documents are illustrative project controls. They are not a demonstration of regulatory compliance.
"""
from __future__ import annotations

import pandas as pd

from ..core import paths

RISKS = [
    ("R01", "Data", "All customer-level data is simulated from illustrative assumptions", "High",
     "Results show method, not market truth. Any real use needs real quote, conversion and retention data.", "Label every output 'Simulated data'. Calibration status table. No claim of real experience.", "Open by design"),
    ("R02", "Elasticity", "Observational price effects are biased when price is set with information the model cannot see", "High",
     "Observational GLM and GBM elasticities are less negative than the truth (see Module 4 table).", "Randomised price test. Optimiser uses the experiment-calibrated response. Oracle model shows the cause.", "Mitigated in simulation. Needs a live test in practice"),
    ("R03", "Elasticity", "Randomised test is too thin to resolve small channels", "Medium",
     "Broker arm confidence interval spans zero. Calibration pools direct and broker.", "Pool thin channels. Report intervals. Increase test share or duration.", "Open"),
    ("R04", "Retention", "No randomised test for renewal price changes, so renewal elasticity is observational", "High",
     "Renewal response may be biased. Renewal moves in the optimiser rely on it.", "Cap renewal moves. Stress elasticity. Add a renewal test before use.", "Open"),
    ("R05", "Profitability", "Cost per policy is a fixed configuration value, not a loss cost model", "High",
     "Contribution is a proxy. Margin conclusions depend on the assumed cost.", "Excluded by design: no claims model. Cost shifts stress-tested. Config status flags every assumption.", "Accepted scope limit"),
    ("R06", "Optimisation", "SLSQP finds a local optimum on a piecewise smooth problem", "Medium",
     "Another start could give a different solution.", "Smoothed rank, analytic Jacobians checked against finite differences, local optimality test, constraint slack report.", "Mitigated"),
    ("R07", "Optimisation", "Rank is smoothed inside the optimiser but exact in evaluation", "Low",
     "Small gap between solver and evaluated constraints.", "Evaluate on exact rank. Report slack after exact evaluation.", "Mitigated"),
    ("R08", "Model", "Out-of-time drift in competitor position or mix", "Medium",
     "Conversion model degrades as the market moves.", "Monthly out-of-time metrics and PSI with amber and red thresholds.", "Monitored"),
    ("R09", "Model", "Gradient boosting can extrapolate poorly outside the observed price range", "Medium",
     "Price-response curves from GBM may be unreliable beyond the training range.", "Monotone constraints on price features. GLM used for optimisation. Curves compared with the truth.", "Mitigated"),
    ("R10", "Market data", "Competitor view limited to eight simulated competitors", "Medium",
     "Real PCW lists vary in length and include insurers we cannot see.", "Manual quote route for sense checks only. Never enters models.", "Open"),
    ("R11", "Customer outcomes", "Segment-level moves can hide adverse individual outcomes", "Medium",
     "Averages per segment do not guarantee fair outcomes for every customer.", "Renewal cap and rank guardrails. Illustrative renewal versus new business check. Compliance review outside this project.", "Open"),
    ("R12", "Governance", "Fair-value and conduct checks here are illustrative project controls", "High",
     "They do not demonstrate regulatory compliance.", "Stated in README, regulatory notes and this register. Independent review required before any real use.", "Open by design"),
    ("R13", "Manual input", "Hand-entered quotes are few, from one site and one time", "Low",
     "Fewer than 30 quotes is anecdotal and supports no inference.", "Sample size printed with every comparison. No use in calibration, training or optimisation.", "Mitigated"),
    ("R14", "Reproducibility", "Results depend on the seed and the simulation design", "Low",
     "Different seeds shift point estimates.", "Fixed seed, output hash check, determinism test.", "Mitigated"),
]


def risk_register_table() -> pd.DataFrame:
    return pd.DataFrame(RISKS, columns=["id", "area", "risk", "rating", "impact", "control", "status"])


def risk_register_markdown(df: pd.DataFrame) -> str:
    lines = ["# Model risk and limitations register", "",
             "**Simulated data.** Illustrative project register. It is not a demonstration of regulatory compliance and does not follow any named regulator template.", "",
             "| ID | Area | Risk | Rating | Impact | Control | Status |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r.id} | {r.area} | {r.risk} | {r.rating} | {r.impact} | {r.control} | {r.status} |" for r in df.itertuples()]
    return "\n".join(lines) + "\n"


def _md(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.get(c, lambda v: str(v))(r[c]) for c in cols) + " |")
    return "\n".join(out)


def approval_template(seg: pd.DataFrame, stress: pd.DataFrame, m5: dict) -> str:
    proc = paths.data_processed()
    imp = pd.read_csv(proc / "m5_baseline_vs_optimised__simulated_data.csv")
    imp = imp[(imp["optimised_with"] == "calibrated") & (imp["evaluated_on"] == "model")]
    cons = pd.read_csv(proc / "m5_constraints__simulated_data.csv")
    ctl = pd.read_csv(proc / "m6_illustrative_renewal_vs_nb_control__simulated_data.csv")
    pct = lambda v: f"{float(v):+.1%}"
    num = lambda v: f"{float(v):,.0f}"
    seg_t = seg[["segment", "nb_price_change", "renewal_price_change", "nb_conversion_base", "nb_conversion_opt", "nb_rank_base", "nb_rank_opt"]].copy()
    seg_t.columns = ["Segment", "NB change", "Renewal change", "NB conversion before", "NB conversion after", "NB rank before", "NB rank after"]
    seg_f = {"NB change": pct, "Renewal change": pct, "NB conversion before": lambda v: f"{v:.1%}", "NB conversion after": lambda v: f"{v:.1%}",
             "NB rank before": lambda v: f"{v:.2f}", "NB rank after": lambda v: f"{v:.2f}"}
    imp_t = imp[["scenario", "nb_policies", "nb_gwp", "nb_contribution", "renewal_policies", "renewal_gwp", "renewal_contribution", "total_gwp", "total_contribution", "margin"]]
    imp_f = {c: num for c in imp_t.columns if c not in ("scenario", "margin")}
    imp_f["margin"] = lambda v: f"{v:.1%}"
    bind = cons[cons["binding"]]["constraint"].tolist()
    st = stress[["scenario", "gain", "nb_policies_vs_baseline", "baseline_margin", "optimised_margin", "nb_volume_floor_met"]].copy()
    st_f = {"gain": pct, "nb_policies_vs_baseline": lambda v: f"{v:.1%}", "baseline_margin": lambda v: f"{v:.1%}", "optimised_margin": lambda v: f"{v:.1%}"}
    flagged = int((ctl["illustrative_flag"] != "ok").sum())
    return "\n".join([
        "# Price change approval template", "",
        "**Simulated data.** This template is an illustrative project control. Completing it does not demonstrate regulatory compliance. The tables below are filled from the simulated optimisation output so the layout can be reviewed.", "",
        "## 1. Change record", "",
        "| Field | Entry |", "|---|---|",
        "| Change reference | |", "| Product and channel | UK motor, all channels (simulated) |", "| Proposed effective date | |",
        "| Proposer (role) | |", "| Rationale in one sentence | |", "| Evidence base | Simulated data. Replace with real data and a live test before use |", "",
        "## 2. Proposed changes by segment", "", _md(seg_t, seg_f), "",
        "## 3. Expected impact (model basis, planning book of test-period quotes)", "", _md(imp_t, imp_f), "",
        "## 4. Constraint check", "",
        f"Optimiser converged: {m5['converged']['calibrated']}. Minimum constraint slack after solve: {m5['min_constraint_slack']['calibrated']:.2e}.",
        f"Binding constraints: {', '.join(bind) if bind else 'none'}.", "",
        "## 5. Stress tests", "", _md(st, st_f), "",
        "## 6. Customer outcome checks (illustrative)", "",
        f"- Renewal price change cap applied per segment: {m5.get('planning_book_renewal_quotes', 'n/a')} renewal offers in the planning book.",
        f"- Illustrative renewal versus new business check: {flagged} of {len(ctl)} segments have an average renewal price above the average new business price. Review each before approval.",
        "- Fair value and conduct review: to be completed by the relevant function. Not performed in this project.", "",
        "## 7. Known limitations to disclose", "",
        "- Renewal elasticity is observational, with no randomised renewal test.",
        "- Cost per policy is a fixed assumption, not a loss cost model.",
        "- See the model risk register for the full list.", "",
        "## 8. Sign-off", "",
        "| Role | Name | Decision | Date |", "|---|---|---|---|",
        "| Pricing actuary | | | |", "| Pricing director | | | |", "| Risk or compliance | | | |", "",
    ])
