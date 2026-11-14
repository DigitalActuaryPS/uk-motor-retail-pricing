"""Streamlit pages. Simulated data only."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import banner, config, csv, js, missing_outputs, pct, planning_book  # noqa: E402
from retailpricing.core import paths  # noqa: E402
from retailpricing.core.plotting import finalise  # noqa: E402


def executive_overview() -> None:
    st.header("Executive overview")
    banner()
    if missing_outputs("m1_summary.json", "m5_summary.json", "m5_baseline_vs_optimised__simulated_data.csv"):
        return
    m1, m5 = js("m1_summary.json"), js("m5_summary.json")
    imp = csv("m5_baseline_vs_optimised__simulated_data.csv")
    imp = imp[(imp["optimised_with"] == "calibrated") & (imp["evaluated_on"] == "model")].set_index("scenario")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Simulated quotes", f"{m1['rows']:,}")
    c2.metric("NB conversion", pct(m1["nb_conversion_rate"]))
    c3.metric("Renewal rate", pct(m1["renewal_rate"]))
    c4.metric("Average NB rank", f"{m1['avg_our_rank']:.2f}")
    st.subheader("Decision: baseline versus optimised (test-period planning book)")
    b, o = imp.loc["baseline"], imp.loc["optimised"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("NB policies", f"{o['nb_policies']:,.0f}", f"{o['nb_policies'] / b['nb_policies'] - 1:+.1%}")
    k2.metric("Total GWP", f"{o['total_gwp']:,.0f}", f"{o['total_gwp'] / b['total_gwp'] - 1:+.1%}")
    k3.metric("Contribution", f"{o['total_contribution']:,.0f}", f"{o['total_contribution'] / b['total_contribution'] - 1:+.1%}")
    k4.metric("Margin", pct(o["margin"]), f"{(o['margin'] - b['margin']) * 100:+.1f} pts")
    seg = csv("m5_segment_decisions_calibrated__simulated_data.csv")
    st.dataframe(seg[["segment", "nb_price_change", "renewal_price_change", "nb_action", "renewal_action"]]
                 .style.format({"nb_price_change": "{:+.1%}", "renewal_price_change": "{:+.1%}"}), use_container_width=True)
    st.caption("Constraints: margin floor, volume floors, rank deterioration cap, renewal price change cap. Cost per policy is a fixed configuration value.")


def competitor_position() -> None:
    st.header("Competitor and PCW position")
    banner()
    if missing_outputs("m2_by_rank_channel__simulated_data.csv"):
        return
    rc, seg, rk = csv("m2_by_rank_channel__simulated_data.csv"), csv("m2_by_segment__simulated_data.csv"), csv("m2_by_rank__simulated_data.csv")
    f = px.line(rc, x="our_rank", y="conversion", color="channel", markers=True)
    f.update_yaxes(tickformat=".0%")
    st.plotly_chart(finalise(f, "Conversion by rank and channel"), use_container_width=True)
    s = seg.sort_values("mean_median_gap")
    f2 = go.Figure(go.Bar(x=s["mean_median_gap"], y=s["segment"], orientation="h",
                          marker_color=s["position_flag"].map({"overpriced": "#b00020", "underpriced": "#1b6ca8", "in_line": "#888"})))
    f2.update_xaxes(tickformat=".1%", title="Mean gap to market median")
    st.plotly_chart(finalise(f2, "Segment price position", 520), use_container_width=True)
    st.subheader("Market position drives volume and contribution")
    st.dataframe(rk[["our_rank", "quotes", "conversion", "policies", "gwp", "contribution", "contribution_per_quote"]]
                 .style.format({"conversion": "{:.1%}", "gwp": "{:,.0f}", "contribution": "{:,.0f}", "contribution_per_quote": "{:.2f}"}), use_container_width=True)
    st.subheader("Flags")
    st.dataframe(seg[["segment", "mean_median_gap", "conversion", "margin", "position_flag"]]
                 .style.format({"mean_median_gap": "{:+.1%}", "conversion": "{:.1%}", "margin": "{:.1%}"}), use_container_width=True)
    rec = paths.docs() / "competitor_recommendations.md"
    if rec.exists():
        with st.expander("Written recommendations (numbers produced by code)"):
            st.markdown(rec.read_text())


def conversion_models() -> None:
    st.header("Conversion model performance")
    banner()
    if missing_outputs("m3_metrics__simulated_data.csv"):
        return
    met = csv("m3_metrics__simulated_data.csv")
    st.dataframe(met.style.format({c: "{:.4f}" for c in met.columns if c not in ("split", "model")}), use_container_width=True)
    cal = csv("m3_calibration__simulated_data.csv")
    f = go.Figure()
    mx = float(cal[["mean_predicted", "observed"]].to_numpy().max())
    f.add_scatter(x=[0, mx], y=[0, mx], name="Perfect", line=dict(dash="dot", color="grey"))
    for m, g in cal.groupby("model"):
        f.add_scatter(x=g["mean_predicted"], y=g["observed"], mode="lines+markers", name=m)
    f.update_xaxes(title="Mean predicted", tickformat=".0%")
    f.update_yaxes(title="Observed", tickformat=".0%")
    st.plotly_chart(finalise(f, "Calibration, test period"), use_container_width=True)
    c1, c2 = st.columns(2)
    shap = csv("m3_shap_importance__simulated_data.csv")
    c1.plotly_chart(finalise(px.bar(shap.sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature", orientation="h"), "Mean absolute SHAP"), use_container_width=True)
    pr = csv("m3_price_response__simulated_data.csv")
    g = pr[pr["group"] == "all"]
    f3 = go.Figure()
    for col in ["true", "glm", "gbm"]:
        f3.add_scatter(x=g["price_change"], y=g[col], name=col)
    f3.update_xaxes(tickformat=".0%")
    f3.update_yaxes(tickformat=".1%")
    c2.plotly_chart(finalise(f3, "Price response curves"), use_container_width=True)
    st.caption("The GLM and GBM are trained on observational data. The 'true' curve is the simulated process and exists only because the data is simulated.")


def elasticity_scenarios() -> None:
    st.header("Elasticity and scenario testing")
    banner()
    if missing_outputs("m4_elasticity_comparison__simulated_data.csv", "m4_scenarios__simulated_data.csv"):
        return
    cmp = csv("m4_elasticity_comparison__simulated_data.csv")
    show = cmp[cmp["group"].isin(["overall", "channel:PCW", "channel:direct", "channel:broker"])]
    st.subheader("Elasticity: truth, observational models, randomised test")
    st.dataframe(show[["group", "true", "glm_observational", "gbm_observational", "experiment", "ci_low", "ci_high", "glm_with_latent_intent"]]
                 .style.format({c: "{:.2f}" for c in ["true", "glm_observational", "gbm_observational", "experiment", "ci_low", "ci_high", "glm_with_latent_intent"]}), use_container_width=True)
    st.subheader("Uniform price scenarios")
    sc = csv("m4_scenarios__simulated_data.csv")
    st.dataframe(sc.style.format({"price_change": "{:+.0%}", "conversion_probability": "{:.2%}", "expected_policies": "{:,.0f}", "gwp": "{:,.0f}",
                                  "average_premium": "{:,.0f}", "contribution": "{:,.0f}", "margin": "{:.1%}"}), use_container_width=True)

    st.subheader("Segment price sliders (live KPIs)")
    try:
        book, resp = planning_book()
    except Exception as exc:  # data not generated yet
        st.info(f"Planning book unavailable. Run `make all`. ({type(exc).__name__})")
        return
    S = book.S
    sol_path = paths.data_processed() / "m5_solution_calibrated.npy"
    sol = np.load(sol_path) if sol_path.exists() else np.zeros(2 * S)
    c_a, c_b, c_c = st.columns(3)
    if c_a.button("Load optimiser solution"):
        for i in range(S):
            st.session_state[f"nb_{i}"] = float(np.round(sol[i] * 100, 1))
            st.session_state[f"rn_{i}"] = float(np.round(sol[S + i] * 100, 1))
    if c_b.button("Reset to zero"):
        for i in range(S):
            st.session_state[f"nb_{i}"] = 0.0
            st.session_state[f"rn_{i}"] = 0.0
    basis = c_c.radio("Evaluate on", ["Model (experiment-calibrated GLM)", "Simulated true process"], horizontal=True)
    st.markdown("**New business price change by segment (%)**")
    cols = st.columns(3)
    x = np.zeros(S)
    for i, s in enumerate(book.segments):
        x[i] = cols[i % 3].slider(s, -5.0, 5.0, st.session_state.get(f"nb_{i}", 0.0), 0.5, key=f"nb_{i}") / 100.0
    with st.expander("Renewal price change by segment (%)"):
        cols2 = st.columns(3)
        y = np.zeros(S)
        for i, s in enumerate(book.segments):
            y[i] = cols2[i % 3].slider(s, -5.0, 5.0, st.session_state.get(f"rn_{i}", 0.0), 0.5, key=f"rn_{i}") / 100.0
    nb_r, rn_r = (resp["nb_calibrated"], resp["ren_model"]) if basis.startswith("Model") else (resp["nb_true"], resp["ren_true"])
    zero = np.zeros(S)
    b = book.evaluate(zero, zero, nb_r, rn_r, smooth=False)
    e = book.evaluate(x, y, nb_r, rn_r, smooth=False)
    cfg = config()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("NB conversion", pct(e["nb_policies"] / len(book.nb_p0), 2), f"{(e['nb_policies'] - b['nb_policies']) / len(book.nb_p0) * 100:+.2f} pts")
    k2.metric("Total policies", f"{e['policies']:,.0f}", f"{e['policies'] - b['policies']:+,.0f}")
    k3.metric("GWP", f"{e['gwp']:,.0f}", f"{e['gwp'] / b['gwp'] - 1:+.2%}")
    k4.metric("Contribution", f"{e['contrib']:,.0f}", f"{e['contrib'] / b['contrib'] - 1:+.2%}")
    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Margin", pct(e["margin"]), f"{(e['margin'] - b['margin']) * 100:+.2f} pts")
    k6.metric("Mean NB rank", f"{np.average(e['nb_rank_seg'], weights=book.nb_count):.2f}", f"{np.average(e['nb_rank_seg'] - b['nb_rank_seg'], weights=book.nb_count):+.2f}")
    k7.metric("Max renewal avg change", pct(float(e["ren_change_seg"].max())), f"cap {pct(cfg['optimisation.renewal_price_change_cap'], 0)}")
    base_vals, names = book.constraints(e, b)
    k8.metric("Constraints met", "yes" if (base_vals >= -1e-9).all() else "no")
    if (base_vals < -1e-9).any():
        st.warning("Violated: " + ", ".join(n for n, v in zip(names, base_vals) if v < -1e-9))
    st.caption("Planning book is the test-period quotes. Volumes are for that period only. Simulated data.")


def optimisation_results() -> None:
    st.header("Optimisation results")
    banner()
    if missing_outputs("m5_summary.json"):
        return
    m5 = js("m5_summary.json")
    st.write(f"Solver converged: {m5['converged']['calibrated']}. Binding constraints: {len(m5['binding_constraints'])}.")
    seg = csv("m5_segment_decisions_calibrated__simulated_data.csv")
    f = go.Figure()
    f.add_bar(y=seg["segment"], x=seg["nb_price_change"], orientation="h", name="New business")
    f.add_bar(y=seg["segment"], x=seg["renewal_price_change"], orientation="h", name="Renewal")
    f.update_xaxes(tickformat=".1%")
    st.plotly_chart(finalise(f, "Optimised price change by segment", 560), use_container_width=True)
    st.subheader("Baseline versus optimised")
    st.dataframe(csv("m5_baseline_vs_optimised__simulated_data.csv").style.format(
        {c: "{:,.0f}" for c in ["nb_policies", "nb_gwp", "nb_contribution", "renewal_policies", "renewal_gwp", "renewal_contribution", "total_policies", "total_gwp", "total_contribution"]} | {"nb_conversion": "{:.2%}", "margin": "{:.1%}"}), use_container_width=True)
    st.caption("'true_process' rows evaluate the same decisions against the simulated data generating process. 'observational' rows use the uncalibrated GLM and show model risk.")
    with st.expander("Constraint slack"):
        st.dataframe(csv("m5_constraints__simulated_data.csv"), use_container_width=True)


def retention_clv() -> None:
    st.header("Retention and CLV")
    banner()
    if missing_outputs("m6_summary.json", "m6_renewal_metrics__simulated_data.csv"):
        return
    st.subheader("Renewal model, test and validation metrics")
    st.dataframe(csv("m6_renewal_metrics__simulated_data.csv").style.format({c: "{:.4f}" for c in ["roc_auc", "log_loss", "brier", "mean_prediction", "observed_rate"]}), use_container_width=True)
    st.subheader("CLV by segment (baseline prices)")
    st.dataframe(csv("m6_clv_by_segment__simulated_data.csv").style.format({c: "{:,.1f}" for c in ["first_year_contribution_per_policy", "clv_per_policy", "expected_policy_years", "clv_to_first_year_multiple"]} | {"retention_year1_renewal": "{:.1%}"}), use_container_width=True)
    cmp = csv("m6_first_year_vs_clv_decisions__simulated_data.csv")
    f = go.Figure()
    f.add_bar(y=cmp["segment"], x=cmp["first_year_nb_change"], orientation="h", name="First-year objective")
    f.add_bar(y=cmp["segment"], x=cmp["clv_nb_change"], orientation="h", name="CLV objective")
    f.update_xaxes(tickformat=".1%")
    st.plotly_chart(finalise(f, "New business action: first-year versus CLV", 560), use_container_width=True)
    st.dataframe(cmp[cmp["actions_differ"]][["segment", "first_year_action", "clv_action"]], use_container_width=True)
    st.subheader("Sensitivities")
    st.dataframe(csv("m6_clv_sensitivity__simulated_data.csv"), use_container_width=True)
    st.dataframe(csv("m6_clv_reoptimisation_sensitivity__simulated_data.csv"), use_container_width=True)
    st.info("Fair-value and conduct checks in this project (renewal cap, renewal versus new business comparison) are illustrative project controls. They are not a compliance demonstration.")
    st.dataframe(csv("m6_illustrative_renewal_vs_nb_control__simulated_data.csv"), use_container_width=True)


def governance() -> None:
    st.header("Governance and monitoring")
    banner()
    if missing_outputs("m8_summary.json"):
        return
    st.subheader("Data quality checks")
    dq = csv("m8_data_quality__simulated_data.csv")
    st.write(f"{int(dq['passed'].sum())} of {len(dq)} checks passed.")
    st.dataframe(dq, use_container_width=True)
    st.subheader("Out-of-time performance by month")
    oot = csv("m8_out_of_time_by_month__simulated_data.csv")
    f = px.line(oot, x="month", y="roc_auc", color="model", line_dash="book", markers=True)
    st.plotly_chart(finalise(f, "ROC-AUC by month, validation and test"), use_container_width=True)
    st.subheader("Population stability index")
    st.dataframe(csv("m8_psi__simulated_data.csv").style.format({"psi": "{:.4f}"}), use_container_width=True)
    st.subheader("Stress tests on the optimised decision")
    st.dataframe(csv("m8_stress_tests__simulated_data.csv"), use_container_width=True)
    st.subheader("Model risk and limitations register")
    st.dataframe(csv("m8_model_risk_register__simulated_data.csv"), use_container_width=True)
    p = paths.docs() / "price_change_approval_template.md"
    if p.exists():
        with st.expander("Price change approval template"):
            st.markdown(p.read_text())
    st.info("Controls here are illustrative project controls. They do not demonstrate regulatory compliance.")


def manual_input() -> None:
    from retailpricing.manual_quotes import quotes as mq

    st.header("Manual quote input")
    st.info("Enter quotes you collected yourself from a live comparison-site session. Nothing here scrapes or calls any site. "
            "Do not enter names, dates of birth, registration numbers, addresses or postcodes. "
            "Manual quotes are never used for training, calibration or optimisation.")
    cfg = config()
    with st.form("manual_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        row = {
            "entry_date": c1.date_input("Entry date", value=date.today(), max_value=date.today()).isoformat(),
            "source_site": c2.text_input("Source site name"),
            "age_band": c1.selectbox("Age band", cfg["manual_input.age_bands"]),
            "region": c2.selectbox("Region", cfg["manual_input.regions"]),
            "vehicle_group": c1.selectbox("Vehicle group", cfg["manual_input.vehicle_groups"]),
            "no_claims_band": c2.selectbox("No-claims band", cfg["manual_input.no_claims_bands"]),
            "cover_type": c1.selectbox("Cover type", cfg["manual_input.cover_types"]),
            "annual_mileage_band": c2.selectbox("Annual mileage band", cfg["manual_input.mileage_bands"]),
            "insurer_name": c1.text_input("Insurer name as shown"),
            "quoted_annual_premium": c2.number_input("Quoted annual premium", min_value=0.0, value=0.0, step=1.0),
            "displayed_rank": c1.number_input("Displayed rank", min_value=0, value=0, step=1),
            "notes": c2.text_input("Notes (optional, no personal data)"),
        }
        submitted = st.form_submit_button("Validate and add")
    if submitted:
        n, report = mq.append_valid_rows(pd.DataFrame([row]), cfg=cfg)
        st.success("Row added.") if n else st.error("Row not added.")
        if len(report):
            st.dataframe(report, use_container_width=True)
    up = st.file_uploader("Or upload a CSV using the template columns", type="csv")
    if up is not None and st.button("Validate and append upload"):
        try:
            raw = pd.read_csv(up, dtype=str, keep_default_na=False)
            n, report = mq.append_valid_rows(raw, cfg=cfg)
            st.write(f"{n} valid row(s) appended.")
            if len(report):
                st.dataframe(report, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
    st.download_button("Download blank template", ",".join(mq.COLUMNS) + "\n", file_name="manual_quotes_template.csv")
    manual = mq.load_manual_quotes(cfg=cfg)
    st.subheader("Loaded manual quotes")
    st.write(f"Sample size: {len(manual)} quote(s).")
    if len(manual):
        st.dataframe(manual, use_container_width=True)
    if (paths.data_raw() / "simulated_quotes.parquet").exists():
        from retailpricing.core.io import load_quotes

        res = mq.compare_with_simulation(manual, load_quotes(), cfg)
        st.subheader("Comparison with the simulated competitor spread")
        st.text(mq.format_report(res))
    else:
        st.info("Simulated data not generated yet. Run `make data` to enable the comparison.")
