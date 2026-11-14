"""Module 2: competitor and PCW position analysis. Simulated data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ..core.config import Config
from ..core.plotting import finalise


def add_position_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["median_gap"] = out["our_premium"] / out["comp_median"] - 1.0
    out["cheapest_gap"] = out["price_gap_to_cheapest"]
    out["pci"] = 100.0 * out["comp_median"] / out["our_premium"]
    out["ln_ratio"] = np.log(out["price_ratio_to_median"])
    return out


def summarise(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = df.groupby(keys, observed=True)
    res = g.agg(
        quotes=("converted", "size"),
        policies=("converted", "sum"),
        conversion=("converted", "mean"),
        avg_rank=("our_rank", "mean"),
        mean_median_gap=("median_gap", "mean"),
        mean_cheapest_gap=("cheapest_gap", "mean"),
        pci=("pci", "mean"),
        gwp=("written_premium", "sum"),
        contribution=("fixed_cost_contribution", "sum"),
    ).reset_index()
    res["margin"] = np.where(res["gwp"] > 0, res["contribution"] / res["gwp"], np.nan)
    res["contribution_per_quote"] = res["contribution"] / res["quotes"]
    return res


def flag_segments(seg: pd.DataFrame, df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    t = cfg["competitor_analysis.position_flag_threshold"]
    mean_ln = df.groupby("segment", observed=True)["ln_ratio"].mean().rename("mean_ln_ratio")
    out = seg.merge(mean_ln, left_on="segment", right_index=True)
    out["position_flag"] = np.where(out["mean_ln_ratio"] >= t, "overpriced",
                                    np.where(out["mean_ln_ratio"] <= -t, "underpriced", "in_line"))
    return out


def ratio_buckets(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 0.90, 0.95, 1.00, 1.05, 1.10, 10]
    labels = ["<0.90", "0.90-0.95", "0.95-1.00", "1.00-1.05", "1.05-1.10", ">=1.10"]
    d = df.copy()
    d["ratio_bucket"] = pd.cut(d["price_ratio_to_median"], bins=bins, labels=labels, right=False)
    return summarise(d, ["ratio_bucket"])


def run_analysis(nb: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    d = add_position_metrics(nb)
    seg = flag_segments(summarise(d, ["segment"]), d, cfg)
    return {
        "overall": summarise(d.assign(all="all"), ["all"]),
        "by_rank": summarise(d, ["our_rank"]),
        "by_channel": summarise(d, ["channel"]),
        "by_rank_channel": summarise(d, ["our_rank", "channel"]),
        "by_segment": seg,
        "by_segment_channel": summarise(d, ["segment", "channel"]),
        "by_ratio_bucket": ratio_buckets(d),
    }


def build_charts(res: dict[str, pd.DataFrame], nb: pd.DataFrame) -> dict[str, go.Figure]:
    figs = {}
    f = px.line(res["by_rank_channel"], x="our_rank", y="conversion", color="channel", markers=True,
                labels={"our_rank": "Our rank", "conversion": "Conversion rate"})
    f.update_yaxes(tickformat=".0%")
    figs["m2_conversion_by_rank_channel"] = finalise(f, "Conversion by rank and channel")

    seg = res["by_segment"].sort_values("mean_median_gap")
    colours = seg["position_flag"].map({"overpriced": "#b00020", "underpriced": "#1b6ca8", "in_line": "#888888"})
    f = go.Figure(go.Bar(x=seg["mean_median_gap"], y=seg["segment"], orientation="h", marker_color=colours))
    f.update_xaxes(tickformat=".1%", title="Mean gap to market median")
    figs["m2_segment_median_gap"] = finalise(f, "Segment price position (red overpriced, blue underpriced)", 520)

    r = res["by_rank"]
    f = go.Figure()
    f.add_bar(x=r["our_rank"], y=r["policies"], name="Policies written")
    f.add_scatter(x=r["our_rank"], y=r["contribution_per_quote"], name="Contribution per quote", yaxis="y2", mode="lines+markers")
    f.update_layout(yaxis2=dict(overlaying="y", side="right", title="Contribution per quote"), xaxis_title="Our rank", yaxis_title="Policies")
    figs["m2_rank_volume_contribution"] = finalise(f, "Market position drives volume and contribution")

    b = res["by_ratio_bucket"]
    f = go.Figure()
    f.add_bar(x=b["ratio_bucket"].astype(str), y=b["conversion"], name="Conversion")
    f.update_yaxes(tickformat=".0%")
    f.update_xaxes(title="Our price / market median")
    figs["m2_conversion_by_ratio_bucket"] = finalise(f, "Conversion by price ratio to market median")

    hm = nb.assign(age_band=nb["age_band"].astype(str), region=nb["region"].astype(str)).pivot_table(
        index="age_band", columns="region", values="converted", aggfunc="mean", observed=True)
    f = px.imshow(hm, text_auto=".1%", aspect="auto", color_continuous_scale="Blues", labels=dict(color="Conversion"))
    figs["m2_conversion_heatmap"] = finalise(f, "NB conversion by age band and region")
    return figs


def recommendations(res: dict[str, pd.DataFrame]) -> str:
    """Commercial recommendations. Every number is read from the computed tables."""
    ov = res["overall"].iloc[0]
    rk = res["by_rank"].set_index("our_rank")
    ch = res["by_channel"].set_index("channel")
    seg = res["by_segment"]
    over = seg[seg["position_flag"] == "overpriced"].sort_values("mean_median_gap", ascending=False)
    under = seg[seg["position_flag"] == "underpriced"].sort_values("mean_median_gap")
    lines = [
        "# Competitor and PCW recommendations",
        "",
        "**Simulated data.** Numbers below are produced by `retailpricing.competitor` on the simulated quote table.",
        "",
        f"Portfolio new business conversion is {ov['conversion']:.1%} on {int(ov['quotes']):,} quotes, "
        f"with an average rank of {ov['avg_rank']:.2f} and an average price competitiveness index of {ov['pci']:.1f} "
        "(100 means we match the market median).",
        "",
        "## Market position drives volume",
        f"Conversion falls from {rk.loc[1, 'conversion']:.1%} at rank 1 to {rk.loc[rk.index.max(), 'conversion']:.1%} at rank {int(rk.index.max())}. "
        f"Rank 1 to 3 quotes are {rk.loc[[1, 2, 3], 'quotes'].sum() / ov['quotes']:.1%} of volume but write "
        f"{rk.loc[[1, 2, 3], 'policies'].sum() / ov['policies']:.1%} of policies.",
        "",
        "## Channel",
    ]
    for c, r in ch.iterrows():
        lines.append(f"- {c}: conversion {r['conversion']:.1%}, average rank {r['avg_rank']:.2f}, first-year margin {r['margin']:.1%}.")
    lines += ["", "## Overpriced segments (mean gap to market median at or above the flag threshold)"]
    if over.empty:
        lines.append("None flagged.")
    for r in over.itertuples():
        lines.append(f"- {r.segment}: {r.mean_median_gap:+.1%} to median, conversion {r.conversion:.1%}, first-year margin {r.margin:.1%}. "
                     + ("Margin is above portfolio, so a test discount is the first action to try." if r.margin > ov["margin"]
                        else "Margin is at or below portfolio, so hold price and review cost before discounting."))
    lines += ["", "## Underpriced segments"]
    if under.empty:
        lines.append("None flagged.")
    for r in under.itertuples():
        lines.append(f"- {r.segment}: {r.mean_median_gap:+.1%} to median, conversion {r.conversion:.1%}, first-year margin {r.margin:.1%}. "
                     + ("Margin is below portfolio, so this segment is a candidate for an increase." if r.margin < ov["margin"]
                        else "Margin is at or above portfolio, so protect volume and monitor."))
    lines += ["", "These are directional. Module 4 shows observational price effects are biased, so size any move with the randomised test, then use Module 5."]
    return "\n".join(lines) + "\n"
