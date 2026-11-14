"""Module 2 runner."""
from __future__ import annotations

from ..core import paths
from ..core.config import load_config
from ..core.io import load_quotes, nb_observational, write_json
from ..core.plotting import save_fig
from .analysis import add_position_metrics, build_charts, recommendations, run_analysis


def main() -> dict:
    cfg = load_config()
    nb = add_position_metrics(nb_observational(load_quotes()))
    res = run_analysis(nb, cfg)
    proc = paths.data_processed()
    for name, tbl in res.items():
        tbl.to_csv(proc / f"m2_{name}__simulated_data.csv", index=False)
    for name, fig in build_charts(res, nb).items():
        save_fig(fig, name)
    (paths.docs() / "competitor_recommendations.md").write_text(recommendations(res), encoding="utf-8")
    seg = res["by_segment"]
    out = {
        "overall_conversion": float(res["overall"]["conversion"].iloc[0]),
        "overall_margin": float(res["overall"]["margin"].iloc[0]),
        "overpriced_segments": seg.loc[seg["position_flag"] == "overpriced", "segment"].tolist(),
        "underpriced_segments": seg.loc[seg["position_flag"] == "underpriced", "segment"].tolist(),
        "conversion_rank1": float(res["by_rank"].set_index("our_rank").loc[1, "conversion"]),
        "conversion_rank9": float(res["by_rank"].set_index("our_rank").loc[9, "conversion"]),
    }
    write_json(out, "m2_summary.json")
    return out


if __name__ == "__main__":
    print(main())
