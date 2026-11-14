"""Module 1 runner: generate, check, write. Simulated data only."""
from __future__ import annotations

import json

import pandas as pd

from ..core import paths
from ..core.config import load_config
from .dictionary import to_markdown
from .generator import frame_hash, generate
from .quality import run_quality_checks


def main() -> dict:
    cfg = load_config()
    df, truth = generate(cfg)
    qc = run_quality_checks(df, cfg, truth)
    raw, proc = paths.data_raw(), paths.data_processed()
    df.to_parquet(raw / "simulated_quotes.parquet", index=False)
    truth.to_parquet(raw / "simulation_truth.parquet", index=False)
    qc.to_csv(proc / "data_quality_report.csv", index=False)
    cfg.status_table().to_csv(proc / "config_status_table.csv", index=False)
    (paths.docs() / "data_dictionary.md").write_text(to_markdown(), encoding="utf-8")
    h = frame_hash(df)
    (raw / "simulated_quotes.sha256").write_text(h + "\n")
    nb = df[(df["business_type"] == "new_business") & (df["treatment_group"] == "none")]
    ren = df[df["business_type"] == "renewal"]
    summary = {
        "data_label": "Simulated data",
        "rows": int(len(df)),
        "new_business_rows": int((df["business_type"] == "new_business").sum()),
        "renewal_rows": int(len(ren)),
        "randomised_test_rows": int((df["treatment_group"] != "none").sum()),
        "nb_conversion_rate": float(nb["converted"].mean()),
        "renewal_rate": float(ren["converted"].mean()),
        "avg_nb_premium": float(df.loc[df["business_type"] == "new_business", "our_premium"].mean()),
        "avg_our_rank": float(df["our_rank"].mean()),
        "quality_checks_passed": int(qc["passed"].sum()),
        "quality_checks_total": int(len(qc)),
        "sha256": h,
        "date_min": str(df["quote_date"].min().date()),
        "date_max": str(df["quote_date"].max().date()),
    }
    (proc / "m1_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
