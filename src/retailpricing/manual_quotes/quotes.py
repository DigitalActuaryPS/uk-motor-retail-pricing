"""Manual comparison-site quote input: template, validator, loader, analysis.

Quotes are typed in by the requester from a live comparison-site session. There is no scraping and no
automated call to any site. This package imports only core configuration and path helpers. No model,
calibration or optimisation code imports it (a test enforces this).
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ..core import paths
from ..core.config import Config, load_config

COLUMNS = ["entry_date", "source_site", "age_band", "region", "vehicle_group", "no_claims_band", "cover_type",
           "annual_mileage_band", "insurer_name", "quoted_annual_premium", "displayed_rank", "notes"]
REQUIRED = [c for c in COLUMNS if c != "notes"]
SESSION_KEYS = ["entry_date", "source_site", "age_band", "region", "vehicle_group", "no_claims_band", "cover_type", "annual_mileage_band"]
BAND_FIELDS = {"age_band": "manual_input.age_bands", "region": "manual_input.regions", "vehicle_group": "manual_input.vehicle_groups",
               "no_claims_band": "manual_input.no_claims_bands", "cover_type": "manual_input.cover_types",
               "annual_mileage_band": "manual_input.mileage_bands"}

# Personal data guard for the free-text notes field
_POSTCODE = re.compile(r"\b[A-Za-z]{1,2}\d[A-Za-z\d]?\s*\d[A-Za-z]{2}\b")
_REG_PLATE = re.compile(r"\b[A-Za-z]{2}\d{2}\s?[A-Za-z]{3}\b")
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_LONG_DIGITS = re.compile(r"\d{6,}")


def manual_path() -> Path:
    return paths.data_manual() / "manual_quotes.csv"


def template_path() -> Path:
    return paths.data_manual() / "manual_quotes_template.csv"


def write_template() -> Path:
    """Header row only."""
    p = template_path()
    p.write_text(",".join(COLUMNS) + "\n", encoding="utf-8")
    return p


def validate_row(row: dict, cfg: Config, today: date | None = None) -> list[str]:
    today = today or date.today()
    errs: list[str] = []
    for c in REQUIRED:
        v = row.get(c)
        if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
            errs.append(f"{c}: required")
    if errs:
        return errs
    try:
        d = pd.to_datetime(str(row["entry_date"]), format="%Y-%m-%d").date()
        if d > today:
            errs.append("entry_date: future date")
    except Exception:
        errs.append("entry_date: must be YYYY-MM-DD")
    for field, key in BAND_FIELDS.items():
        if str(row[field]) not in [str(x) for x in cfg[key]]:
            errs.append(f"{field}: {row[field]!r} not in {list(cfg[key])}")
    try:
        prem = float(row["quoted_annual_premium"])
        if not np.isfinite(prem) or prem <= 0:
            errs.append("quoted_annual_premium: must be positive")
    except Exception:
        errs.append("quoted_annual_premium: must be a number")
    try:
        r = float(row["displayed_rank"])
        if r != int(r) or int(r) < 1:
            errs.append("displayed_rank: must be an integer of 1 or more")
    except Exception:
        errs.append("displayed_rank: must be an integer")
    for field in ["insurer_name", "source_site"]:
        if len(str(row[field])) > 80:
            errs.append(f"{field}: too long")
    notes = str(row.get("notes") or "")
    if any(p.search(notes) for p in (_POSTCODE, _REG_PLATE, _EMAIL, _LONG_DIGITS)):
        errs.append("notes: looks like personal data (postcode, registration, email or long number). Remove it")
    return errs


def validate_frame(df: pd.DataFrame, cfg: Config | None = None, today: date | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (valid_rows, error_report). Unknown columns are rejected so personal data cannot ride along."""
    cfg = cfg or load_config()
    extra = [c for c in df.columns if c not in COLUMNS]
    report = []
    if extra:
        report.append({"row": "file", "errors": f"columns not allowed: {extra}"})
        return pd.DataFrame(columns=COLUMNS), pd.DataFrame(report)
    valid_idx = []
    for i, row in df.reset_index(drop=True).iterrows():
        errs = validate_row(row.to_dict(), cfg, today)
        if errs:
            report.append({"row": int(i) + 2, "errors": "; ".join(errs)})
        else:
            valid_idx.append(i)
    valid = df.reset_index(drop=True).loc[valid_idx, :].copy()
    for c in COLUMNS:
        if c not in valid.columns:
            valid[c] = ""
    valid = valid[COLUMNS]
    valid["quoted_annual_premium"] = valid["quoted_annual_premium"].astype(float)
    valid["displayed_rank"] = valid["displayed_rank"].astype(float).astype(int)
    return valid, pd.DataFrame(report, columns=["row", "errors"])


def load_manual_quotes(path: Path | None = None, cfg: Config | None = None) -> pd.DataFrame:
    """Valid manual quotes. An absent, empty or header-only file gives an empty frame."""
    path = path or manual_path()
    empty = pd.DataFrame(columns=COLUMNS)
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return empty
    try:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return empty
    if raw.empty:
        return empty
    valid, _ = validate_frame(raw, cfg)
    return valid


def append_valid_rows(df: pd.DataFrame, path: Path | None = None, cfg: Config | None = None) -> tuple[int, pd.DataFrame]:
    """Validate and append. Returns (rows_written, error_report). Invalid rows are never written."""
    path = Path(path or manual_path())
    valid, report = validate_frame(df.astype(str) if len(df) else df, cfg)
    if len(valid):
        write_header = (not path.exists()) or path.stat().st_size == 0
        valid.to_csv(path, mode="a", header=write_header, index=False)
    return len(valid), report


# ----------------------------------------------------------------------------- analysis

def session_stats(manual: pd.DataFrame) -> pd.DataFrame:
    if manual.empty:
        return pd.DataFrame(columns=["n_quotes", "median_to_min", "rank_spread"])
    g = manual.groupby(SESSION_KEYS)
    out = g.agg(n_quotes=("quoted_annual_premium", "size"), median_prem=("quoted_annual_premium", "median"),
                min_prem=("quoted_annual_premium", "min"), rank_min=("displayed_rank", "min"), rank_max=("displayed_rank", "max")).reset_index()
    out["median_to_min"] = out["median_prem"] / out["min_prem"]
    out["rank_spread"] = out["rank_max"] - out["rank_min"]
    return out


def compare_with_simulation(manual: pd.DataFrame, sim: pd.DataFrame, cfg: Config) -> dict:
    """Descriptive comparison of manual quote spread with the simulated competitor spread. No inference."""
    n = int(len(manual))
    minimum = int(cfg["manual_input.min_quotes_for_inference"])
    sim_ratio = (sim["comp_median"] / sim["comp_min"]).to_numpy()
    sim_rank = sim["our_rank"].to_numpy()
    out = {
        "n_quotes": n, "n_sessions": 0, "min_quotes_for_inference": minimum, "anecdotal": n < minimum,
        "simulated_median_to_min": {"mean": float(sim_ratio.mean()), "p10": float(np.percentile(sim_ratio, 10)), "p90": float(np.percentile(sim_ratio, 90))},
        "simulated_rank_spread_p10_to_p90": float(np.percentile(sim_rank, 90) - np.percentile(sim_rank, 10)),
        "manual_median_to_min": None, "manual_rank_spread": None,
    }
    if n == 0:
        out["statement"] = "No manual quotes loaded (sample size 0). Nothing to compare. The project runs fully without manual data."
        return out
    ss = session_stats(manual)
    out["n_sessions"] = int(len(ss))
    out["manual_median_to_min"] = float(ss["median_to_min"].mean())
    out["manual_rank_spread"] = float(ss["rank_spread"].mean())
    stmt = f"Sample size: {n} quotes in {len(ss)} session(s). "
    stmt += (f"Fewer than {minimum} quotes is anecdotal and supports no inference. " if n < minimum else
             f"{n} quotes meets the {minimum}-quote threshold for description, but one site, one period and hand entry still limit what it shows. ")
    stmt += "Manual quotes are never used for training, calibration or optimisation."
    out["statement"] = stmt
    return out


def format_report(res: dict) -> str:
    lines = [f"Manual quote comparison (simulated spread is Simulated data)", res["statement"]]
    sm = res["simulated_median_to_min"]
    lines.append(f"Simulated competitor median-to-minimum ratio: mean {sm['mean']:.3f}, 10th to 90th percentile {sm['p10']:.3f} to {sm['p90']:.3f}")
    lines.append(f"Simulated rank spread (90th minus 10th percentile of our rank): {res['simulated_rank_spread_p10_to_p90']:.1f}")
    if res["manual_median_to_min"] is not None:
        lines.append(f"Manual median-to-minimum ratio (mean over sessions): {res['manual_median_to_min']:.3f}")
        lines.append(f"Manual rank spread (mean over sessions): {res['manual_rank_spread']:.1f}")
    return "\n".join(lines)


def run_analysis(print_report: bool = True) -> dict:
    from ..core.io import load_quotes  # reads simulated data only

    cfg = load_config()
    res = compare_with_simulation(load_manual_quotes(cfg=cfg), load_quotes(), cfg)
    if print_report:
        print(format_report(res))
    return res


if __name__ == "__main__":
    write_template()
    run_analysis()
