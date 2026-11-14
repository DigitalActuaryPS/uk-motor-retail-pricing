"""Loading helpers and the time-based split. Simulated data only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .config import Config
from .features import NB, REN


def load_quotes() -> pd.DataFrame:
    f = paths.data_raw() / "simulated_quotes.parquet"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found. Run `make data` first.")
    return pd.read_parquet(f)


def load_truth() -> pd.DataFrame:
    return pd.read_parquet(paths.data_raw() / "simulation_truth.parquet")


def nb_observational(df: pd.DataFrame) -> pd.DataFrame:
    """New business quotes outside the randomised price test."""
    return df[(df["business_type"].astype(str) == NB) & (df["treatment_group"].astype(str) == "none")].copy()


def nb_experiment(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["treatment_group"].astype(str) != "none"].copy()


def renewals(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["business_type"].astype(str) == REN].copy()


def split_dates(cfg: Config) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(cfg["volumes.start_date"])
    train_end = start + pd.DateOffset(months=cfg["modelling.train_months"])
    val_end = train_end + pd.DateOffset(months=cfg["modelling.validation_months"])
    return train_end, val_end


def time_split(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train, validation, test by quote date. No shuffling, so no future information leaks backwards."""
    a, b = split_dates(cfg)
    d = df["quote_date"]
    return df[d < a].copy(), df[(d >= a) & (d < b)].copy(), df[d >= b].copy()


def write_json(obj: dict, name: str) -> Path:
    p = paths.data_processed() / name
    p.write_text(json.dumps(obj, indent=2, default=_json_default))
    return p


def read_json(name: str) -> dict:
    return json.loads((paths.data_processed() / name).read_text())


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
