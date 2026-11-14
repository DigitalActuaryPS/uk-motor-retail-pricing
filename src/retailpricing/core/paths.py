"""Repository paths. Set RP_ROOT to redirect all outputs (used by tests)."""
from __future__ import annotations

import os
from pathlib import Path


def root() -> Path:
    env = os.environ.get("RP_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3]


def config_path() -> Path:
    # The configuration always comes from the repository, even when RP_ROOT redirects outputs.
    return Path(__file__).resolve().parents[3] / "config" / "calibration.yaml"


def data_raw() -> Path:
    return _ensure(root() / "data" / "raw")


def data_processed() -> Path:
    return _ensure(root() / "data" / "processed")


def data_manual() -> Path:
    return _ensure(root() / "data" / "manual")


def figures() -> Path:
    return _ensure(root() / "docs" / "figures")


def docs() -> Path:
    return _ensure(root() / "docs")


def sql_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "sql"


def _ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
