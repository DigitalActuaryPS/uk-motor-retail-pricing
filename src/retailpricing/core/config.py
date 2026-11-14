"""Calibration configuration loader. Each parameter carries value, status and source_note."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import yaml

from .paths import config_path

ALLOWED_STATUS = {"illustrative_assumption", "user_supplied_public"}


def _is_param(node: Any) -> bool:
    return isinstance(node, dict) and {"value", "status", "source_note"} <= set(node.keys())


class Config:
    def __init__(self, raw: dict):
        self.raw = raw

    def __getitem__(self, dotted: str) -> Any:
        section, name = dotted.split(".", 1)
        return self.raw[section][name]["value"]

    def iter_params(self) -> Iterator[tuple[str, dict]]:
        for section, body in self.raw.items():
            for name, node in body.items():
                yield f"{section}.{name}", node

    def validate(self) -> list[str]:
        problems: list[str] = []
        for path, node in self.iter_params():
            if not _is_param(node):
                problems.append(f"{path}: missing value, status or source_note")
                continue
            if node["status"] not in ALLOWED_STATUS:
                problems.append(f"{path}: invalid status {node['status']!r}")
            if node["status"] == "illustrative_assumption" and node["source_note"] not in ("", None):
                problems.append(f"{path}: illustrative_assumption must have an empty source_note")
            if node["status"] == "user_supplied_public" and not node["source_note"]:
                problems.append(f"{path}: user_supplied_public needs a source_note")
        return problems

    def status_table(self):
        import pandas as pd

        rows = [
            {"parameter": p, "status": n.get("status"), "source_note": n.get("source_note", "")}
            for p, n in self.iter_params()
        ]
        return pd.DataFrame(rows)


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else config_path()
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = Config(raw)
    problems = cfg.validate()
    if problems:
        raise ValueError("Invalid calibration config: " + "; ".join(problems))
    return cfg
