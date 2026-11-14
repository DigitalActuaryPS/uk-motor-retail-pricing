import re
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from retailpricing.manual_quotes import quotes as mq

ROOT = Path(__file__).resolve().parents[1]


def good(**over):
    r = {"entry_date": "2024-06-01", "source_site": "Site A", "age_band": "35-49", "region": "Midlands_East",
         "vehicle_group": "medium", "no_claims_band": "3-5", "cover_type": "comprehensive", "annual_mileage_band": "5001-8000",
         "insurer_name": "Insurer X", "quoted_annual_premium": "512.40", "displayed_rank": "3", "notes": ""}
    r.update(over)
    return r


TODAY = date(2024, 7, 1)


def test_good_row_valid(cfg):
    assert mq.validate_row(good(), cfg, TODAY) == []


@pytest.mark.parametrize("field,value,fragment", [
    ("quoted_annual_premium", "-5", "positive"),
    ("quoted_annual_premium", "0", "positive"),
    ("quoted_annual_premium", "abc", "number"),
    ("displayed_rank", "2.5", "integer"),
    ("displayed_rank", "0", "integer"),
    ("displayed_rank", "x", "integer"),
    ("age_band", "99-100", "age_band"),
    ("region", "Mars", "region"),
    ("vehicle_group", "tank", "vehicle_group"),
    ("no_claims_band", "20", "no_claims_band"),
    ("cover_type", "none", "cover_type"),
    ("annual_mileage_band", "lots", "annual_mileage_band"),
    ("entry_date", "2024-12-01", "future"),
    ("entry_date", "01/06/2024", "YYYY-MM-DD"),
    ("insurer_name", "", "required"),
    ("source_site", "", "required"),
])
def test_invalid_fields_rejected(cfg, field, value, fragment):
    errs = mq.validate_row(good(**{field: value}), cfg, TODAY)
    assert errs and any(fragment in e for e in errs), errs


@pytest.mark.parametrize("note", ["SW1A 1AA", "AB12 CDE", "me@example.com", "tel 07123456789"])
def test_notes_personal_data_guard(cfg, note):
    assert mq.validate_row(good(notes=note), cfg, TODAY)


def test_unknown_columns_rejected(cfg):
    df = pd.DataFrame([{**good(), "postcode": "M1 1AA"}])
    valid, report = mq.validate_frame(df, cfg, TODAY)
    assert valid.empty and "not allowed" in report.iloc[0]["errors"]


def test_template_is_header_only(tmp_path, monkeypatch):
    monkeypatch.setenv("RP_ROOT", str(tmp_path))
    p = mq.write_template()
    assert p.read_text().strip().split(",") == mq.COLUMNS
    assert len(p.read_text().strip().splitlines()) == 1


def test_loader_handles_absent_empty_and_header_only(tmp_path, cfg):
    assert mq.load_manual_quotes(tmp_path / "nope.csv", cfg).empty
    (tmp_path / "e.csv").write_text("")
    assert mq.load_manual_quotes(tmp_path / "e.csv", cfg).empty
    (tmp_path / "h.csv").write_text(",".join(mq.COLUMNS) + "\n")
    assert mq.load_manual_quotes(tmp_path / "h.csv", cfg).empty


def test_append_writes_only_valid_rows_and_one_header(tmp_path, cfg):
    path = tmp_path / "manual_quotes.csv"
    df = pd.DataFrame([good(), good(quoted_annual_premium="-1"), good(insurer_name="Insurer Y", displayed_rank="4")])
    n, report = mq.append_valid_rows(df, path, cfg)
    assert n == 2 and len(report) == 1
    n2, _ = mq.append_valid_rows(pd.DataFrame([good(insurer_name="Insurer Z", displayed_rank="5")]), path, cfg)
    assert n2 == 1
    text = path.read_text().splitlines()
    assert text[0] == ",".join(mq.COLUMNS) and sum(l.startswith("entry_date") for l in text) == 1
    assert len(mq.load_manual_quotes(path, cfg)) == 3


def test_analysis_states_sample_size_and_anecdotal(small, cfg):
    sim = small[0]
    rows = [good(insurer_name=f"I{i}", quoted_annual_premium=str(500 + 20 * i), displayed_rank=str(i + 1)) for i in range(12)]
    valid, _ = mq.validate_frame(pd.DataFrame(rows), cfg, TODAY)
    res = mq.compare_with_simulation(valid, sim, cfg)
    assert res["n_quotes"] == 12 and res["n_sessions"] == 1 and res["anecdotal"]
    assert "Sample size: 12 quotes" in res["statement"]
    assert "Fewer than 30 quotes is anecdotal and supports no inference" in res["statement"]
    expected_ratio = pd.Series([500 + 20 * i for i in range(12)]).median() / 500
    assert res["manual_median_to_min"] == pytest.approx(expected_ratio)
    assert res["manual_rank_spread"] == 11


def test_analysis_runs_with_no_manual_data(small, cfg):
    res = mq.compare_with_simulation(mq.load_manual_quotes(Path("/nonexistent.csv"), cfg), small[0], cfg)
    assert res["n_quotes"] == 0 and "No manual quotes" in res["statement"]


def test_manual_quotes_never_imported_by_modelling_code():
    offenders = []
    for f in (ROOT / "src" / "retailpricing").rglob("*.py"):
        if "manual_quotes" in f.parts:
            continue
        if re.search(r"(?:from|import)\s+[\w.]*manual_quotes|manual_quotes\.csv|paths\.data_manual\(", f.read_text()):
            offenders.append(str(f))
    assert not offenders, offenders


def test_gitignore_excludes_manual_quotes_file():
    assert "data/manual/manual_quotes.csv" in (ROOT / ".gitignore").read_text().splitlines()


def test_no_scraping_or_site_calls_in_repository():
    banned = re.compile(r"^\s*(import|from)\s+(requests|selenium|playwright|bs4|scrapy|urllib|httpx|mechanize|aiohttp)\b", re.M)
    for f in list((ROOT / "src").rglob("*.py")) + list((ROOT / "app").rglob("*.py")):
        assert not banned.search(f.read_text()), f
