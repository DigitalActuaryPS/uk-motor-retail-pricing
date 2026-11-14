"""DuckDB layer. Loads the simulated quotes and runs the SQL files in /sql. Simulated data only."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from ..core import paths

SQL_FILES = {
    "segment_summary": "01_segment_summary.sql",
    "rank_conversion": "02_rank_conversion.sql",
    "channel_summary": "03_channel_summary.sql",
    "monthly_trend": "04_monthly_trend.sql",
    "price_ratio_buckets": "05_price_ratio_buckets.sql",
    "renewal_summary": "06_renewal_summary.sql",
    "price_test_arms": "07_price_test_arms.sql",
}


def connect(df: pd.DataFrame | None = None, parquet: Path | None = None) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    if df is not None:
        d = df.copy()
        for c in d.select_dtypes("category").columns:
            d[c] = d[c].astype(str)
        con.register("quotes_src", d)
        con.execute("CREATE TABLE quotes AS SELECT * FROM quotes_src")
    else:
        parquet = parquet or paths.data_raw() / "simulated_quotes.parquet"
        con.execute(f"CREATE TABLE quotes AS SELECT * FROM read_parquet('{parquet.as_posix()}')")
    return con


def run_sql(con: duckdb.DuckDBPyConnection, name: str) -> pd.DataFrame:
    text = (paths.sql_dir() / SQL_FILES[name]).read_text()
    return con.execute(text).fetchdf()


def run_all(con) -> dict[str, pd.DataFrame]:
    return {k: run_sql(con, k) for k in SQL_FILES}


def main() -> dict:
    con = connect()
    out = paths.data_processed()
    res = run_all(con)
    for k, v in res.items():
        v.to_csv(out / f"m7_sql_{k}__simulated_data.csv", index=False)
    return {k: len(v) for k, v in res.items()}


if __name__ == "__main__":
    print(main())
