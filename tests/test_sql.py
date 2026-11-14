import numpy as np
import pandas as pd
import pytest

from retailpricing.analytics_sql.duck import connect, run_all, run_sql


@pytest.fixture(scope="module")
def con_df(small):
    df, _ = small
    return connect(df), df


def nb_obs(df):
    return df[(df.business_type.astype(str) == "new_business") & (df.treatment_group.astype(str) == "none")]


def test_segment_summary_matches_pandas(con_df):
    con, df = con_df
    sql = run_sql(con, "segment_summary").set_index("segment")
    pdf = nb_obs(df).assign(segment=lambda d: d.segment.astype(str)).groupby("segment").agg(
        quotes=("converted", "size"), policies=("converted", "sum"), conversion=("converted", "mean"),
        gwp=("written_premium", "sum"), contribution=("fixed_cost_contribution", "sum"))
    assert np.allclose(sql["quotes"], pdf["quotes"]) and np.allclose(sql["policies"], pdf["policies"])
    assert np.allclose(sql["conversion"], pdf["conversion"])
    assert np.allclose(sql["gwp"], pdf["gwp"]) and np.allclose(sql["contribution"], pdf["contribution"])


def test_rank_conversion_matches_pandas(con_df):
    con, df = con_df
    sql = run_sql(con, "rank_conversion").set_index("our_rank")
    pdf = nb_obs(df).groupby("our_rank").agg(quotes=("converted", "size"), conversion=("converted", "mean"),
                                             contribution=("fixed_cost_contribution", "sum"))
    assert np.allclose(sql["quotes"], pdf["quotes"]) and np.allclose(sql["conversion"], pdf["conversion"])
    assert np.allclose(sql["contribution"], pdf["contribution"])


def test_channel_and_buckets_and_renewals_match_pandas(con_df):
    con, df = con_df
    sql = run_sql(con, "channel_summary").set_index("channel")
    pdf = nb_obs(df).assign(channel=lambda d: d.channel.astype(str)).groupby("channel")["converted"].agg(["size", "mean"])
    assert np.allclose(sql["quotes"], pdf["size"]) and np.allclose(sql["conversion"], pdf["mean"])

    b = run_sql(con, "price_ratio_buckets").set_index("ratio_bucket")
    d = nb_obs(df)
    cuts = pd.cut(d.price_ratio_to_median, [0, .90, .95, 1.0, 1.05, 1.10, 99], right=False,
                  labels=["<0.90", "0.90-0.95", "0.95-1.00", "1.00-1.05", "1.05-1.10", ">=1.10"])
    pb = d.groupby(cuts, observed=True)["converted"].agg(["size", "mean"])
    pb.index = pb.index.astype(str)
    assert np.allclose(b.loc[pb.index, "quotes"], pb["size"]) and np.allclose(b.loc[pb.index, "conversion"], pb["mean"])

    r = run_sql(con, "renewal_summary")
    ren = df[df.business_type.astype(str) == "renewal"]
    assert r["offers"].sum() == len(ren)
    assert r["offers"].mul(r["renewal_rate"]).sum() == pytest.approx(ren["converted"].sum())


def test_all_queries_run(con_df):
    con, _ = con_df
    res = run_all(con)
    assert all(len(v) > 0 for v in res.values())
    m = res["monthly_trend"]
    assert m["quotes"].sum() == len(con_df[1])
