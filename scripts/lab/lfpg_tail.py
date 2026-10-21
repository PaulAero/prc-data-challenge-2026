"""LFPG tail: no-NM rows, long targets, full-year 2025; ranking counts."""
import os, numpy as np, polars as pl
S = os.environ["LFPG_OUT"]
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(16); pl.Config.set_fmt_str_lengths(30); pl.Config.set_tbl_width_chars(250)
d = pl.read_parquet(f"{S}/dep25.parquet")
d = d.with_columns(ts=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
                   tl=(pl.col("MVT_TIME_UTC_mvt") - pl.col("LOBT_flt")).dt.total_seconds())
print("no-NM by month", d.group_by("m").agg((~pl.col("nm")).sum().alias("nonm"), pl.len()).sort("m"))
x = d.filter(~pl.col("nm"))
print("no-NM target quantiles", x["y"].quantile(0.5), x["y"].quantile(0.9), x["y"].quantile(0.99), "n>3h", (x["y"] > 10800).sum(), "n>1h", (x["y"] > 3600).sum())
print("matched >3h", d.filter(pl.col("nm") & (pl.col("y") > 10800)).height)
cols = ["FLIGHT_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "LOBT_flt", "AOBT_3_flt", "STAND_mvt", "RUNWAY_mvt", "y", "ts"]
print(d.filter(pl.col("y") > 3 * 3600).sort("y", descending=True).select(cols))
print(d.filter((pl.col("y") > 3600) & (pl.col("y") <= 3 * 3600)).with_columns(nmf=pl.col("nm")).group_by("nmf").len())
# date / tz shifts on all rows: block - (T - typical)?
x = x.with_columns(bs=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds())
print("no-NM rows block-sched minute hist", x.select(((pl.col("bs") + 30) // 60).clip(-10, 10).alias("b")).group_by("b").len().sort("b"))
# ranking side
a = pl.read_parquet(f"{S}/lfpg_all.parquet")
r = a.filter((pl.col("src") == "rank") & (pl.col("PHASE_mvt") == "DEP")).with_columns(m=pl.col("MVT_TIME_UTC_mvt").dt.month(), nm=pl.col("LOBT_flt").is_not_null())
print("ranking no-NM by month", r.group_by("m").agg((~pl.col("nm")).sum().alias("nonm"), pl.len()).sort("m"))
