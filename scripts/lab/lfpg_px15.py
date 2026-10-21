"""Is there a 15-min default in LFPG AOBT_3 (take-off - AOBT_3 = 900 s) in July 2026?"""
import os, numpy as np, polars as pl
S = os.environ["LFPG_OUT"]
pl.Config.set_tbl_rows(70); pl.Config.set_tbl_width_chars(250)
a = pl.read_parquet(f"{S}/lfpg_all.parquet").filter(pl.col("PHASE_mvt") == "DEP").with_columns(
    ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"), day=pl.col("MVT_TIME_UTC_mvt").dt.date(),
    px=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds(),
    al=pl.col("FLIGHT_mvt").str.extract(r"^([A-Z]{3})", 1))
# fine histogram near 900 (seconds, with takeoff jitter)
for ym in ["2025-01", "2025-07", "2025-08", "2026-01", "2026-07"]:
    x = a.filter(pl.col("ym") == ym)
    h = x.select(((pl.col("px") + 30) // 60).alias("pm")).filter(pl.col("pm").is_between(10, 20))["pm"].value_counts().sort("pm")
    n = x.height
    print(ym, n, {k: round(v / n * 1000, 1) for k, v in zip(h["pm"].to_list(), h["count"].to_list())})
# per day share at px minute 15 vs neighbours (14,16) in Jul 2025/2026
for ym in ["2025-07", "2026-07", "2026-01"]:
    x = a.filter(pl.col("ym") == ym).with_columns(pm=((pl.col("px") + 30) // 60))
    g = x.group_by("day").agg(pl.len(), (pl.col("pm") == 15).sum().alias("p15"), ((pl.col("pm") == 14) | (pl.col("pm") == 16)).sum().alias("p1416")).with_columns(ex=pl.col("p15") - pl.col("p1416") / 2).sort("day")
    print(ym, "daily excess at 15 min:", g["ex"].to_list())
    gl = x.group_by("al").agg(pl.len(), (pl.col("pm") == 15).sum().alias("p15"), ((pl.col("pm") == 14) | (pl.col("pm") == 16)).sum().alias("p1416")).with_columns(ex=pl.col("p15") - pl.col("p1416") / 2).sort("ex", descending=True)
    print(gl.head(8))
