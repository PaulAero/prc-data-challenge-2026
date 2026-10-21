"""Q3: did LFPG recording change between 2025 and the ranking months? Label-free signals."""
import os, numpy as np, polars as pl
S = os.environ["LFPG_OUT"]
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250)
a = pl.read_parquet(f"{S}/lfpg_all.parquet").with_columns(
    ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"), nm=pl.col("LOBT_flt").is_not_null())
dep = a.filter(pl.col("PHASE_mvt") == "DEP").with_columns(
    px=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds(),
    tl=(pl.col("MVT_TIME_UTC_mvt") - pl.col("LOBT_flt")).dt.total_seconds(),
    ts=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
    ul=(pl.col("AOBT_3_flt") - pl.col("LOBT_flt")).dt.total_seconds(),
    tsec=pl.col("MVT_TIME_UTC_mvt").dt.second(), stp=pl.col("STAND_mvt").str.slice(0, 1))
keep = ["2025-01", "2025-02", "2025-06", "2025-07", "2025-08", "2025-12", "2026-01", "2026-07"]
print("DEP by month: n, no-NM share, takeoff - AOBT3 quantiles, T-LOBT, AOBT3-LOBT median, takeoff sec in 54..6, stand null, runway")
print(dep.filter(pl.col("ym").is_in(keep)).group_by("ym").agg(pl.len(), (~pl.col("nm")).mean().alias("nonm"),
    pl.col("px").quantile(0.1).alias("px10"), pl.col("px").median().alias("px50"), pl.col("px").quantile(0.9).alias("px90"), pl.col("px").mean().alias("pxmean"),
    pl.col("tl").median().alias("tl50"), pl.col("ul").median().alias("ul50"),
    ((pl.col("tsec") >= 54) | (pl.col("tsec") <= 6)).mean().alias("t_jit"),
    pl.col("STAND_mvt").is_null().mean().alias("st_null"), pl.col("RUNWAY_mvt").is_null().mean().alias("rw_null"),
    ((pl.col("px") % 60) == 0).mean().alias("px_round")).sort("ym"))
# px exact default values? mode of px rounded to 10 s
for ym in ["2025-01", "2025-07", "2026-01", "2026-07"]:
    v = dep.filter(pl.col("ym") == ym).select((pl.col("px") / 60).round(0).alias("pm")).drop_nulls()["pm"].value_counts().sort("count", descending=True).head(6)
    print(ym, "px minute modes", list(zip(v["pm"].to_list(), v["count"].to_list())))
# stand prefix mix by month
sm = dep.filter(pl.col("ym").is_in(["2025-01", "2025-07", "2026-01", "2026-07"])).group_by("ym", "stp").len().with_columns(sh=pl.col("len") / pl.col("len").sum().over("ym")).pivot(on="ym", index="stp", values="sh").sort("2025-07", descending=True)
print(sm.head(25))
rw = dep.filter(pl.col("ym").is_in(["2025-01", "2025-07", "2026-01", "2026-07"])).group_by("ym", "RUNWAY_mvt").len().with_columns(sh=pl.col("len") / pl.col("len").sum().over("ym")).pivot(on="ym", index="RUNWAY_mvt", values="sh").sort("2025-07", descending=True)
print(rw)
# arrivals
arr = a.filter(pl.col("PHASE_mvt") == "ARR").with_columns(
    ti=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds(),
    bsec=pl.col("BLOCK_TIME_UTC_mvt").dt.second(), lsec=pl.col("MVT_TIME_UTC_mvt").dt.second(),
    bs=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
    ba=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("ARVT_3_flt")).dt.total_seconds())
print("ARR by month: taxi-in quantiles, in-block null, in-block==sched(<=10s), in-block jitter share, in-block - ARVT_3")
print(arr.filter(pl.col("ym").is_in(keep)).group_by("ym").agg(pl.len(), pl.col("ti").quantile(0.1).alias("ti10"), pl.col("ti").median().alias("ti50"),
    pl.col("ti").quantile(0.9).alias("ti90"), pl.col("ti").quantile(0.99).alias("ti99"), (pl.col("ti") > 3600).sum().alias("ti_gt1h"), (pl.col("ti") < 60).sum().alias("ti_lt1m"),
    pl.col("BLOCK_TIME_UTC_mvt").is_null().mean().alias("ib_null"), (pl.col("bs").abs() <= 10).mean().alias("ib_sched"),
    ((pl.col("bsec") >= 54) | (pl.col("bsec") <= 6)).mean().alias("ib_jit"), pl.col("ba").median().alias("ba50"),
    (~pl.col("nm")).mean().alias("nonm")).sort("ym"))
for ym in ["2025-01", "2025-07", "2026-01", "2026-07"]:
    v = arr.filter(pl.col("ym") == ym).select(((pl.col("bs") + 30) // 60).clip(-8, 8).alias("b")).drop_nulls()["b"].value_counts().sort("b")
    print(ym, "arr inblock-sched minute bins", dict(zip(v["b"].to_list(), v["count"].to_list())))
# twins (dup departures) per month
dd = dep.drop_nulls("FLIGHT_mvt").select("MVT_ID_mvt", "FLIGHT_mvt", "SCHED_TIME_UTC_mvt", "ym", "nm", pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").alias("t"))
j = dd.join(dd, on=["FLIGHT_mvt", "SCHED_TIME_UTC_mvt"], suffix="_b").filter((pl.col("MVT_ID_mvt") < pl.col("MVT_ID_mvt_b")) & ((pl.col("t") - pl.col("t_b")).abs() <= 60))
print("dup departure pairs by month", j.group_by("ym").agg(pl.len(), (pl.col("nm") != pl.col("nm_b")).sum().alias("one_matched")).sort("ym"))
da = arr.drop_nulls("FLIGHT_mvt").select("MVT_ID_mvt", "FLIGHT_mvt", "SCHED_TIME_UTC_mvt", "ym", pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").alias("t"))
j = da.join(da, on=["FLIGHT_mvt", "SCHED_TIME_UTC_mvt"], suffix="_b").filter((pl.col("MVT_ID_mvt") < pl.col("MVT_ID_mvt_b")) & ((pl.col("t") - pl.col("t_b")).abs() <= 60))
print("dup arrival pairs by month", j.group_by("ym").len().sort("ym"))
