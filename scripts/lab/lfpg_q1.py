"""Q1: how is the LFPG recorded off-block produced? Mechanism fractions."""
import os, polars as pl
S = os.environ["LFPG_OUT"]
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(20); pl.Config.set_fmt_str_lengths(40)
a = pl.read_parquet(f"{S}/lfpg_all.parquet")
d = a.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("src") == "train"))
sec = lambda c: (pl.col("BLOCK_TIME_UTC_mvt") - pl.col(c)).dt.total_seconds()
d = d.with_columns(
    m=pl.col("MVT_TIME_UTC_mvt").dt.month(),
    y=pl.col("TAXITIME_SEC_mvt"),
    nm=pl.col("LOBT_flt").is_not_null(),
    b_s=pl.col("BLOCK_TIME_UTC_mvt").dt.second(),
    t_s=pl.col("MVT_TIME_UTC_mvt").dt.second(),
    d_sched=sec("SCHED_TIME_UTC_mvt"), d_eobt=sec("EOBT_1_flt"), d_lobt=sec("LOBT_flt"),
    d_iobt=sec("IOBT_flt"), d_aobt=sec("AOBT_3_flt"),
    stp=pl.col("STAND_mvt").str.slice(0, 1), stp2=pl.col("STAND_mvt").str.slice(0, 2),
    al=pl.col("FLIGHT_mvt").str.extract(r"^([A-Z0-9]{2,3}?)\d", 1),
)
print("n", d.height, "nm frac", d["nm"].mean())
print("block seconds-of-minute dist (top)"); print(d["b_s"].value_counts().sort("count", descending=True).head(15))
print("takeoff seconds-of-minute (top)"); print(d["t_s"].value_counts().sort("count", descending=True).head(8))
# exact matches
for c in ["d_sched", "d_eobt", "d_lobt", "d_iobt", "d_aobt"]:
    x = d[c].drop_nulls()
    print(c, "n", x.len(), "|d|<=10", (x.abs() <= 10).mean(), "|d|<=60", (x.abs() <= 60).mean(), "exact0", (x == 0).mean(), "q", x.quantile(0.05), x.median(), x.quantile(0.95))
# mechanism classes
d = d.with_columns(
    mech=pl.when(pl.col("d_sched").abs() <= 10).then(pl.lit("sched"))
    .when(pl.col("d_aobt").abs() <= 10).then(pl.lit("aobt3"))
    .when(pl.col("d_eobt").abs() <= 10).then(pl.lit("eobt1"))
    .when(pl.col("d_lobt").abs() <= 10).then(pl.lit("lobt"))
    .otherwise(pl.lit("other")))
print(d.group_by("m", "mech").len().pivot(on="mech", index="m", values="len").sort("m"))
print(d.group_by("stp").agg(pl.len(), (pl.col("mech") == "sched").mean().alias("sched"), (pl.col("mech") == "aobt3").mean().alias("aobt"), pl.col("nm").mean().alias("nm"), pl.col("y").median().alias("ymed")).sort("len", descending=True).head(25))
print(d.group_by("RUNWAY_mvt").agg(pl.len(), pl.col("y").median().alias("ymed"), pl.col("y").mean().alias("ymean")).sort("len", descending=True).head(12))
print(d.group_by("al").agg(pl.len(), (pl.col("mech") == "sched").mean().alias("sched"), pl.col("nm").mean().alias("nm"), pl.col("y").median().alias("ymed")).sort("len", descending=True).head(20))
# taxi distribution by mech
print(d.group_by("mech").agg(pl.len(), pl.col("y").mean().alias("ymean"), pl.col("y").median().alias("ymed"), pl.col("y").quantile(0.99).alias("y99"), (pl.col("y") > 3600).sum().alias("gt1h")))
d.write_parquet(f"{S}/dep25.parquet")

# minute-bin histograms of block - X
for c in ["d_sched", "d_eobt", "d_lobt", "d_aobt"]:
    x = d.select(((pl.col(c) + 30) // 60).alias("mb")).drop_nulls()
    h = x.filter(pl.col("mb").abs() <= 6)["mb"].value_counts().sort("mb")
    print(c, dict(zip(h["mb"].to_list(), h["count"].to_list())))
