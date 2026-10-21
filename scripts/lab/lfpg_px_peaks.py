"""Label-free flag: LFPG departures whose take-off - AOBT_3 falls on a peak minute of its (stand letter,
runway, month) histogram (candidate NM default taxi values). Does the flag carry information in 2025?"""
import os, numpy as np, polars as pl
S = os.environ["LFPG_OUT"]
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250)
N = 344_841
a = pl.read_parquet(f"{S}/lfpg_all.parquet").filter(pl.col("PHASE_mvt") == "DEP").with_columns(
    ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"), stp=pl.col("STAND_mvt").str.slice(0, 1).fill_null("NA"),
    rw=pl.col("RUNWAY_mvt").fill_null("NA"),
    px=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64))
a = a.with_columns(pm=((pl.col("px") + 30) // 60).cast(pl.Int64))
key = ["stp", "rw", "ym"]
h = a.drop_nulls("pm").group_by(key + ["pm"]).len()
# neighbours +-1 and +-2
nb = h.select(key + ["pm", "len"])
def shifted(k):
    return nb.with_columns((pl.col("pm") - k).alias("pm")).rename({"len": f"n{k:+d}"})
hh = h
for k in [-2, -1, 1, 2]:
    hh = hh.join(shifted(k), on=key + ["pm"], how="left")
hh = hh.with_columns([pl.col(f"n{k:+d}").fill_null(0) for k in [-2, -1, 1, 2]])
# peak if count exceeds the mean of the +-2 neighbours (excluding the paired neighbour) by 50% and >= 8 rows
hh = hh.with_columns(bg=(pl.col("n-2") + pl.col("n+2")) / 2)
hh = hh.with_columns(peak=(pl.col("len") >= 8) & (pl.col("len") > 1.5 * pl.max_horizontal("bg", pl.lit(1.0))) & (pl.col("len") >= pl.col("n-1")) & (pl.col("len") >= pl.col("n+1")),
                     excess=(pl.col("len") - pl.col("bg")).clip(0))
a = a.join(hh.select(key + ["pm", "peak", "bg", "len"]), on=key + ["pm"], how="left").with_columns(pl.col("peak").fill_null(False))
print("peak share and estimated excess by month")
print(a.group_by("ym").agg(pl.len(), pl.col("peak").mean().alias("peak_share"),
      ((pl.col("len") - pl.col("bg")) / pl.col("len")).filter(pl.col("peak")).mean().alias("excess_frac_in_peaks")).sort("ym"))
# 2025 label check
o = pl.read_parquet(f"{S}/oof_ens.parquet").select("MVT_ID_mvt", "fold", "pred")
x = a.filter(pl.col("src") == "train").with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64)).join(o, on="MVT_ID_mvt").with_columns(
    y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)).with_columns(r=pl.col("y") - pl.col("pred")).filter(pl.col("y") <= 10800)
x = x.with_columns(gmed=pl.col("y").median().over("stp", "rw", "fold"))
print(x.filter(pl.col("px").is_between(300, 2400)).group_by("fold", "peak").agg(pl.len(), pl.corr("px", "y").alias("corr_px_y"),
      ((pl.col("y") - pl.col("px")).abs().median()).alias("mad_y_px"), (pl.col("r") ** 2).mean().sqrt().alias("rmse_model"), pl.col("r").mean().alias("bias")).sort("fold", "peak"))
# all-2025 version: |block - AOBT3| on peak vs non-peak rows
y25 = a.filter((pl.col("src") == "train") & pl.col("px").is_between(300, 2400)).with_columns(ba=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().abs())
print(y25.group_by("peak").agg(pl.len(), pl.col("ba").median().alias("med_abs_block_aobt"), (pl.col("ba") <= 120).mean().alias("within2m"), pl.corr("px", "TAXITIME_SEC_mvt").alias("corr")))
a.select("MVT_ID_mvt", "ym", "src", "peak", "pm", "stp", "rw").write_parquet(f"{S}/px_peaks.parquet")
