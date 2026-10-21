"""Day-level test: does the model's shift of (pred - px) with the EOBT_1/AOBT_3 composition track the
real (y - px) across 2025 days? Full-year parts (two-part mixture) and jan/jul ensemble."""
import os, numpy as np, polars as pl
S = os.environ["LFPG_OUT"]
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250)
a = pl.read_parquet(f"{S}/lfpg_all.parquet").filter((pl.col("PHASE_mvt") == "DEP") & pl.col("LOBT_flt").is_not_null()).with_columns(
    pl.col("MVT_ID_mvt").cast(pl.Int64), ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"), day=pl.col("MVT_TIME_UTC_mvt").dt.date(),
    px=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64),
    ae=(pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt")).dt.total_seconds() / 60,
    eupd=(pl.col("EOBT_1_flt") != pl.col("LOBT_flt")), y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
p = pl.read_parquet("experiments/lab-year/parts.parquet").select("MVT_ID_mvt", "mvt_vs_schedule_sec", "reg", "p_base", "p_wsq", "win_lo", "win_hi").with_columns(pf=(pl.col("p_base") + pl.col("p_wsq")) / 2)
p = p.with_columns(ypart=(pl.col("pf") * pl.col("mvt_vs_schedule_sec") + (1 - pl.col("pf")) * pl.col("reg")).clip(pl.col("win_lo").clip(0), pl.col("win_hi")))
o = pl.read_parquet(f"{S}/oof_ens.parquet").select("MVT_ID_mvt", pl.col("pred").alias("ens"))
s = pl.read_parquet("data/submissions/candidate_v18.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("sub"))
a = a.join(p.select("MVT_ID_mvt", "ypart", "reg"), on="MVT_ID_mvt", how="left").join(o, on="MVT_ID_mvt", how="left").join(s, on="MVT_ID_mvt", how="left")
ok = pl.col("px").is_between(0, 3600) & ((pl.col("y") < 7200) | pl.col("y").is_null())
x = a.filter(ok)
dd = x.group_by("ym", "day").agg(pl.len().alias("n"), (pl.col("y") - pl.col("px")).mean().alias("y_px"), (pl.col("reg") - pl.col("px")).mean().alias("reg_px"),
    (pl.col("ens") - pl.col("px")).mean().alias("ens_px"), (pl.col("sub") - pl.col("px")).mean().alias("sub_px"),
    pl.col("eupd").mean().alias("eupd"), pl.col("ae").median().alias("ae50"), (pl.col("ae").abs() <= 3).mean().alias("ae_small")).sort("day")
t = dd.filter(pl.col("y_px").is_not_null())
for col in ["reg_px", "ens_px"]:
    z = t.drop_nulls(col)
    b = np.polyfit(z[col].to_numpy(), z["y_px"].to_numpy(), 1)
    print(f"daily y-px ~ {col}: slope {b[0]:.2f} intercept {b[1]:.1f}, corr {np.corrcoef(z[col].to_numpy(), z['y_px'].to_numpy())[0,1]:.2f}, n days {z.height}, range of {col} {z[col].min():.0f}..{z[col].max():.0f}")
z = t
print("daily y-px ~ ae_small share:", np.polyfit(z["ae_small"].to_numpy(), z["y_px"].to_numpy(), 1), "corr", np.corrcoef(z["ae_small"].to_numpy(), z["y_px"].to_numpy())[0, 1], "range ae_small", z["ae_small"].min(), z["ae_small"].max())
print("daily reg-px ~ ae_small share:", np.polyfit(z["ae_small"].to_numpy(), z["reg_px"].to_numpy(), 1))
print(dd.group_by("ym").agg(pl.col("y_px").mean(), pl.col("reg_px").mean(), pl.col("ens_px").mean(), pl.col("sub_px").mean(), pl.col("eupd").mean(), pl.col("ae_small").mean(), pl.col("ae_small").max().alias("ae_small_max")).sort("ym"))
# 2025 days most like July 2026 (highest ae_small)
print(t.sort("ae_small", descending=True).head(10))
dd.write_parquet(f"{S}/eobt_days.parquet")
