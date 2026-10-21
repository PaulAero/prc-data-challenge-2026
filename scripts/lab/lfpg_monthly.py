"""Monthly LFPG series: target, take-off - AOBT_3, T - LOBT, runway config; ranking predictions of several submissions."""
import os, glob, numpy as np, polars as pl
S = os.environ["LFPG_OUT"]
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(20)
a = pl.read_parquet(f"{S}/lfpg_all.parquet").filter(pl.col("PHASE_mvt") == "DEP").with_columns(
    pl.col("MVT_ID_mvt").cast(pl.Int64), ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"),
    px=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64),
    y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
    north=pl.col("RUNWAY_mvt").is_in(["27L", "09R", "27R", "09L"]), west=pl.col("RUNWAY_mvt").is_in(["26R", "27L", "26L", "27R"]),
    inner=pl.col("RUNWAY_mvt").is_in(["27L", "09R", "26R", "08L"]))
core = pl.col("y") < 10800
pxc = pl.col("px").is_between(0, 3600)
print(a.group_by("ym").agg(pl.len(), pl.col("y").filter(core).mean().alias("y_mean"), pl.col("y").median().alias("y50"),
    pl.col("px").filter(pxc).mean().alias("px_mean"), pl.col("px").median().alias("px50"),
    (pl.col("y") - pl.col("px")).filter(core & pxc).mean().alias("y_minus_px"),
    pl.col("north").mean().alias("north"), pl.col("west").mean().alias("west"), pl.col("inner").mean().alias("inner_rwy")).sort("ym"))
subs = {"v11(no adsb)": "data/submissions/quick-quicksand_v11.parquet", "v18": "data/submissions/candidate_v18.parquet", "v5": "data/submissions/quick-quicksand_v5.parquet"}
r = a.filter(pl.col("src") == "rank")
for k, p in subs.items():
    if not os.path.exists(p):
        continue
    s = pl.read_parquet(p)
    z = r.join(s.select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("pred")), on="MVT_ID_mvt")
    print(k, z.group_by("ym").agg(pl.col("pred").mean().alias("mean"), pl.col("pred").median().alias("p50"), (pl.col("pred") - pl.col("px")).filter(pxc).mean().alias("pred_minus_px")).sort("ym").rows())
o = pl.read_parquet(f"{S}/oof_ens.parquet").select("MVT_ID_mvt", "pred")
z = a.join(o, on="MVT_ID_mvt")
print("OOF 2025", z.group_by("ym").agg(pl.col("pred").mean().alias("mean"), (pl.col("pred") - pl.col("px")).filter(pxc).mean().alias("pred_minus_px")).sort("ym").rows())
