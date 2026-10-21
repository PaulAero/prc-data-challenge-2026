"""LFPG July 2026 EOBT_1 regime: composition by AOBT_3 - EOBT_1 bin, target and prediction relative to
take-off - AOBT_3 (px) per bin, 2025 OOF vs 2026 ranking (v18)."""
import os, numpy as np, polars as pl
S = os.environ["LFPG_OUT"]
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(20)
a = pl.read_parquet(f"{S}/lfpg_all.parquet").filter((pl.col("PHASE_mvt") == "DEP") & pl.col("LOBT_flt").is_not_null()).with_columns(
    pl.col("MVT_ID_mvt").cast(pl.Int64), ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"),
    px=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64),
    ae=(pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt")).dt.total_seconds() / 60,
    el=(pl.col("EOBT_1_flt") - pl.col("LOBT_flt")).dt.total_seconds() / 60,
    y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
a = a.with_columns(aeb=pl.col("ae").clip(-10, 30).floor().cast(pl.Int64).map_elements(lambda v: v if -3 <= v <= 5 else (v // 5) * 5, return_dtype=pl.Int64),
                   eupd=(pl.col("el") != 0))
o = pl.read_parquet(f"{S}/oof_ens.parquet").select("MVT_ID_mvt", pl.col("pred").alias("oof"))
s = pl.read_parquet("data/submissions/candidate_v18.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("sub"))
a = a.join(o, on="MVT_ID_mvt", how="left").join(s, on="MVT_ID_mvt", how="left").with_columns(pred=pl.coalesce("oof", "sub"))
ok = pl.col("px").is_between(0, 3600) & ((pl.col("y") < 10800) | pl.col("y").is_null())
x = a.filter(ok & pl.col("ym").is_in(["2025-01", "2025-07", "2026-01", "2026-07"]))
t = x.group_by("ym", "aeb").agg(pl.len().alias("n"), (pl.col("y") - pl.col("px")).mean().alias("y_px"), (pl.col("pred") - pl.col("px")).mean().alias("pred_px"))
t = t.with_columns(sh=pl.col("n") / pl.col("n").sum().over("ym"))
print(t.pivot(on="ym", index="aeb", values=["sh", "y_px", "pred_px"]).sort("aeb"))
print(x.group_by("ym", "eupd").agg(pl.len(), (pl.col("y") - pl.col("px")).mean().alias("y_px"), (pl.col("pred") - pl.col("px")).mean().alias("pred_px"), pl.col("ae").median().alias("ae50")).sort("ym", "eupd"))
# daily share of EOBT updated in July 2026 and 2025
d = a.filter(pl.col("ym").is_in(["2025-07", "2026-01", "2026-07"])).group_by("ym", pl.col("MVT_TIME_UTC_mvt").dt.day().alias("day")).agg(pl.col("eupd").mean()).sort("ym", "day")
for ym in ["2025-07", "2026-01", "2026-07"]:
    print(ym, [round(v, 2) for v in d.filter(pl.col("ym") == ym)["eupd"].to_list()])
