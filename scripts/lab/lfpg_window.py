"""Matched LFPG departures that take off long after LOBT: where does the recorded off-block sit?"""
import os, numpy as np, polars as pl
S = os.environ["LFPG_OUT"]
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(16); pl.Config.set_tbl_width_chars(250)
d = pl.read_parquet(f"{S}/res25.parquet").filter(pl.col("nm"))
d = d.with_columns(tl=(pl.col("MVT_TIME_UTC_mvt") - pl.col("LOBT_flt")).dt.total_seconds(),
                   ul=(pl.col("AOBT_3_flt") - pl.col("LOBT_flt")).dt.total_seconds(),
                   bl=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("LOBT_flt")).dt.total_seconds())
d = d.with_columns(bl_pred=pl.col("tl") - pl.col("pred"))
d = d.with_columns(tlb=(pl.col("tl") // 900 * 15).clip(-15, 180), ulb=(pl.col("ul") // 900 * 15).clip(-30, 120))
print(d.group_by("tlb").agg(pl.len(), pl.col("bl").quantile(0.1).alias("bl10"), pl.col("bl").median().alias("bl50"), pl.col("bl").quantile(0.9).alias("bl90"),
      pl.col("bl_pred").median().alias("blp50"), pl.col("r").mean().alias("mr"), (pl.col("r") ** 2).sum().alias("sse"),
      pl.col("ul").median().alias("ul50")).sort("tlb"))
x = d.filter(pl.col("tl") > 3600)
print("T-LOBT>60min, by AOBT3-LOBT bin")
print(x.group_by("ulb").agg(pl.len(), pl.col("bl").quantile(0.1).alias("bl10"), pl.col("bl").median().alias("bl50"), pl.col("bl").quantile(0.9).alias("bl90"),
      pl.col("bl_pred").median().alias("blp50"), pl.col("r").mean().alias("mr"), (pl.col("r") ** 2).sum().alias("sse")).sort("ulb"))
# block vs AOBT_3 for those rows
x = x.with_columns(ba=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds())
print("block-AOBT3 for T-LOBT>60, ul<=60min:", x.filter(pl.col("ul") <= 3600)["ba"].quantile(0.1), x.filter(pl.col("ul") <= 3600)["ba"].median(), x.filter(pl.col("ul") <= 3600)["ba"].quantile(0.9))
print("block-AOBT3 for T-LOBT>60, ul>60min:", x.filter(pl.col("ul") > 3600)["ba"].quantile(0.1), x.filter(pl.col("ul") > 3600)["ba"].median(), x.filter(pl.col("ul") > 3600)["ba"].quantile(0.9))
# block - AOBT3 general relation for ordinary rows: by ul bin
d = d.with_columns(ba=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds())
print(d.group_by("ulb").agg(pl.len(), pl.col("ba").quantile(0.1).alias("ba10"), pl.col("ba").median().alias("ba50"), pl.col("ba").quantile(0.9).alias("ba90"), (pl.col("ba").abs() <= 300).mean().alias("ba_in5")).sort("ulb"))
