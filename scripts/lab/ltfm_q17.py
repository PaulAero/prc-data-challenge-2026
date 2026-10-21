"""Soft upper bound from A-CDM: off-block rarely earlier than LOBT - 5 min, i.e. y <= to_lobt + 300."""
import polars as pl, numpy as np
from ltfm_common import *
x = pl.read_parquet("experiments/lab-year/parts.parquet").filter(pl.col("airport") == "LTFM")
x = x.with_columns(p=(pl.col("p_base") + pl.col("p_wsq")) / 2).with_columns(mix=pl.col("p") * pl.col("mvt_vs_schedule_sec") + (1 - pl.col("p")) * pl.col("reg"))
x = x.with_columns(res=pl.col("target") - pl.col("mix"), tol=pl.col("mvt_vs_lobt_sec"), early=(pl.col("target") - pl.col("mvt_vs_lobt_sec")))  # early = LOBT - block
x = x.filter(pl.col("tol").is_not_null() & (pl.col("target") <= 10800))
print("share block earlier than LOBT-5min:", (x["early"] > 330).mean(), " LOBT-10min:", (x["early"] > 630).mean())
x = x.with_columns(slack=pl.col("tol") + 300 - pl.col("mix"))  # negative = prediction implies off-block before LOBT-5
print(x.with_columns(sb=pl.col("slack").cut([-600, -300, -120, 0, 120, 300, 600])).group_by("sb").agg(n=pl.len(), mres=pl.col("res").mean(),
     rmse=(pl.col("res")**2).mean().sqrt(), viol=(pl.col("early") > 330).mean()).sort("sb"))
# does conditioning on slack bins (LOBO) help? fit mean residual per slack bin per block-out
x = x.with_columns(sb=pl.col("slack").cut([-600, -300, -120, 0, 120, 300, 600]).cast(pl.Utf8))
sse_b = (x["res"]**2).sum(); sse_a = 0
for b in range(6):
    g = x.filter(pl.col("block") != b).group_by("sb").agg(c=pl.col("res").sum() / (pl.len() + 50))
    te = x.filter(pl.col("block") == b).join(g, on="sb", how="left")
    sse_a += ((te["res"] - te["c"].fill_null(0))**2).sum()
print("slack-bin correction: SSE/N gain %.1f, board-scale (LTFM ranking 47,581 rows) %.1f" % ((sse_b - sse_a) / x.height, (sse_b - sse_a) / x.height * 47581 / N_BOARD))
