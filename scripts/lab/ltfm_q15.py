import polars as pl, numpy as np
from ltfm_common import *
x = pl.read_parquet("experiments/lab-year/parts.parquet").filter(pl.col("airport") == "LTFM")
x = x.with_columns(p=(pl.col("p_base") + pl.col("p_wsq")) / 2).with_columns(mix=pl.col("p") * pl.col("mvt_vs_schedule_sec") + (1 - pl.col("p")) * pl.col("reg"))
c = x.filter(pl.col("target") <= 10800)
for col in ["mix", "reg", "nn_pred"]:
    print(col, "core RMSE %.1f" % (((c["target"] - c[col]) ** 2).mean() ** 0.5))
print("p mean %.3f, fill rate %.3f" % (c["p"].mean(), c["fill"].mean()))
# calibration of p at LTFM by bins
print(c.with_columns(pb=pl.col("p").cut([0.02, 0.05, 0.1, 0.2, 0.3, 0.5])).group_by("pb").agg(n=pl.len(), p=pl.col("p").mean(), fill=pl.col("fill").mean()).sort("pb"))
# Is the fill point mass real at LTFM? compare to a smooth density: P(block minute == sched minute) vs P(== sched +-1 min) given to_sched
