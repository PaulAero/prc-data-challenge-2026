"""Full-year LOBO group-mean corrections of the LTFM residual (mix of the year OOF parts)."""
import polars as pl, numpy as np
from ltfm_common import *
x = pl.read_parquet("experiments/lab-year/parts.parquet").filter(pl.col("airport") == "LTFM")
x = x.with_columns(p=(pl.col("p_base") + pl.col("p_wsq")) / 2).with_columns(mix=pl.col("p") * pl.col("mvt_vs_schedule_sec") + (1 - pl.col("p")) * pl.col("reg"))
x = x.with_columns(res=pl.col("target") - pl.col("mix")).filter(pl.col("target") <= 10800)
df = load().filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("src") == "train")).select("MVT_ID_mvt", "STAND_mvt", "RUNWAY_mvt", "stand_area", "cs3", "AIRCRAFT_TYPE_mvt", "hr", "wd", "px", "ADES_mvt",
    tomin=pl.col("MVT_TIME_UTC_mvt").dt.minute(), schmin=pl.col("SCHED_TIME_UTC_mvt").dt.minute(), m=pl.col("m"))
x = x.join(df, on="MVT_ID_mvt").with_columns(pxr=(pl.col("px") / 60).round() * 60)
print("rows", x.height, "core RMSE %.1f" % ((x["res"]**2).mean()**0.5))
print("block means of residual:", x.group_by("block").agg(pl.col("res").mean()).sort("block")["res"].round(1).to_list())
def lobo(keys, shrink=50):
    sse_b = (x["res"]**2).sum(); sse_a = 0
    for b in range(6):
        g = x.filter(pl.col("block") != b).group_by(keys).agg(c=pl.col("res").sum() / (pl.len() + shrink))
        te = x.filter(pl.col("block") == b).join(g, on=keys, how="left")
        sse_a += ((te["res"] - te["c"].fill_null(0))**2).sum()
    return (sse_b - sse_a) / x.height
for keys in [["STAND_mvt"], ["RUNWAY_mvt"], ["STAND_mvt", "RUNWAY_mvt"], ["stand_area", "RUNWAY_mvt"], ["cs3"], ["AIRCRAFT_TYPE_mvt"], ["hr"], ["hr", "RUNWAY_mvt"], ["wd"], ["pxr"], ["pxr", "RUNWAY_mvt"], ["ADES_mvt"], ["tomin"], ["schmin"], ["m"]]:
    gn = lobo(keys); print(keys, "SSE/N gain %.1f  board-scale %.1f" % (gn, gn * 47581 / N_BOARD))
