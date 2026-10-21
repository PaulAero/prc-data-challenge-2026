"""Per month: point-mass excess at schedule / LOBT minute, early-pushback share, no-NM share, y tail."""
import polars as pl
from ltfm_common import *
d = load().filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("src") == "train")).with_columns(
    bs=(pl.col("d_sched") / 60).round(), bl=(s("BLOCK_TIME_UTC_mvt", "LOBT_flt") / 60).round())
out = d.group_by("m").agg(n=pl.len(),
    at_sched=(pl.col("bs") == 0).mean(), nb_sched=((pl.col("bs") == -1) | (pl.col("bs") == 1)).mean() / 2,
    at_lobt=(pl.col("bl") == 0).mean(), nb_lobt=((pl.col("bl") == -1) | (pl.col("bl") == 1)).mean() / 2,
    early5=(pl.col("bl") < -5).mean(), nonm=(~pl.col("nm")).mean(), y_gt3600=(pl.col("y") > 3600).mean(), ymax=pl.col("y").max()
).with_columns(excess_sched=pl.col("at_sched") - pl.col("nb_sched"), excess_lobt=pl.col("at_lobt") - pl.col("nb_lobt")).sort("m")
print(out.select("m", "n", "at_sched", "excess_sched", "at_lobt", "excess_lobt", "early5", "nonm", "y_gt3600", "ymax"))
