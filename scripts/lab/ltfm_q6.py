import polars as pl
from ltfm_common import *
df = load()
d = df.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("src") == "train"))
print("y tail counts", [(t, d.filter(pl.col("y") > t).height) for t in [2700, 3600, 5400, 7200, 9000]])
print(d.filter(pl.col("y") > 5400).select("day", "hr", "FLIGHT_mvt", "y", "to_sched", "to_lobt", "px", "RUNWAY_mvt", "STAND_mvt").sort("y").tail(30))
# daily count of y>3600
print(d.filter(pl.col("y") > 3600).group_by("day").agg(n=pl.len(), ymax=pl.col("y").max()).sort("n", descending=True).head(15))
# y histogram 600-s bins up to 12000 at 60-s near 3600 multiples
print(d.with_columns(b=(pl.col("y")/600).floor()*600)["b"].value_counts().sort("b").filter(pl.col("b")>=2400))
p = pl.read_parquet("experiments/v15-lgb/predictions.parquet").filter(pl.col("airport")=="LTFM")
print(p.select(pl.col("fill").mean(), pl.col("no_nm").mean()))
