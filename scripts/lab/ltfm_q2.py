import polars as pl
from ltfm_common import *
df = load()
d = df.filter(pl.col("pred").is_not_null())
print("rows", d.height, "rmse all", (d["res"]**2).mean()**0.5)
core = d.filter(pl.col("target") <= 10800)
print("rmse core", (core["res"]**2).mean()**0.5, "rmse nonfill core", (core.filter(~pl.col("fill"))["res"]**2).mean()**0.5)
# SSE decomposition
d = d.with_columns(tb=pl.col("target").cut([600, 1200, 1800, 2700, 3600, 7200]))
print(d.group_by("fill", "tb").agg(n=pl.len(), sse_share=(pl.col("res")**2).sum(), mres=pl.col("res").mean(), rmse=(pl.col("res")**2).mean().sqrt()).with_columns(pl.col("sse_share")/ (d["res"]**2).sum()).sort("fill", "tb"))
print(d.group_by("fold").agg(n=pl.len(), rmse=(pl.col("res")**2).mean().sqrt(), mres=pl.col("res").mean()))
dd = d.with_columns(to_sched_b=pl.col("to_sched").cut([0, 600, 1200, 1800, 3600, 7200]))
print(dd.group_by("to_sched_b").agg(n=pl.len(), fillr=pl.col("fill").mean(), rmse=(pl.col("res")**2).mean().sqrt(), sse=(pl.col("res")**2).sum()/(d["res"]**2).sum()).sort("to_sched_b"))
# group gains
D = d.filter(pl.col("target") <= 10800)
base = (D["res"]**2).sum()
print("core SSE", base, "N", D.height)
for keys in [["al"], ["cs3"], ["stand_area"], ["STAND_mvt"], ["RUNWAY_mvt"], ["hr"], ["wd"], ["day"], ["AIRCRAFT_TYPE_mvt"], ["ADES_mvt"], ["RUNWAY_mvt","stand_area"], ["al","hr"], ["nm"], ["FLIGHT_mvt"]]:
    b, a = oof_group_gain(D.with_columns(day=pl.col("day").dt.day()) if keys==["day"] else D, keys)
    print(keys, "SSE/N before %.0f after %.0f gain/N %.1f  board-scale (x rows/N_BOARD) %.1f" % (b/D.height, a/D.height, (b-a)/D.height, (b-a)/N_BOARD))
