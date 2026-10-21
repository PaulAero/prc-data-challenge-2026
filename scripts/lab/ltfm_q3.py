import polars as pl
from ltfm_common import *
df = load()
d = df.filter(pl.col("pred").is_not_null())
D = d.filter(pl.col("target") <= 10800)
print("fold-demeaned group corrections (fold mean removed: the season shift itself is SSE/N",
      ((D["res"]**2).mean() - (D.with_columns(r=pl.col("res")-pl.col("res").mean().over("fold"))["r"]**2).mean()), ")")
for keys in [["al"], ["stand_area"], ["STAND_mvt"], ["RUNWAY_mvt"], ["hr"], ["wd"], ["AIRCRAFT_TYPE_mvt"], ["ADES_mvt"], ["RUNWAY_mvt","stand_area"], ["al","hr"], ["nm"], ["FLIGHT_mvt"], ["fill"]]:
    b, a = oof_group_gain(D, keys)
    print(keys, "SSE/N before %.0f after %.0f gain/N %.1f  board-scale %.1f" % (b/D.height, a/D.height, (b-a)/D.height, (b-a)/N_BOARD))
# minute of hour of take-off / schedule
D = D.with_columns(to_min=pl.col("MVT_TIME_UTC_mvt").dt.minute(), sch_min=pl.col("SCHED_TIME_UTC_mvt").dt.minute())
for keys in [["to_min"], ["sch_min"]]:
    b, a = oof_group_gain(D, keys)
    print(keys, "gain/N %.1f board %.1f" % ((b-a)/D.height, (b-a)/N_BOARD))
print(D.group_by("sch_min").agg(n=pl.len(), mres=pl.col("res").mean(), fillr=pl.col("fill").mean()).sort("n", descending=True).head(12))
