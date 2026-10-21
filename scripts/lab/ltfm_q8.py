import polars as pl, numpy as np
from ltfm_common import *
df = load()
arr = df.filter(pl.col("PHASE_mvt") == "ARR").select(
    pl.col("STAND_mvt"), ib=pl.col("BLOCK_TIME_UTC_mvt"), land=pl.col("MVT_TIME_UTC_mvt"), asched=pl.col("SCHED_TIME_UTC_mvt"),
    afl=pl.col("FLIGHT_mvt"), asrc=pl.col("src")).filter(pl.col("ib").is_not_null() & pl.col("STAND_mvt").is_not_null())
print("arrivals: in-block null share by src", df.filter(pl.col("PHASE_mvt")=="ARR").group_by("src").agg(pl.col("BLOCK_TIME_UTC_mvt").is_null().mean()))
a2 = df.filter(pl.col("PHASE_mvt")=="ARR").with_columns(ti=s("BLOCK_TIME_UTC_mvt","MVT_TIME_UTC_mvt"), ib_sched=(s("BLOCK_TIME_UTC_mvt","SCHED_TIME_UTC_mvt").abs()<=10))
print("arrivals by src/month: taxi-in median, ib==sched share, mean ti")
print(a2.group_by("src","m").agg(n=pl.len(), ti_med=pl.col("ti").median(), ti_mean=pl.col("ti").mean(), ibsch=pl.col("ib_sched").mean(), ti_gt30m=(pl.col("ti")>1800).mean()).sort("src","m"))
dep = df.filter(pl.col("PHASE_mvt") == "DEP").sort("MVT_TIME_UTC_mvt")
arr = arr.sort("ib")
# previous arrival at same stand with in-block before the take-off
j = dep.join_asof(arr, left_on="MVT_TIME_UTC_mvt", right_on="ib", by="STAND_mvt", strategy="backward")
j = j.with_columns(b_ib=s("BLOCK_TIME_UTC_mvt", "ib"), to_ib=s("MVT_TIME_UTC_mvt", "ib"))
t = j.filter(pl.col("src") == "train")
print("share with prev arrival found", t["ib"].is_not_null().mean())
# when to_ib < y, the prev arrival blocked in after my off-block, i.e., it is not my predecessor
h = t.filter(pl.col("b_ib").is_not_null()).with_columns(b=(pl.col("b_ib")/60).round())
print("block - prev in-block (min), -5..60:"); vc = h.filter(pl.col("b").is_between(-5, 60))["b"].value_counts().sort("b"); print(vc["b"].to_list()); print(vc["count"].to_list())
print("share block < prev in-block", (h["b_ib"] < 0).mean())
