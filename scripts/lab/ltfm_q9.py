import polars as pl, numpy as np
from ltfm_common import *
df = load()
arr = df.filter(pl.col("PHASE_mvt") == "ARR").select("STAND_mvt", ib=pl.col("BLOCK_TIME_UTC_mvt")).filter(pl.col("STAND_mvt").is_not_null()).sort("ib")
dep = df.filter(pl.col("PHASE_mvt") == "DEP").sort("MVT_TIME_UTC_mvt")
j = dep.join_asof(arr, left_on="MVT_TIME_UTC_mvt", right_on="ib", by="STAND_mvt", strategy="backward")
j = j.with_columns(to_ib=s("MVT_TIME_UTC_mvt", "ib"), b_ib=s("BLOCK_TIME_UTC_mvt", "ib"))
t = j.filter(pl.col("src") == "train" & pl.col("ib").is_not_null()) if False else j.filter((pl.col("src") == "train") & pl.col("ib").is_not_null())
t = t.with_columns(succ=pl.col("b_ib") < 0)
# bounds: successor => ib - block >= ?; predecessor => block - ib >= ?
print("successor: ib-block quantiles", [(-t.filter(pl.col("succ"))["b_ib"]).quantile(q) for q in [0, .001, .01, .05, .5]])
print("predecessor: block-ib quantiles", [t.filter(~pl.col("succ"))["b_ib"].quantile(q) for q in [0, .001, .01, .05, .5]])
# forbidden interval in y: (to_ib - P, to_ib + S)
P, S = 1500, 100
o = j.filter(pl.col("pred").is_not_null() & pl.col("ib").is_not_null())
o = o.with_columns(lo=pl.col("to_ib") - P, hi=pl.col("to_ib") + S)
o = o.with_columns(inside=(pl.col("pred") > pl.col("lo")) & (pl.col("pred") < pl.col("hi")), ytrue_inside=(pl.col("target") > pl.col("lo")) & (pl.col("target") < pl.col("hi")))
print("pred inside forbidden:", o["inside"].sum(), "of", o.height, " true inside:", o["ytrue_inside"].sum())
ins = o.filter(pl.col("inside"))
print(ins.select(pl.col("to_ib").quantile(.5), pl.col("pred").quantile(.5), pl.col("target").quantile(.5), (pl.col("target") >= pl.col("hi")).mean().alias("above"), (pl.col("res")**2).mean().alias("mse")))
# Projection to hi (successor reading) when pred close to hi, to lo otherwise -> try simple: snap to nearest bound
for rule in ["hi", "nearest"]:
    if rule == "hi": newp = ins["hi"]
    else: newp = pl.Series(np.where((ins["hi"] - ins["pred"]) < (ins["pred"] - ins["lo"]), ins["hi"], ins["lo"]))
    b = (ins["res"]**2).sum(); a = ((ins["target"] - newp)**2).sum()
    print(rule, "SSE change", a - b, "per jan+jul row", (a-b)/o.height, "board-scale (x 47581/46539)", (a-b)*47581/46539/N_BOARD)
# general: the bound y >= to_ib + S whenever pred is under it and to_ib small (arrival just before take-off)
