"""LFPG: is the recorded off-block tied to the in-block of the previous arrival at the same stand
(own inbound)? Histogram of off-block - in-block (minutes), spike test; arrival schedule link."""
import os, numpy as np, polars as pl
S = os.environ["LFPG_OUT"]
a = pl.read_parquet(f"{S}/lfpg_all.parquet").filter(pl.col("src") == "train")
dep = a.filter(pl.col("PHASE_mvt") == "DEP").drop_nulls(["STAND_mvt", "BLOCK_TIME_UTC_mvt"]).sort("BLOCK_TIME_UTC_mvt")
arr = a.filter(pl.col("PHASE_mvt") == "ARR").drop_nulls(["STAND_mvt", "BLOCK_TIME_UTC_mvt"]).select("STAND_mvt", pl.col("BLOCK_TIME_UTC_mvt").alias("ib"), pl.col("AIRCRAFT_TYPE_mvt").alias("atype"), pl.col("SCHED_TIME_UTC_mvt").alias("asched")).sort("ib")
j = dep.join_asof(arr, left_on="BLOCK_TIME_UTC_mvt", right_on="ib", by="STAND_mvt", strategy="backward")
j = j.with_columns(turn=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("ib")).dt.total_seconds(), same_type=pl.col("atype") == pl.col("AIRCRAFT_TYPE_mvt"))
k = j.filter(pl.col("turn").is_between(0, 6 * 3600))
print("rows with an earlier arrival at the stand within 6 h:", k.height, "of", dep.height, "same type share", k["same_type"].mean())
m = ((k["turn"] + 30) // 60).cast(pl.Int64)
h = np.bincount(np.clip(m.to_numpy(), 0, 360), minlength=361)
print("turn minutes 0..10:", h[:11].tolist())
print("turn minutes 20..70 (every minute):", h[20:71].tolist())
ex = [h[k] - (h[k - 1] + h[k + 1]) / 2 for k in range(1, 359)]
top = sorted(range(1, 359), key=lambda k: -ex[k - 1])[:8]
print("largest single-minute excess:", [(t, h[t], round(ex[t - 1])) for t in top])
