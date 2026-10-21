"""Is there learnable structure left in the LTFM OOF residual? Cross-fold residual boosting on label-free features."""
import polars as pl, numpy as np, lightgbm as lgb
from ltfm_common import *
df = load()
sg = lambda c: pl.when(pl.col(c).dt.second() >= 30).then(pl.col(c).dt.second().cast(pl.Int32) - 60).otherwise(pl.col(c).dt.second().cast(pl.Int32))
df = df.with_columns(h=pl.col("MVT_TIME_UTC_mvt").dt.truncate("1h"), jt=sg("MVT_TIME_UTC_mvt").cast(pl.Float64))
arr = df.filter(pl.col("PHASE_mvt") == "ARR").with_columns(ti=s("BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt"), adel=s("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"))
ha = arr.group_by("h").agg(n_arr=pl.len(), ti_mean=pl.col("ti").mean(), ti_med=pl.col("ti").median(), adel=pl.col("adel").median())
dep = df.filter(pl.col("PHASE_mvt") == "DEP")
hd = dep.group_by("h").agg(n_dep=pl.len(), tol_h=pl.col("to_lobt").median(), tos_h=pl.col("to_sched").median(), px_h=pl.col("px").median(),
                           px_hi=(pl.col("px") >= 1400).mean(), nonm_h=(~pl.col("nm")).mean(), rw36=(pl.col("RUNWAY_mvt").is_in(["36","35L","35R","34L","34R"])).mean())
dd = dep.group_by("day").agg(tol_d=pl.col("to_lobt").median(), tos_d=pl.col("to_sched").median(), n_d=pl.len())
dda = arr.group_by(pl.col("day")).agg(ti_d=pl.col("ti").median(), adel_d=pl.col("adel").median())
d = dep.join(ha, on="h", how="left").join(hd, on="h", how="left").join(dd, on="day", how="left").join(dda, on="day", how="left")
d = d.with_columns(tol_dev=pl.col("to_lobt") - pl.col("tol_h"), stand_cat=pl.col("STAND_mvt").cast(pl.Categorical))
feats = ["jt", "to_lobt", "to_sched", "sched_lobt", "px", "hr", "wd", "n_arr", "ti_mean", "ti_med", "adel", "n_dep", "tol_h", "tos_h", "px_h", "px_hi", "nonm_h", "rw36", "tol_d", "tos_d", "ti_d", "adel_d", "tol_dev"]
o = d.filter(pl.col("pred").is_not_null()).with_columns(r=pl.col("res") - pl.col("res").mean().over("fold"))
tot_b = tot_a = 0
for tr, te in [("jan", "jul"), ("jul", "jan")]:
    A = o.filter(pl.col("fold") == tr); T = o.filter(pl.col("fold") == te)
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=15, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, num_threads=6, verbose=-1)
    m.fit(A.select(feats).to_numpy(), A["r"].to_numpy())
    pt = m.predict(T.select(feats).to_numpy())
    b = (T["r"]**2).sum(); a = ((T["r"] - pt)**2).sum(); tot_b += b; tot_a += a
    imp = sorted(zip(m.feature_importances_, feats), reverse=True)[:8]
    print(tr, "->", te, "SSE/N %.0f -> %.0f" % (b / T.height, a / T.height), imp)
print("total gain/N %.1f, board-scale %.1f" % ((tot_b - tot_a) / o.height, (tot_b - tot_a) / o.height * 47581 / N_BOARD))
# linear: residual on 2*jt only
for tr, te in [("jan", "jul"), ("jul", "jan")]:
    T = o.filter(pl.col("fold") == te)
    print("2jt shift alone on", te, "SSE/N change %.1f" % (((T["r"] - 2 * T["jt"])**2).mean() - (T["r"]**2).mean()))
