"""LFPG-only residual corrector on the 8-model OOF average, cross-fitted jan <-> jul 2025.
Label-free features (record and NM timing, stand/runway, minute structure). Measures structure left."""
import os, numpy as np, polars as pl, lightgbm as lgb
S = os.environ["LFPG_OUT"]
N = 344_841
d = pl.read_parquet(f"{S}/res25.parquet")
s = lambda a, b: (pl.col(a) - pl.col(b)).dt.total_seconds().cast(pl.Float64)
d = d.with_columns(px=s("MVT_TIME_UTC_mvt", "AOBT_3_flt"), tl=s("MVT_TIME_UTC_mvt", "LOBT_flt"), te=s("MVT_TIME_UTC_mvt", "EOBT_1_flt"),
    ts=s("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"), ae=s("AOBT_3_flt", "EOBT_1_flt"), el=s("EOBT_1_flt", "LOBT_flt"), ls=s("LOBT_flt", "SCHED_TIME_UTC_mvt"),
    ar=s("ARVT_3_flt", "MVT_TIME_UTC_mvt"), a1=s("ARVT_1_flt", "ARVT_3_flt"),
    amin=pl.col("AOBT_3_flt").dt.minute().cast(pl.Float64), emin=pl.col("EOBT_1_flt").dt.minute().cast(pl.Float64),
    stp_c=pl.col("stp").cast(pl.Categorical), rw_c=pl.col("rw").cast(pl.Categorical), al_c=pl.col("al").fill_null("NA").cast(pl.Categorical),
    st_c=pl.col("st").cast(pl.Categorical), ty_c=pl.col("AIRCRAFT_TYPE_mvt").fill_null("NA").cast(pl.Categorical),
    wd=pl.col("MVT_TIME_UTC_mvt").dt.weekday().cast(pl.Float64), lhf=pl.col("lh").cast(pl.Float64), tminf=pl.col("tmin").cast(pl.Float64))
feats = ["px", "tl", "te", "ts", "ae", "el", "ls", "ar", "a1", "amin", "emin", "stp_c", "rw_c", "al_c", "st_c", "ty_c", "wd", "lhf", "tminf", "pred"]
params = dict(objective="huber", alpha=300.0, learning_rate=0.03, num_leaves=15, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=10.0, num_threads=8, verbose=-1, seed=1, max_cat_to_onehot=8, cat_smooth=50)
tot = {}
for tr, te in [("jan", "jul"), ("jul", "jan")]:
    A = d.filter((pl.col("fold") == tr) & pl.col("core")); B = d.filter(pl.col("fold") == te)
    XA = A.select(feats).to_pandas(); XB = B.select(feats).to_pandas()
    for rounds in [100, 300]:
        m = lgb.train(params, lgb.Dataset(XA, A["r"].clip(-1800, 1800).to_numpy()), num_boost_round=rounds)
        c = m.predict(XB)
        r = B["r"].to_numpy(); core = B["core"].to_numpy()
        for k, msk in [("all", np.ones(len(r), bool)), ("core", core)]:
            dd = ((r - c) ** 2 - r ** 2)[msk].sum()
            tot[(rounds, k)] = tot.get((rounds, k), 0) + dd
            print(te, rounds, k, f"dSSE/N {dd / N:+.1f}", "rmse before", np.sqrt((r[msk] ** 2).mean()).round(2), "after", np.sqrt(((r - c)[msk] ** 2).mean()).round(2))
    imp = sorted(zip(m.feature_importance("gain"), feats), reverse=True)[:8]
    print("top gain", [(f, round(g / 1e6)) for g, f in imp])
print({k: round(v / N, 1) for k, v in tot.items()})
