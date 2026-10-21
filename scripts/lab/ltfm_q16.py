"""Full-year OOF parts: (a) day/hour-level delay correction, leave-one-block-out; (b) 2jt jitter correction at the five jitter airports."""
import polars as pl, numpy as np
from ltfm_common import *
x = pl.read_parquet("experiments/lab-year/parts.parquet")
x = x.with_columns(p=(pl.col("p_base") + pl.col("p_wsq")) / 2).with_columns(mix=pl.col("p") * pl.col("mvt_vs_schedule_sec") + (1 - pl.col("p")) * pl.col("reg"))
x = x.with_columns(res=pl.col("target") - pl.col("mix"))
# (b) jitter
raw = pl.concat([pl.scan_parquet(f).filter(pl.col("PHASE_mvt") == "DEP").select(pl.col("MVT_ID_mvt").cast(pl.Int64), "MVT_TIME_UTC_mvt").collect()
                 for f in sorted(__import__("glob").glob("data/raw/training_2025-*.parquet"))])
sg = pl.when(pl.col("MVT_TIME_UTC_mvt").dt.second() >= 30).then(pl.col("MVT_TIME_UTC_mvt").dt.second().cast(pl.Int32) - 60).otherwise(pl.col("MVT_TIME_UTC_mvt").dt.second().cast(pl.Int32))
x = x.join(raw.with_columns(jt=sg.cast(pl.Float64)).select("MVT_ID_mvt", "jt"), on="MVT_ID_mvt", how="left")
JA = ["EDDM", "EGLL", "LFPG", "LIRF", "LTFM"]
tot = 0
for a in JA:
    c = x.filter((pl.col("airport") == a) & (pl.col("target") <= 10800))
    bp = np.cov(c["jt"], c["mix"])[0, 1] / c["jt"].var(); by = np.cov(c["jt"], c["target"])[0, 1] / c["jt"].var()
    # correction: mix + (2 - bp_block) * jt where bp estimated leave-one-block-out
    sse_b = (c["res"] ** 2).sum(); sse_a = 0
    for b in c["block"].unique():
        tr = c.filter(pl.col("block") != b); te = c.filter(pl.col("block") == b)
        k = 2 - np.cov(tr["jt"], tr["mix"])[0, 1] / tr["jt"].var()
        sse_a += ((te["res"] - k * te["jt"]) ** 2).sum()
    n_rank = None
    print("%s slope target %.2f pred %.2f; core SSE/N change %.1f (n=%d)" % (a, by, bp, (sse_a - sse_b) / c.height, c.height))
    tot += (sse_a - sse_b)
print("five airports: year-scale MSE change %.2f (SSE / 2,085,047)" % (tot / 2_085_047))
# (a) day-level delay correction at LTFM
l = x.filter(pl.col("airport") == "LTFM").with_columns(day=pl.col("event_time").dt.convert_time_zone("Europe/Istanbul").dt.date(), h=pl.col("event_time").dt.truncate("1h"))
for lvl in ["day", "h"]:
    g = l.group_by(lvl).agg(tos=pl.col("mvt_vs_schedule_sec").median(), tol=pl.col("mvt_vs_lobt_sec").median(), n=pl.len(), mres=pl.col("res").mean(), blk=pl.col("block").first())
    l2 = l.join(g.select(lvl, "tos", "tol", "n"), on=lvl)
    sse_b = (l2.filter(pl.col("target") <= 10800)["res"] ** 2).sum(); sse_a = 0; coefs = []
    for b in range(6):
        tr = g.filter(pl.col("blk") != b)
        X = np.column_stack([np.ones(tr.height), tr["tos"], tr["tol"], tr["n"]]); w = tr["n"].to_numpy()
        beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], tr["mres"].to_numpy() * np.sqrt(w), rcond=None)[0]; coefs.append(beta)
        te = l2.filter((pl.col("block") == b) & (pl.col("target") <= 10800))
        corr = np.column_stack([np.ones(te.height), te["tos"], te["tol"], te["n"]]) @ beta
        sse_a += ((te["res"] - corr) ** 2).sum()
    nn = l2.filter(pl.col("target") <= 10800).height
    print(lvl, "LTFM core SSE/N %.0f -> %.0f (gain %.1f); coef (1,tos,tol,n) mean" % (sse_b / nn, sse_a / nn, (sse_b - sse_a) / nn), np.round(np.mean(coefs, 0), 3), "sd", np.round(np.std(coefs, 0), 3))
