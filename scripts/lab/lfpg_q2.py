"""Q2: structure of the 8-model OOF residual at LFPG (jan/jul 2025), cross-fitted group corrections."""
import os, numpy as np, polars as pl
S = os.environ["LFPG_OUT"]
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(20); pl.Config.set_fmt_str_lengths(40)
N = 344_841
d = pl.read_parquet(f"{S}/dep25.parquet").drop("y")
o = pl.read_parquet(f"{S}/oof_ens.parquet")
print("total OOF MSE", ((o["target"] - o["pred"]) ** 2).mean(), "rows", o.height)
d = d.join(o.select("MVT_ID_mvt", "fold", "target", "pred", "fill"), on="MVT_ID_mvt", how="inner")
d = d.with_columns(r=pl.col("target") - pl.col("pred"),
    lh=pl.col("MVT_TIME_UTC_mvt").dt.convert_time_zone("Europe/Paris").dt.hour(),
    tmin=pl.col("MVT_TIME_UTC_mvt").dt.minute(), smin=pl.col("SCHED_TIME_UTC_mvt").dt.minute(),
    afr=pl.col("al") == "AFR",
    px=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds(),
    core=pl.col("target") <= 10800)
print("LFPG rows", d.height, "SSE share of all OOF", (d["r"] ** 2).sum() / ((o["target"] - o["pred"]) ** 2).sum())
for f in ["jan", "jul"]:
    x = d.filter(pl.col("fold") == f)
    print(f, "n", x.height, "rmse all", np.sqrt((x["r"] ** 2).mean()), "rmse core", np.sqrt((x.filter("core")["r"] ** 2).mean()), "mean r", x["r"].mean(), "median r", x["r"].median())
# top error rows
0 and print(d.sort(pl.col("r").abs(), descending=True).select("fold", "FLIGHT_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "LOBT_flt", "AOBT_3_flt", "STAND_mvt", "target", "pred").head(15))
tot_sse = (d["r"] ** 2).sum()
print("top3 share of LFPG SSE", (d.sort(pl.col("r").abs(), descending=True).head(3)["r"] ** 2).sum() / tot_sse)

def crossfit(key, k=200, clip=None, df=None, label=None):
    df = d if df is None else df
    """shrunk group mean of residual fitted on one fold, applied on the other; core rows only for fitting"""
    res = []
    for tr, te in [("jan", "jul"), ("jul", "jan")]:
        a = df.filter((pl.col("fold") == tr) & pl.col("core"))
        g = a.group_by(key).agg(pl.col("r").sum().alias("s"), pl.len().alias("n")).with_columns(c=pl.col("s") / (pl.col("n") + k))
        b = df.filter(pl.col("fold") == te).join(g.select(*(key if isinstance(key, list) else [key]), "c"), on=key, how="left").with_columns(pl.col("c").fill_null(0))
        for pop in ["all", "core"]:
            bb = b if pop == "all" else b.filter("core")
            res.append((te, pop, ((bb["r"] - bb["c"]) ** 2).sum() - (bb["r"] ** 2).sum()))
    out = {}
    for te, pop, dd in res:
        out[(te, pop)] = dd
    tot_all = out[("jan", "all")] + out[("jul", "all")]
    tot_core = out[("jan", "core")] + out[("jul", "core")]
    print(f"{label or key}: dSSE/N all {tot_all / N:+.1f} (jan {out[('jan','all')]/N:+.1f}, jul {out[('jul','all')]/N:+.1f}); core {tot_core / N:+.1f}")

def table(key, df=None):
    df = d if df is None else df
    t = df.filter("core").group_by("fold", key).agg(pl.len().alias("n"), pl.col("r").mean().alias("mr")).pivot(on="fold", index=key, values=["n", "mr"]).sort("n_jan", descending=True, nulls_last=True)
    print(t.head(30))

d = d.with_columns(al_lobt=((pl.col("AOBT_3_flt") - pl.col("LOBT_flt")).dt.total_seconds() / 600).floor().clip(-3, 12),
    t_lobt=((pl.col("MVT_TIME_UTC_mvt") - pl.col("LOBT_flt")).dt.total_seconds() / 600).floor().clip(-3, 15))
d = d.with_columns(stp=pl.col("STAND_mvt").str.slice(0, 1).fill_null("NA"), rw=pl.col("RUNWAY_mvt").fill_null("NA"),
    pxb=(pl.col("px") / 120).floor().clip(-5, 20), nmflag=pl.col("LOBT_flt").is_not_null(),
    st=pl.col("STAND_mvt").fill_null("NA"))
for key in ["al_lobt", "t_lobt", "afr", "stp", "rw", "lh", "tmin", "smin", "pxb", "nmflag", "AIRCRAFT_TYPE_mvt", "al", "st"]:
    table(key) if key in ["stp", "rw", "lh", "afr", "pxb", "al_lobt", "t_lobt"] else None
    for k in [50, 200, 1000]:
        crossfit(key, k=k, label=f"{key} k={k}")
for key in [["stp", "rw"], ["afr", "stp"], ["stp", "lh"], ["rw", "lh"], ["st", "rw"]]:
    crossfit(key, k=200, label=str(key))
d.write_parquet(f"{S}/res25.parquet")
