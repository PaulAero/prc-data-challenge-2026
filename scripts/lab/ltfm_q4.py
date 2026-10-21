import polars as pl
from ltfm_common import *
df = load()
d = df.filter(pl.col("PHASE_mvt") == "DEP")
d = d.with_columns(d_aobt=s("BLOCK_TIME_UTC_mvt", "AOBT_3_flt"), d_lobt=s("BLOCK_TIME_UTC_mvt", "LOBT_flt"),
                   aobt_sec=pl.col("AOBT_3_flt").dt.second(), pxr=(pl.col("px") / 10).round() * 10)
# top px values overall and per src
for src in ["train", "rank"]:
    x = d.filter(pl.col("src") == src)
    print(src, "px (10 s) top values"); print(x["pxr"].value_counts().sort("count", descending=True).head(12).with_columns(share=pl.col("count") / x.height))
    print(src, "AOBT_3 second-of-minute top"); print(x["aobt_sec"].value_counts().sort("count", descending=True).head(5).with_columns(share=pl.col("count") / x.height))
# day default
dd = d.group_by("day").agg(pl.col("pxr").mode().first().alias("dflt"))
d = d.join(dd, on="day").with_columns(is_dflt=(pl.col("pxr") - pl.col("dflt")).abs() <= 20)
t = d.filter(pl.col("src") == "train")
print("train: default share", t["is_dflt"].mean())
print(t.group_by("is_dflt", "nm").agg(n=pl.len(), eq10=(pl.col("d_aobt").abs() <= 10).mean(), eq60=(pl.col("d_aobt").abs() <= 60).mean(),
      eq120=(pl.col("d_aobt").abs() <= 120).mean(), med_abs=pl.col("d_aobt").abs().median(), fill=pl.col("fill").mean()).sort("is_dflt", "nm"))
# non-default rows: d_aobt histogram in 60 s bins
nd = t.filter(~pl.col("is_dflt") & pl.col("nm"))
print("non-default: d_aobt 60s bins"); print(nd.with_columns(b=(pl.col("d_aobt") / 60).round().clip(-20, 20))["b"].value_counts().sort("b"))
print("non-default: px quantiles", [nd["px"].quantile(q) for q in [0.05, .25, .5, .75, .95]])
print("non-default, aobt seconds:", nd["aobt_sec"].value_counts().sort("count", descending=True).head(5))
# residual by class (OOF rows)
o = d.filter(pl.col("pred").is_not_null())
print(o.group_by("is_dflt").agg(n=pl.len(), rmse=(pl.col("res")**2).mean().sqrt(), mres=pl.col("res").mean(),
      rmse_px=((pl.col("target") - pl.col("px"))**2).mean().sqrt()))
# AOBT_3 vs block, non-default: is block == AOBT_3 + k?
print("non-default rows with |d_aobt|<=10: share by month"); print(nd.group_by("m").agg(n=pl.len(), eq=(pl.col("d_aobt").abs() <= 10).mean(), eq60=(pl.col("d_aobt").abs()<=60).mean()).sort("m"))
