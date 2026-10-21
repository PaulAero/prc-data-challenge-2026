import polars as pl, numpy as np
from ltfm_common import *
df = load()
sub = pl.read_parquet("data/submissions/quick-quicksand_v11.parquet")
print(sub.columns)
idc = [c for c in sub.columns if "ID" in c.upper()][0]; vc = [c for c in sub.columns if c != idc][0]
sub = sub.select(pl.col(idc).cast(pl.Int64).alias("MVT_ID_mvt"), pl.col(vc).cast(pl.Float64).alias("sub"))
df = df.join(sub, on="MVT_ID_mvt", how="left")
sg = lambda c: pl.when(pl.col(c).dt.second() >= 30).then(pl.col(c).dt.second().cast(pl.Int32) - 60).otherwise(pl.col(c).dt.second().cast(pl.Int32))
a = df.filter(pl.col("PHASE_mvt") == "ARR").with_columns(jl=sg("MVT_TIME_UTC_mvt"), ji=sg("BLOCK_TIME_UTC_mvt"), ti=s("BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt"))
print("arrivals jitter structure by src:"); print(a.group_by("src").agg(small=(pl.col("jl").abs()<=6).mean(), anti=((pl.col("jl")+pl.col("ji")).is_in([0,-1])).mean()))
dp = df.filter(pl.col("PHASE_mvt") == "DEP").with_columns(jt=sg("MVT_TIME_UTC_mvt"))
print("dep take-off jitter small by src:", dp.group_by("src").agg((pl.col("jt").abs()<=6).mean()))
dp = dp.with_columns(pxr=(pl.col("px")/60).round()*60)
ad = a.group_by("day").agg(ti_med=pl.col("ti").median(), n_arr=pl.len())
dd = dp.group_by("day").agg(src=pl.col("src").first(), n=pl.len(), dflt=pl.col("pxr").mode().first(), nonm=(~pl.col("nm")).mean(),
    to_lobt_med=pl.col("to_lobt").median(), y_med=pl.col("y").median(), sub_med=pl.col("sub").median(), y_mean=pl.col("y").mean(), sub_mean=pl.col("sub").mean()).join(ad, on="day")
pl.Config.set_tbl_rows(100)
print(dd.filter((pl.col("day").dt.month().is_in([1, 2])) ).sort("day").select("day","src","n","dflt","nonm","to_lobt_med","ti_med","y_med","y_mean","sub_med","sub_mean"))
print(dd.filter(pl.col("day").dt.month()==7).group_by("src").agg(pl.col("ti_med").mean(), pl.col("y_med").mean(), pl.col("sub_med").mean(), pl.col("to_lobt_med").mean()))
# 2025 relation: day y_med vs ti_med and dflt: simple regression on non-snow days for comparison
t = dd.filter(pl.col("src")=="train")
print(t.group_by("dflt").agg(n=pl.len(), y_med=pl.col("y_med").median(), ti=pl.col("ti_med").median(), tol=pl.col("to_lobt_med").median()))
r = dd.filter(pl.col("src")=="rank")
print(r.group_by("dflt", pl.col("day").dt.month()).agg(n=pl.len(), sub_med=pl.col("sub_med").median(), ti=pl.col("ti_med").median(), tol=pl.col("to_lobt_med").median()))
