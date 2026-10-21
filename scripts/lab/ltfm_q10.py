import polars as pl, numpy as np
from ltfm_common import *
df = load()
dep = df.filter(pl.col("PHASE_mvt") == "DEP").with_columns(pxr=(pl.col("px")/60).round()*60, per=pl.col("src") + "-" + pl.col("m").cast(pl.Utf8))
dep = dep.filter(pl.col("m").is_in([1, 7]))
print(dep.group_by("per").agg(n=pl.len(), days=pl.col("day").n_unique(), nonm=(~pl.col("nm")).mean(), sched_eq_lobt=(pl.col("sched_lobt").abs()<=60).mean(),
    to_sched_med=pl.col("to_sched").median(), to_lobt_med=pl.col("to_lobt").median(), to_lobt_p90=pl.col("to_lobt").quantile(.9),
    px_med=pl.col("px").median(), px1200=(pl.col("pxr")==1200).mean(), px1500=(pl.col("pxr")==1500).mean(), px1140=(pl.col("pxr")==1140).mean(),
    px1440=(pl.col("pxr")==1440).mean(), px_ge1740=(pl.col("pxr")>=1740).mean(), px_le960=(pl.col("pxr")<=960).mean(),
    thy=(pl.col("al")=="TK").mean(), pgt=(pl.col("al")=="PC").mean()).sort("per"))
print("runway mix"); print(dep.group_by("per", "RUNWAY_mvt").len().with_columns(sh=pl.col("len")/pl.col("len").sum().over("per")).pivot(on="per", index="RUNWAY_mvt", values="sh").sort("RUNWAY_mvt"))
print("px by runway (median, 2025 train vs rank)"); print(dep.group_by("RUNWAY_mvt", "per").agg(pl.col("px").median()).pivot(on="per", index="RUNWAY_mvt", values="px").sort("RUNWAY_mvt"))
print("stand area mix"); print(dep.group_by("per", "stand_area").len().with_columns(sh=pl.col("len")/pl.col("len").sum().over("per")).pivot(on="per", index="stand_area", values="sh").sort("stand_area"))
# daily modal px default per day, by period
dd = dep.group_by("per", "day").agg(pl.col("pxr").mode().first().alias("dflt")).group_by("per", "dflt").len().sort("per", "dflt")
print(dd.pivot(on="per", index="dflt", values="len"))
# stands unseen in 2025
st25 = set(df.filter(pl.col("src")=="train")["STAND_mvt"].drop_nulls().unique())
r = dep.filter(pl.col("src")=="rank")
print("rank deps on stands unseen in 2025:", r.filter(~pl.col("STAND_mvt").is_in(list(st25))).height, r.filter(~pl.col("STAND_mvt").is_in(list(st25)))["STAND_mvt"].value_counts().sort("count", descending=True).head(10))
