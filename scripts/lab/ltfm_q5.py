import polars as pl
from ltfm_common import *
df = load()
d = df.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("src") == "train"))
d = d.with_columns(d_lobt=s("BLOCK_TIME_UTC_mvt", "LOBT_flt"), d_iobt=s("BLOCK_TIME_UTC_mvt", "IOBT_flt"), d_eobt=s("BLOCK_TIME_UTC_mvt", "EOBT_1_flt"))
for c in ["d_sched", "d_lobt", "d_iobt"]:
    h = d.with_columns(b=(pl.col(c) / 60).round()).filter(pl.col("b").abs() <= 8)["b"].value_counts().sort("b")
    print(c, h["b"].to_list(), h["count"].to_list())
print("eobt==lobt", (d["d_eobt"] == d["d_lobt"]).mean(), "lobt==iobt", (d["d_lobt"] == d["d_iobt"]).mean())
print("sched-lobt (min) top:", d.with_columns(b=(pl.col("sched_lobt")/60).round())["b"].value_counts().sort("count", descending=True).head(8))
# fill = block minute equals sched minute AND take-off later; given block == sched, what is the take-off - sched dist vs others
d = d.with_columns(bm=(pl.col("d_sched")/60).round(), lm=(pl.col("d_lobt")/60).round())
print("rows block=sched, by sched==lobt"); print(d.filter(pl.col("bm")==0).group_by(pl.col("sched_lobt").abs()<=60).agg(n=pl.len(), ymed=pl.col("y").median()))
print("rows block=lobt but block!=sched"); print(d.filter((pl.col("lm")==0)&(pl.col("bm")!=0)).select(pl.len(), pl.col("y").median(), (pl.col("sched_lobt")/60).median()))
# is block==sched excess concentrated where take-off much later than sched? (classic fill)
print(d.with_columns(tb=pl.col("to_sched").cut([0,300,600,900,1200,1800,2700,3600,7200])).group_by("tb").agg(n=pl.len(), eq0=(pl.col("bm")==0).mean(), eqm1=(pl.col("bm")==-1).mean(), eqp1=(pl.col("bm")==1).mean(), eqp2=(pl.col("bm")==2).mean()).sort("tb"))
