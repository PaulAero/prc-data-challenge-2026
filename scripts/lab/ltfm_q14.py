import polars as pl, numpy as np
from ltfm_common import *
df = load()
sub = pl.read_parquet("data/submissions/quick-quicksand_v11.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("sub"))
df = df.join(sub, on="MVT_ID_mvt", how="left")
dep = df.filter(pl.col("PHASE_mvt") == "DEP").with_columns(pxr=(pl.col("px") / 60).round() * 60)
arr = df.filter(pl.col("PHASE_mvt") == "ARR").with_columns(ti=s("BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt"), adel=s("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"))
ad = arr.group_by("day").agg(ti_d=pl.col("ti").median(), adel_d=pl.col("adel").median())
dd = dep.group_by("day").agg(src=pl.col("src").first(), dflt=pl.col("pxr").mode().first(), tol=pl.col("to_lobt").median(), tos=pl.col("to_sched").median(),
    n=pl.len(), y=pl.col("y").median(), sub=pl.col("sub").median(), pred=pl.col("pred").median(), res=pl.col("res").mean(), m=pl.col("m").first()).join(ad, on="day")
dd = dd.filter(pl.col("dflt").is_in([1200, 1500]))
t = dd.filter(pl.col("src") == "train")
X = lambda f: np.column_stack([np.ones(f.height), f["tol"], f["tos"], (f["dflt"] == 1500).cast(int), f["ti_d"], f["adel_d"], f["n"]])
beta, *_ = np.linalg.lstsq(X(t), t["y"].to_numpy(), rcond=None)
print("2025 day-level y_med ~ [1, tol, tos, d1500, ti_d, adel_d, n]:", np.round(beta, 3), "R2", 1 - ((t["y"] - X(t) @ beta) ** 2).sum() / ((t["y"] - t["y"].mean()) ** 2).sum())
# leave-month-out check of this day-level regression for jan/jul 2025, and model's day median there
for mm in [1, 7]:
    tr = t.filter(pl.col("m") != mm); te = t.filter(pl.col("m") == mm)
    b2, *_ = np.linalg.lstsq(X(tr), tr["y"].to_numpy(), rcond=None)
    print("2025 m=%d: day-reg pred median-of-days %.0f, true %.0f, model OOF %.0f; RMSE days day-reg %.0f model %.0f" % (
        mm, np.median(X(te) @ b2), te["y"].median(), te["pred"].median(), np.sqrt(((te["y"] - X(te) @ b2) ** 2).mean()), np.sqrt(((te["y"] - te["pred"]) ** 2).mean())))
r = dd.filter(pl.col("src") == "rank").with_columns(dayreg=pl.Series(X(dd.filter(pl.col("src") == "rank")) @ beta))
print(r.group_by("m", "dflt").agg(n=pl.len(), sub=pl.col("sub").median(), dayreg=pl.col("dayreg").median(), tol=pl.col("tol").median(), tos=pl.col("tos").median(), ti=pl.col("ti_d").median(), adel=pl.col("adel_d").median()).sort("m", "dflt"))
print(t.filter(pl.col("m").is_in([1, 7])).group_by("m", "dflt").agg(n=pl.len(), y=pl.col("y").median(), tol=pl.col("tol").median(), tos=pl.col("tos").median(), ti=pl.col("ti_d").median(), adel=pl.col("adel_d").median()).sort("m", "dflt"))
# OOF day residual vs tos (does the model over-react to day-level delay in 2025?)
o = t.filter(pl.col("res").is_not_null())
print("2025 jan+jul: corr(day mean res, tos)", np.corrcoef(o["res"], o["tos"])[0, 1], "corr(res, tol)", np.corrcoef(o["res"], o["tol"])[0, 1], "n days", o.height)
