import polars as pl, numpy as np
from ltfm_common import *
df = load()
sg = lambda c: pl.when(pl.col(c).dt.second() >= 30).then(pl.col(c).dt.second().cast(pl.Int32) - 60).otherwise(pl.col(c).dt.second().cast(pl.Int32))
o = df.filter(pl.col("pred").is_not_null()).with_columns(jt=sg("MVT_TIME_UTC_mvt").cast(pl.Float64))
for f in ["jan", "jul"]:
    T = o.filter(pl.col("fold") == f)
    sl = lambda y: np.cov(T["jt"], T[y])[0, 1] / T["jt"].var()
    print(f, "slope target~jt %.2f pred~jt %.2f res~jt %.2f  (SE approx %.2f)" % (sl("target"), sl("pred"), sl("res"), T["res"].std() / (T["jt"].std() * T.height ** 0.5)))
# K = (y - 2jt)/60 integer? check
o = o.with_columns(K=(pl.col("target") - 2 * pl.col("jt")) / 60)
print("frac K integer-ish", ((o["K"] - o["K"].round()).abs() < 0.02).mean())
# expected gain if residual slope is truly 2-b: var(jt)*(2-b)^2
b = np.cov(o["jt"], o["pred"])[0, 1] / o["jt"].var()
print("pooled pred slope", b, "theoretical gain/N", o["jt"].var() * (2 - b) ** 2)
