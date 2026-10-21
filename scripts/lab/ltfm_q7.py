import polars as pl, numpy as np
from ltfm_common import *
df = load()
d = df.filter(pl.col("pred").is_not_null()).with_columns(
    jt=pl.when(pl.col("MVT_TIME_UTC_mvt").dt.second() >= 30).then(pl.col("MVT_TIME_UTC_mvt").dt.second() - 60).otherwise(pl.col("MVT_TIME_UTC_mvt").dt.second()).cast(pl.Float64),
    jb=pl.when(pl.col("BLOCK_TIME_UTC_mvt").dt.second() >= 30).then(pl.col("BLOCK_TIME_UTC_mvt").dt.second() - 60).otherwise(pl.col("BLOCK_TIME_UTC_mvt").dt.second()).cast(pl.Float64))
print("corr jt,jb", np.corrcoef(d["jt"], d["jb"])[0,1], "var jt", d["jt"].var(), "var jb", d["jb"].var())
x = d["jt"].to_numpy(); r = d["res"].to_numpy()
b = np.cov(x, r)[0,1] / x.var(); print("slope res~jt", b)
for tr, te in [("jan","jul"),("jul","jan")]:
    a = d.filter(pl.col("fold")==tr); t = d.filter(pl.col("fold")==te)
    bb = np.cov(a["jt"], a["res"])[0,1]/a["jt"].var()
    print(tr,"->",te, "slope", bb, "SSE/N gain", (t["res"]**2).mean() - ((t["res"]-bb*t["jt"])**2).mean())
# also does the model's pred mod 60 carry jt? check residual mod structure
# Oracle: true mean residual per (day,hour) and per day, leave-one-out
d = d.with_columns(h=pl.col("MVT_TIME_UTC_mvt").dt.truncate("1h"))
for keys in [["day"], ["h"]]:
    g = d.with_columns(sr=pl.col("res").sum().over(keys), n=pl.len().over(keys)).with_columns(loo=(pl.col("sr")-pl.col("res"))/(pl.col("n")-1+5))
    print("LOO oracle", keys, "SSE/N", (d["res"]**2).mean(), "->", ((g["res"]-g["loo"])**2).mean())
