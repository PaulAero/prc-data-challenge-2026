"""Minute-of-hour structure of AOBT_3, take-off and LOBT at LFPG by month (source of the July 2026 comb)."""
import os, numpy as np, polars as pl
S = os.environ["LFPG_OUT"]
a = pl.read_parquet(f"{S}/lfpg_all.parquet").filter(pl.col("PHASE_mvt") == "DEP").with_columns(ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"))
def modhist(x, c):
    m = x.select(pl.col(c).dt.minute().alias("m")).drop_nulls()["m"].to_numpy()
    h = np.bincount(m, minlength=60) / len(m) * 60
    return h
for ym in ["2025-07", "2026-01", "2026-07"]:
    x = a.filter(pl.col("ym") == ym)
    for c in ["AOBT_3_flt", "MVT_TIME_UTC_mvt", "LOBT_flt"]:
        h = modhist(x, c)
        print(ym, c[:8], "max/min ratio", round(h.max() / h.min(), 2), "chi-ish", round(((h - 1) ** 2).sum(), 3), " ".join(f"{v:.2f}" for v in h[:20]))
    # AOBT_3 seconds
    print(ym, "AOBT3 second != 0 share", (x["AOBT_3_flt"].dt.second() != 0).mean())
    # take-off minute mod 5 vs AOBT mod 5
    m5 = x.select(((pl.col("AOBT_3_flt").dt.minute()) % 5).alias("a")).drop_nulls()["a"].value_counts().sort("a")
    print(ym, "AOBT3 minute mod 5", m5["count"].to_list())
