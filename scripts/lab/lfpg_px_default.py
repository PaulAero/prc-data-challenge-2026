"""Is LFPG take-off - AOBT_3 a deterministic default per (stand, runway) in July 2026?"""
import os, numpy as np, polars as pl
S = os.environ["LFPG_OUT"]
pl.Config.set_tbl_rows(70); pl.Config.set_tbl_width_chars(250)
a = pl.read_parquet(f"{S}/lfpg_all.parquet").filter(pl.col("PHASE_mvt") == "DEP").with_columns(
    ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"), day=pl.col("MVT_TIME_UTC_mvt").dt.date(),
    px=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds(),
    stp=pl.col("STAND_mvt").str.slice(0, 1))
a = a.with_columns(pm=((pl.col("px") + 30) // 60)).drop_nulls("pm")
for key in [["STAND_mvt", "RUNWAY_mvt"], ["stp", "RUNWAY_mvt"]]:
    print("key", key)
    for ym in ["2025-01", "2025-07", "2025-08", "2026-01", "2026-07"]:
        x = a.filter(pl.col("ym") == ym)
        g = x.group_by(key + ["pm"]).len()
        mode = g.sort("len", descending=True).group_by(key, maintain_order=True).first().rename({"pm": "mode", "len": "nmode"})
        tot = g.group_by(key).agg(pl.col("len").sum().alias("n"))
        x2 = x.join(mode.select(key + ["mode"]), on=key).join(tot, on=key).filter(pl.col("n") >= 5)
        print(" ", ym, "rows", x2.height, "share at group mode", round((x2["pm"] == x2["mode"]).mean(), 3), "within 1 min", round(((x2["pm"] - x2["mode"]).abs() <= 1).mean(), 3),
              "px sd within group (median over groups)", round(x2.group_by(key).agg(pl.col("px").std())["px"].median(), 1))
# July 2026 vs 2025: target-free check of AOBT_3 vs EOBT / LOBT relation
for ym in ["2025-07", "2026-07"]:
    x = a.filter(pl.col("ym") == ym).with_columns(ae=(pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt")).dt.total_seconds() // 60)
    print(ym, "AOBT3-EOBT1 minute: share 0", (x["ae"] == 0).mean(), "q", x["ae"].quantile(0.25), x["ae"].median(), x["ae"].quantile(0.75))
