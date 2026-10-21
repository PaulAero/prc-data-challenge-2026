"""July 2026 comb in take-off - AOBT_3 at LFPG: by runway, stand area, month; a 'peakiness' index."""
import os, numpy as np, polars as pl
S = os.environ["LFPG_OUT"]
pl.Config.set_tbl_rows(70); pl.Config.set_tbl_width_chars(250)
a = pl.read_parquet(f"{S}/lfpg_all.parquet").filter(pl.col("PHASE_mvt") == "DEP").with_columns(
    ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"), day=pl.col("MVT_TIME_UTC_mvt").dt.date(),
    px=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds(),
    stp=pl.col("STAND_mvt").str.slice(0, 1))
a = a.with_columns(pm=((pl.col("px") + 30) // 60))
def peak(x):
    """excess mass of each px minute over the mean of its two neighbours, summed over positive excess (5..30 min)"""
    h = x.filter(pl.col("pm").is_between(4, 31))["pm"].value_counts().sort("pm")
    v = dict(zip(h["pm"].to_list(), h["count"].to_list()))
    ex = 0
    for k in range(5, 31):
        e = v.get(k, 0) - (v.get(k - 1, 0) + v.get(k + 1, 0)) / 2
        ex += max(e, 0)
    return ex / max(x.height, 1)
print("peakiness index by month (0.0x = smooth)")
for ym in sorted(a["ym"].unique().to_list()):
    x = a.filter(pl.col("ym") == ym)
    print(ym, round(peak(x), 4), " by runway:", {r: round(peak(x.filter(pl.col("RUNWAY_mvt") == r)), 3) for r in ["26R", "27L", "08L", "09R", "27R"] if x.filter(pl.col("RUNWAY_mvt") == r).height > 500})
for ym in ["2025-07", "2026-07"]:
    for r in ["26R", "27L", "08L", "09R"]:
        x = a.filter((pl.col("ym") == ym) & (pl.col("RUNWAY_mvt") == r))
        h = x.filter(pl.col("pm").is_between(8, 24))["pm"].value_counts().sort("pm")
        print(ym, r, x.height, {k: round(c / x.height * 1000) for k, c in zip(h["pm"].to_list(), h["count"].to_list())})
# daily peakiness in 2026-07 and 2026-01
for ym in ["2025-07", "2026-01", "2026-07"]:
    x = a.filter(pl.col("ym") == ym)
    print(ym, "daily", [round(peak(x.filter(pl.col("day") == dd)), 2) for dd in sorted(x["day"].unique().to_list())])
