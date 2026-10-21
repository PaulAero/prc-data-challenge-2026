"""How much does the model lean on take-off - AOBT_3 at LFPG, and how does the July 2026 px shift move
our ranking predictions? Also per-day px peakiness across 2025 to find analogue days."""
import os, numpy as np, polars as pl
S = os.environ["LFPG_OUT"]
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250)
a = pl.read_parquet(f"{S}/lfpg_all.parquet").filter(pl.col("PHASE_mvt") == "DEP").with_columns(
    pl.col("MVT_ID_mvt").cast(pl.Int64), ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"), day=pl.col("MVT_TIME_UTC_mvt").dt.date(),
    stp=pl.col("STAND_mvt").str.slice(0, 1).fill_null("NA"), rw=pl.col("RUNWAY_mvt").fill_null("NA"),
    px=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64))
a = a.with_columns(pm=((pl.col("px") + 30) // 60).cast(pl.Int64))
def peak(pm):
    v = np.bincount(np.clip(pm[(pm >= 0) & (pm < 40)], 0, 39), minlength=40).astype(float)
    ex = sum(max(v[k] - (v[k - 1] + v[k + 1]) / 2, 0) for k in range(5, 31))
    return ex / max(len(pm), 1)
# per-day peakiness distribution, per runway-day to control for config
rows = []
for (ym, day, rw), g in a.drop_nulls("pm").group_by("ym", "day", "rw"):
    if g.height >= 150:
        rows.append((ym, day, rw, g.height, peak(g["pm"].to_numpy())))
pk = pl.DataFrame(rows, schema=["ym", "day", "rw", "n", "peak"], orient="row")
print(pk.group_by("ym").agg(pl.col("peak").quantile(0.1).alias("p10"), pl.col("peak").median().alias("p50"), pl.col("peak").quantile(0.9).alias("p90"), pl.col("peak").max().alias("pmax"), pl.len()).sort("ym"))
# model slope on px (OOF, within stand-letter x runway x fold)
o = pl.read_parquet(f"{S}/oof_ens.parquet").select("MVT_ID_mvt", "fold", "pred")
x = a.join(o, on="MVT_ID_mvt").filter(pl.col("px").is_between(300, 2400) & (pl.col("TAXITIME_SEC_mvt") <= 10800))
x = x.with_columns(pxc=pl.col("px") - pl.col("px").mean().over("stp", "rw", "fold"), pc=pl.col("pred") - pl.col("pred").mean().over("stp", "rw", "fold"),
                   yc=pl.col("TAXITIME_SEC_mvt") - pl.col("TAXITIME_SEC_mvt").mean().over("stp", "rw", "fold"))
for f in ["jan", "jul"]:
    z = x.filter(pl.col("fold") == f)
    b_pred = (z["pxc"] * z["pc"]).sum() / (z["pxc"] ** 2).sum()
    b_y = (z["pxc"] * z["yc"]).sum() / (z["pxc"] ** 2).sum()
    print(f, "within-group slope pred~px", round(b_pred, 3), "target~px", round(b_y, 3))
# group means of px: Jul 2025 vs Jul 2026, and our v18 ranking predictions
sub = pl.read_parquet("data/submissions/candidate_v18.parquet")
print(sub.columns[:5])
idc = [c for c in sub.columns if "ID" in c.upper()][0]; pc = [c for c in sub.columns if "TAXI" in c.upper()][0]
r = a.filter(pl.col("src") == "rank").join(sub.select(pl.col(idc).cast(pl.Int64).alias("MVT_ID_mvt"), pl.col(pc).cast(pl.Float64).alias("pred")), on="MVT_ID_mvt")
t = a.filter(pl.col("src") == "train").join(o, on="MVT_ID_mvt")
both = pl.concat([t.select("ym", "stp", "rw", "px", "pred", pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y")),
                  r.select("ym", "stp", "rw", "px", "pred", pl.lit(None, dtype=pl.Float64).alias("y"))])
print(both.group_by("ym").agg(pl.len(), pl.col("px").median().alias("px50"), pl.col("px").mean().alias("pxmean"), pl.col("pred").mean().alias("pred_mean"), pl.col("pred").median().alias("pred50"), pl.col("y").filter(pl.col("y") < 10800).mean().alias("y_mean_core"), pl.col("y").median().alias("y50")).sort("ym"))
g = both.filter(pl.col("ym").is_in(["2025-07", "2026-07"])).group_by("ym", "rw").agg(pl.len(), pl.col("px").median().alias("px50"), pl.col("pred").median().alias("pred50"), pl.col("y").median().alias("y50")).sort("rw", "ym")
print(g)
