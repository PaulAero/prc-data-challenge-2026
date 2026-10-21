"""Label-free quality tests of AOBT_3 at LFPG by month.
(1) stand bound: the next arrival blocking in at my stand after my take-off-side timeline; a real off-block
    must precede the in-block of the next arrival at the same stand. Violation share of AOBT_3 vs (2025) block.
(2) variance of px explained by (stand, runway) means and by the runway queue (departures on my runway in the
    20 min before take-off), within month."""
import os, numpy as np, polars as pl
S = os.environ["LFPG_OUT"]
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250)
a = pl.read_parquet(f"{S}/lfpg_all.parquet").with_columns(ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"))
dep = a.filter(pl.col("PHASE_mvt") == "DEP").drop_nulls("STAND_mvt").sort("MVT_TIME_UTC_mvt")
arr = a.filter((pl.col("PHASE_mvt") == "ARR")).drop_nulls(["STAND_mvt", "BLOCK_TIME_UTC_mvt"]).select("STAND_mvt", pl.col("BLOCK_TIME_UTC_mvt").alias("ib")).sort("ib")
# last arrival in-block at my stand before my take-off (that arrival is at the stand when I take off -> I left before it)
j = dep.join_asof(arr, left_on="MVT_TIME_UTC_mvt", right_on="ib", by="STAND_mvt", strategy="backward")
j = j.with_columns(gap_ib=(pl.col("MVT_TIME_UTC_mvt") - pl.col("ib")).dt.total_seconds(),
                   px=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds())
# bound active when the in-block happened within 2 h before take-off: my off-block <= ib  =>  taxi >= gap_ib
k = j.filter(pl.col("gap_ib") <= 3600)
print("rows with an arrival in-block at my stand within 60 min before take-off, by month; share px < gap (AOBT_3 after the in-block), share y < gap")
print(k.group_by("ym").agg(pl.len(), (pl.col("px") < pl.col("gap_ib") - 60).mean().alias("px_viol"),
      (pl.col("TAXITIME_SEC_mvt") < pl.col("gap_ib") - 60).mean().alias("y_viol"),
      pl.col("px").is_null().mean().alias("px_null")).sort("ym"))
# (2) px explained variance
dep = dep.with_columns(px=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64))
t = dep["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy()
rw = dep["RUNWAY_mvt"].fill_null("NA").to_numpy()
q = np.zeros(len(t))
for r in np.unique(rw):
    idx = np.where(rw == r)[0]; tr = t[idx]
    q[idx] = np.searchsorted(tr, tr, "left") - np.searchsorted(tr, tr - 1200, "left")
dep = dep.with_columns(queue=pl.Series(q))
for ym in ["2025-01", "2025-07", "2025-08", "2025-12", "2026-01", "2026-07"]:
    x = dep.filter((pl.col("ym") == ym) & pl.col("px").is_between(0, 3600))
    x = x.with_columns(gm=pl.col("px").mean().over("STAND_mvt", "RUNWAY_mvt"), gy=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).mean().over("STAND_mvt", "RUNWAY_mvt"))
    r2 = 1 - ((x["px"] - x["gm"]) ** 2).sum() / ((x["px"] - x["px"].mean()) ** 2).sum()
    resid = x["px"] - x["gm"]
    cq = np.corrcoef(resid.to_numpy(), x["queue"].to_numpy())[0, 1]
    line = f"{ym} n={x.height} R2(px|stand,rwy)={r2:.3f} corr(px resid, queue)={cq:.3f}"
    if x["TAXITIME_SEC_mvt"].null_count() < x.height:
        yy = x.filter(pl.col("TAXITIME_SEC_mvt").is_between(0, 3600))
        ry = 1 - ((yy["TAXITIME_SEC_mvt"] - yy["gy"]) ** 2).sum() / ((yy["TAXITIME_SEC_mvt"] - yy["TAXITIME_SEC_mvt"].mean()) ** 2).sum()
        cqy = np.corrcoef((yy["TAXITIME_SEC_mvt"] - yy["gy"]).to_numpy(), yy["queue"].to_numpy())[0, 1]
        cpy = np.corrcoef(yy["px"].to_numpy(), yy["TAXITIME_SEC_mvt"].to_numpy())[0, 1]
        line += f" | target: R2={ry:.3f} corr(resid,queue)={cqy:.3f} corr(px,y)={cpy:.3f}"
    print(line)
