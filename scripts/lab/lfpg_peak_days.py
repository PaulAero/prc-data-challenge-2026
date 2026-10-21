"""2025 runway-days with a peaky take-off - AOBT_3 histogram (like July 2026): is px less informative there,
and is the full-year regression worse?"""
import os, numpy as np, polars as pl
S = os.environ["LFPG_OUT"]
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(250)
a = pl.read_parquet(f"{S}/lfpg_all.parquet").filter((pl.col("PHASE_mvt") == "DEP")).with_columns(
    pl.col("MVT_ID_mvt").cast(pl.Int64), ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"), day=pl.col("MVT_TIME_UTC_mvt").dt.date(), rw=pl.col("RUNWAY_mvt").fill_null("NA"),
    px=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64), y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
a = a.with_columns(pm=((pl.col("px") + 30) // 60).cast(pl.Int64))
def peak(pm):
    pm = pm[(pm >= 0) & (pm < 40)]
    v = np.bincount(pm, minlength=40).astype(float)
    return sum(max(v[k] - (v[k - 1] + v[k + 1]) / 2, 0) for k in range(5, 31)) / max(len(pm), 1)
rows = [(day, rw, g.height, peak(g["pm"].to_numpy())) for (day, rw), g in a.drop_nulls("pm").group_by("day", "rw") if g.height >= 150]
pk = pl.DataFrame(rows, schema=["day", "rw", "n", "peak"], orient="row")
p = pl.read_parquet("experiments/lab-year/parts.parquet").select("MVT_ID_mvt", "reg")
x = a.join(pk.select("day", "rw", "peak"), on=["day", "rw"]).join(p, on="MVT_ID_mvt").filter(pl.col("px").is_between(0, 3600) & (pl.col("y") < 7200))
x = x.with_columns(pq=pl.when(pl.col("peak") < 0.15).then(pl.lit("a <0.15")).when(pl.col("peak") < 0.22).then(pl.lit("b 0.15-0.22")).when(pl.col("peak") < 0.28).then(pl.lit("c 0.22-0.28")).otherwise(pl.lit("d >=0.28")))
print(x.group_by("pq").agg(pl.len(), pl.col("day").n_unique().alias("days"), pl.corr("px", "y").alias("corr_px_y"), ((pl.col("y") - pl.col("px")).abs().median()).alias("mad_y_px"),
      ((pl.col("y") - pl.col("reg")) ** 2).mean().sqrt().alias("rmse_reg"), (pl.col("y") - pl.col("reg")).mean().alias("bias_reg"), pl.col("y").std().alias("sd_y")).sort("pq"))
r = a.filter(pl.col("ym") == "2026-07").join(pk.select("day", "rw", "peak"), on=["day", "rw"])
print("July 2026 rows by peak class:", r.with_columns(pq=pl.when(pl.col("peak") < 0.15).then(pl.lit("a")).when(pl.col("peak") < 0.22).then(pl.lit("b")).when(pl.col("peak") < 0.28).then(pl.lit("c")).otherwise(pl.lit("d")))["pq"].value_counts().sort("pq").rows())
