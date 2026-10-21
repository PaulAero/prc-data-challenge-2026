"""Label-free check of AOBT_3 quality per month: agreement of take-off - AOBT_3 with ADS-B taxi
estimates (time since leaving the stand), LFPG and all airports. In 2025 also against the target."""
import os, numpy as np, polars as pl
S = os.environ["LFPG_OUT"]
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250)
f = pl.read_parquet("data/processed/adsb_features.parquet").select("MVT_ID_mvt", "adsb_leave_stand", "adsb_d0_stand", "adsb_seen", "adsb_move0", "adsb_still0")
import glob
fr = []
for p in sorted(glob.glob("data/raw/training_2025-0[17]-*.parquet")) + ["data/raw/ranking.parquet"]:
    fr.append(pl.scan_parquet(p).filter(pl.col("PHASE_mvt") == "DEP").select(pl.col("MVT_ID_mvt").cast(pl.Int64), "ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "TAXITIME_SEC_mvt", "RUNWAY_mvt").collect())
d = pl.concat(fr).join(f, on="MVT_ID_mvt", how="inner").with_columns(
    ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"), px=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64))
x = d.filter(pl.col("adsb_leave_stand").is_not_null() & pl.col("px").is_not_null())
def stats(g):
    return g.agg(pl.len().alias("n"),
        ((pl.col("px") - pl.col("adsb_leave_stand")).median()).alias("med_px_minus_ls"),
        ((pl.col("px") - pl.col("adsb_leave_stand")).abs().median()).alias("mad_px_ls"),
        (((pl.col("px") - pl.col("adsb_leave_stand")).abs() <= 120).mean()).alias("px_ls_within2m"),
        pl.corr("px", "adsb_leave_stand").alias("corr_px_ls"),
        ((pl.col("TAXITIME_SEC_mvt") - pl.col("adsb_leave_stand")).abs().median()).alias("mad_y_ls"),
        ((pl.col("TAXITIME_SEC_mvt") - pl.col("px")).abs().median()).alias("mad_y_px"))
print(stats(x.group_by("ADEP_mvt", "ym")).sort("ADEP_mvt", "ym"))
# LFPG: also adsb_seen (first ground point) as a looser reference
y = d.filter((pl.col("ADEP_mvt") == "LFPG") & pl.col("adsb_seen").is_not_null() & pl.col("px").is_not_null())
print(y.group_by("ym").agg(pl.len(), ((pl.col("px") - pl.col("adsb_seen")).abs().median()).alias("mad_px_seen"), pl.corr("px", "adsb_seen").alias("corr"),
      ((pl.col("TAXITIME_SEC_mvt") - pl.col("adsb_seen")).abs().median()).alias("mad_y_seen")).sort("ym"))
