"""NM time fields relative to each other by airport and month (2025 Jan/Jul vs 2026 Jan/Jul): EOBT_1, LOBT, IOBT, AOBT_3."""
import glob, numpy as np, polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(20)
fr = []
for p in sorted(glob.glob("data/raw/training_2025-0[1678]-*.parquet")) + ["data/raw/ranking.parquet"]:
    fr.append(pl.scan_parquet(p).filter((pl.col("PHASE_mvt") == "DEP") & pl.col("LOBT_flt").is_not_null()).select(
        "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "EOBT_1_flt", "LOBT_flt", "IOBT_flt", "AOBT_3_flt", "TAXITIME_SEC_mvt").collect())
d = pl.concat(fr).with_columns(ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"))
s = lambda a, b: (pl.col(a) - pl.col(b)).dt.total_seconds()
d = d.with_columns(ae=s("AOBT_3_flt", "EOBT_1_flt"), el=s("EOBT_1_flt", "LOBT_flt"), al=s("AOBT_3_flt", "LOBT_flt"), te=s("MVT_TIME_UTC_mvt", "EOBT_1_flt"),
                   es=s("EOBT_1_flt", "SCHED_TIME_UTC_mvt"), il=s("IOBT_flt", "LOBT_flt"), ls=s("LOBT_flt", "SCHED_TIME_UTC_mvt"))
g = d.group_by("ADEP_mvt", "ym").agg(pl.len(),
    pl.col("ae").median().alias("aobt-eobt50"), pl.col("ae").quantile(0.75).alias("aobt-eobt75"),
    pl.col("el").median().alias("eobt-lobt50"), (pl.col("el") == 0).mean().alias("eobt=lobt"),
    pl.col("al").median().alias("aobt-lobt50"), pl.col("te").median().alias("T-eobt50"),
    (pl.col("es") == 0).mean().alias("eobt=sched"), (pl.col("ls") == 0).mean().alias("lobt=sched"), (pl.col("il") == 0).mean().alias("iobt=lobt")).sort("ADEP_mvt", "ym")
print(g)
