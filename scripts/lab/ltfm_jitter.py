import polars as pl
pl.Config.set_tbl_rows(40)
sg = lambda c: pl.when(pl.col(c).dt.second() >= 30).then(pl.col(c).dt.second().cast(pl.Int32) - 60).otherwise(pl.col(c).dt.second().cast(pl.Int32))
d = pl.scan_parquet("data/raw/training_2025-07-01_2025-08-01.parquet").filter(pl.col("PHASE_mvt") == "DEP").select(
    "ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt").collect()
d = d.with_columns(jt=sg("MVT_TIME_UTC_mvt"), jb=sg("BLOCK_TIME_UTC_mvt"))
d = d.with_columns(r=(pl.col("TAXITIME_SEC_mvt") - 2 * pl.col("jt")) % 60, jsum=pl.col("jt") + pl.col("jb"))
print(d.group_by("ADEP_mvt").agg(n=pl.len(), jt_small=(pl.col("jt").abs() <= 6).mean(), jb_small=(pl.col("jb").abs() <= 6).mean(),
    jsum0=(pl.col("jsum") == 0).mean(), jsum_abs1=(pl.col("jsum").abs() <= 1).mean(), corr=pl.corr("jt", "jb")).sort("ADEP_mvt"))
x = d.filter(pl.col("ADEP_mvt") == "LTFM")
print(x["jsum"].value_counts().sort("jsum"))
print("y - 2jt mod 60:", x["r"].value_counts().sort("count", descending=True).head(5))
