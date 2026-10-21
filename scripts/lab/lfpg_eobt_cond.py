"""2025 LFPG: y - px given AOBT_3 - EOBT_1, split by whether EOBT_1 was updated (EOBT_1 != LOBT) and by
take-off - LOBT (delay). Does the conditional relation depend on how EOBT came to be close to AOBT_3?"""
import os, numpy as np, polars as pl
S = os.environ["LFPG_OUT"]
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(20)
a = pl.read_parquet(f"{S}/lfpg_all.parquet").filter((pl.col("PHASE_mvt") == "DEP") & pl.col("LOBT_flt").is_not_null() & (pl.col("src") == "train")).with_columns(
    px=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64),
    ae=((pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt")).dt.total_seconds() / 60),
    al=((pl.col("AOBT_3_flt") - pl.col("LOBT_flt")).dt.total_seconds() / 60),
    eupd=(pl.col("EOBT_1_flt") != pl.col("LOBT_flt")), y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
a = a.filter(pl.col("px").is_between(0, 3600) & (pl.col("y") < 7200))
a = a.with_columns(aeb=pl.col("ae").clip(-6, 12).floor().cast(pl.Int64), alb=(pl.col("al").clip(-10, 40) // 10 * 10).cast(pl.Int64))
t = a.group_by("aeb", "eupd").agg(pl.len().alias("n"), (pl.col("y") - pl.col("px")).mean().alias("y_px")).pivot(on="eupd", index="aeb", values=["n", "y_px"]).sort("aeb")
print(t)
t = a.filter(pl.col("ae").abs() <= 3).group_by("alb").agg(pl.len().alias("n"), (pl.col("y") - pl.col("px")).mean().alias("y_px_ae_small")).sort("alb")
print("rows with |AOBT3-EOBT1|<=3 min, by AOBT3-LOBT bin (min):"); print(t)
