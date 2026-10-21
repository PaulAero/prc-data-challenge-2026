"""EXP-055: compare our predictions with EUROCONTROL's published monthly taxi-out averages.

ATXOT population (methodology pdf): departures with stand, runway and off-block, taxi-out <= 120 min,
local-time months. Mean = (TOTAL_REF_TIME_MIN + TOTAL_ADD_TIME_MIN) / TOTAL_REF_NB_FL (minutes).
Implied 2026 true mean on that population = published 2026 + (true 2025 - published 2025), the gap
taken from the same airport and calendar month (or a neighbouring month for validation).
"""
from __future__ import annotations
import glob, sys
import numpy as np
import polars as pl
from prc.data import load_movements

TZ = {"EGLL": "Europe/London", "LTFM": "Europe/Istanbul"}

def pru(year: int) -> pl.DataFrame:
    d = pl.read_csv(f"data/external/pru/taxi_out_additional_time_{year}.csv", infer_schema_length=0)
    return d.select(pl.col("APT_ICAO").alias("ap"), pl.col("MONTH_NUM").cast(pl.Int32).alias("m"),
                    ((pl.col("TOTAL_REF_TIME_MIN").cast(pl.Float64) + pl.col("TOTAL_ADD_TIME_MIN").cast(pl.Float64))
                     / pl.col("TOTAL_REF_NB_FL").cast(pl.Float64) * 60).alias(f"pru{year}"))

def local_month(df: pl.DataFrame) -> pl.DataFrame:
    parts = []
    for ap, g in df.group_by("ap"):
        tz = TZ.get(ap[0], "Europe/Paris")
        parts.append(g.with_columns(pl.col("t").dt.convert_time_zone(tz).dt.month().cast(pl.Int32).alias("m")))
    return pl.concat(parts)

def main(sub: str) -> None:
    tr = pl.concat([load_movements(p) for p in sorted(glob.glob("data/raw/training_2025-*.parquet"))], how="diagonal_relaxed")
    tr = tr.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("STAND_mvt").is_not_null() & pl.col("RUNWAY_mvt").is_not_null()
                   & (pl.col("TAXITIME_SEC_mvt") <= 7200)).select(pl.col("ADEP_mvt").alias("ap"), pl.col("MVT_TIME_UTC_mvt").alias("t"),
                   pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"))
    true25 = local_month(tr).group_by("ap", "m").agg(pl.col("y").mean().alias("true2025"), pl.len().alias("n25"))
    g = true25.join(pru(2025), on=["ap", "m"]).with_columns((pl.col("true2025") - pl.col("pru2025")).alias("gap")).sort("ap", "m")
    # validation within 2025: predict month m with the gap of month m-1 (and m+1 for January)
    v = g.with_columns(pl.col("gap").shift(1).over("ap").alias("gap_prev"), pl.col("gap").shift(-1).over("ap").alias("gap_next"))
    v = v.with_columns(pl.coalesce("gap_prev", "gap_next").alias("gap_nb")).with_columns((pl.col("gap_nb") - pl.col("gap")).alias("err"))
    print("gap transfer error (s), month from neighbour month, 2025:")
    print(v.group_by("ap").agg(pl.col("err").pow(2).mean().sqrt().round(1).alias("rms"), pl.col("gap").mean().round(1).alias("mean_gap")).sort("ap"))
    rk = load_movements("data/raw/ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP").select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("ADEP_mvt").alias("ap"), pl.col("MVT_TIME_UTC_mvt").alias("t"),
        (pl.col("STAND_mvt").is_not_null() & pl.col("RUNWAY_mvt").is_not_null()).alias("has_sr"))
    p = pl.read_parquet(sub).select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("pred"))
    x = local_month(rk.join(p, on="MVT_ID_mvt")).filter(pl.col("has_sr") & (pl.col("pred") <= 7200) & pl.col("m").is_in([1, 7]))
    pm = x.group_by("ap", "m").agg(pl.col("pred").mean().alias("pred_mean"), pl.len().alias("n26"))
    out = pm.join(pru(2026), on=["ap", "m"]).join(g.select("ap", "m", "gap"), on=["ap", "m"]).with_columns(
        (pl.col("pru2026") + pl.col("gap")).alias("implied")).with_columns((pl.col("pred_mean") - pl.col("implied")).alias("bias"))
    out = out.with_columns((pl.col("bias") ** 2 * pl.col("n26") / 344841).alias("board_mse_if_true"))
    pl.Config.set_tbl_rows(30)
    print(out.select("ap", "m", "n26", pl.col("pru2026").round(1), pl.col("gap").round(1), pl.col("implied").round(1),
                     pl.col("pred_mean").round(1), pl.col("bias").round(1), pl.col("board_mse_if_true").round(0)).sort("ap", "m"))

if __name__ == "__main__":
    main(sys.argv[1])
