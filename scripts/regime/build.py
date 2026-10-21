"""Recording-regime features from arrivals (EXP-054), label-free.

EXP-050 found that an airline's LIRF arrivals switch to substitute in-block times (in-block = schedule,
or landing + 10 min) together with its departures switching to schedule-filled off-blocks. Across
airline-months at LIRF in 2025 the two rates correlate at 0.79 (0.41 for month-to-month changes within
an airline); elsewhere about 0. Arrivals' in-block and schedule are inputs in the ranking file, so the
same features exist for 2026. Per departure: the substitute rate of its airline's arrivals at its
airport in the same calendar month (and the count), and the airport-month rate.

Usage: uv run python scripts/regime/build.py -> data/processed/regime/{jan,jul,q4,final}.parquet
"""

from __future__ import annotations

import glob

import polars as pl

from prc.data import load_movements


def main() -> None:
    mv = pl.concat([load_movements(p) for p in sorted(glob.glob("data/raw/training_2025-*.parquet"))
                    + ["data/raw/ranking.parquet"]], how="diagonal_relaxed").with_columns(
        pl.col("FLIGHT_mvt").str.extract(r"^([A-Z0-9]{2,3}?)\d", 1).alias("al"),
        pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("ym"))
    arr = mv.filter(pl.col("PHASE_mvt") == "ARR").with_columns(
        (((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 10)
         | ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds().is_between(595, 605))
         ).cast(pl.Float64).alias("sub"), pl.col("ADES_mvt").alias("ap"))
    al = arr.group_by("ap", "al", "ym").agg(pl.len().alias("reg_al_n"), pl.col("sub").mean().alias("reg_al_sub"))
    apm = arr.group_by("ap", "ym").agg(pl.col("sub").mean().alias("reg_ap_sub"))
    dep = mv.filter(pl.col("PHASE_mvt") == "DEP").select(pl.col("MVT_ID_mvt").cast(pl.Int64),
                                                        pl.col("ADEP_mvt").alias("ap"), "al", "ym")
    out = dep.join(al, on=["ap", "al", "ym"], how="left").join(apm, on=["ap", "ym"], how="left").with_columns(
        pl.when(pl.col("reg_al_n") >= 10).then(pl.col("reg_al_sub")).alias("reg_al_sub"),
        pl.col("reg_al_n").fill_null(0).cast(pl.Float64),
        (pl.col("ap") == "LIRF").cast(pl.Float64).alias("_lirf"))
    out = out.with_columns((pl.col("reg_al_sub") * pl.col("_lirf")).alias("reg_al_sub_lirf")).select(
        "MVT_ID_mvt", "reg_al_sub", "reg_al_n", "reg_ap_sub", "reg_al_sub_lirf")
    import os
    os.makedirs("data/processed/regime", exist_ok=True)
    for f in ("jan", "jul", "q4", "final"):
        out.write_parquet(f"data/processed/regime/{f}.parquet")
    # EXP-057: per flight, the latest same-airline arrival at the departure's stand within 8 h before its
    # schedule: was its in-block a substitute (LIRF: departure fill 30% vs 14%)?
    a2 = mv.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("STAND_mvt").is_not_null() & pl.col("BLOCK_TIME_UTC_mvt").is_not_null()).select(
        pl.col("ADES_mvt").alias("ap"), "STAND_mvt", pl.col("BLOCK_TIME_UTC_mvt").alias("t_in"), pl.col("al").alias("al_in"),
        ((((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 10)
          | ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds().is_between(595, 605)))).cast(pl.Float64).alias("in_sub")).sort("t_in")
    d2 = mv.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("STAND_mvt").is_not_null()).select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("ADEP_mvt").alias("ap"), "STAND_mvt", "SCHED_TIME_UTC_mvt", "al").sort("SCHED_TIME_UTC_mvt")
    j = d2.join_asof(a2, left_on="SCHED_TIME_UTC_mvt", right_on="t_in", by=["ap", "STAND_mvt"], strategy="backward", tolerance="8h")
    j = j.with_columns(pl.when(pl.col("al") == pl.col("al_in")).then(pl.col("in_sub")).alias("reg_inbound_sub"),
                       (pl.col("al") == pl.col("al_in")).cast(pl.Float64).alias("reg_inbound_same_al"),
                       ((pl.col("SCHED_TIME_UTC_mvt") - pl.col("t_in")).dt.total_seconds() / 60).alias("reg_inbound_gap_min"))
    j = j.with_columns((pl.col("reg_inbound_sub") * (pl.col("ap") == "LIRF").cast(pl.Float64)).alias("reg_inbound_sub_lirf"))
    out2 = out.join(j.select("MVT_ID_mvt", "reg_inbound_sub", "reg_inbound_same_al", "reg_inbound_gap_min", "reg_inbound_sub_lirf"),
                    on="MVT_ID_mvt", how="left")
    import os
    os.makedirs("data/processed/regime2", exist_ok=True)
    for f in ("jan", "jul", "q4", "final"):
        out2.write_parquet(f"data/processed/regime2/{f}.parquet")
    print(out2.select(pl.col("reg_inbound_sub").mean().alias("rate"), pl.col("reg_inbound_sub").is_not_null().mean().alias("coverage")))
    print(out.describe())


if __name__ == "__main__":
    main()
