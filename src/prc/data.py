"""Loading official movement files, and the masking that defines what a feature may see."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

RAW = Path("data/raw")
MASKED_DEPARTURE_COLUMNS = ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt")


def training_paths(raw: Path = RAW) -> list[str]:
    return sorted(glob.glob(str(raw / "training_*.parquet")))


def load_movements(paths: list[str] | str) -> pl.DataFrame:
    """All movements (ARR and DEP) with a stable schema (MVT_ID as Int64)."""
    frame = pl.read_parquet(paths) if isinstance(paths, str) else pl.concat(
        [pl.read_parquet(p) for p in paths], how="vertical_relaxed"
    )
    return frame.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("FLIGHT_ID_mvt").cast(pl.Int64))


def load_training() -> pl.DataFrame:
    return load_movements(training_paths())


def load_ranking() -> pl.DataFrame:
    return load_movements(str(RAW / "ranking.parquet"))


def mask_departures(movements: pl.DataFrame) -> pl.DataFrame:
    """Blank departures' off-block and taxi time, exactly as in the ranking file.

    Every feature builder receives only the output of this function, so no feature can read a
    departure's off-block time, on train or on ranking.
    """
    is_dep = pl.col("PHASE_mvt") == "DEP"
    return movements.with_columns(
        [pl.when(is_dep).then(None).otherwise(pl.col(c)).alias(c) for c in MASKED_DEPARTURE_COLUMNS]
    )


def departure_labels(movements: pl.DataFrame) -> pl.DataFrame:
    """Target and label-derived columns for training departures (never used as features)."""
    return (
        movements.filter(pl.col("PHASE_mvt") == "DEP")
        .select(
            "MVT_ID_mvt",
            pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("target"),
            # Off-block equal to the schedule within 10 s: the schedule-fill recording regime.
            ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 10)
            .cast(pl.Int8)
            .alias("fill"),
        )
    )
