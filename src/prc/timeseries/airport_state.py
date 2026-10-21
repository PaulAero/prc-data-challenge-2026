"""Build regular airport-level time series from movement records.

The challenge target is per departure, but TimesFM-style experiments need a
regular sequence.  This module aggregates movements without mixing columns
that are available in the ranking set (traffic counts and schedules) with the
taxi-out target, which is absent there.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import polars as pl


REQUIRED_COLUMNS = {
    "ADEP_mvt",
    "ADES_mvt",
    "PHASE_mvt",
    "MVT_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "TAXITIME_SEC_mvt",
}


def _contiguous_date_ranges(dates: Iterable[date]) -> list[tuple[date, date]]:
    """Split observed calendar dates at gaps longer than one day."""

    ordered = sorted(set(dates))
    if not ordered:
        return []

    ranges: list[tuple[date, date]] = []
    start = previous = ordered[0]
    for current in ordered[1:]:
        if current - previous > timedelta(days=1):
            ranges.append((start, previous))
            start = current
        previous = current
    ranges.append((start, previous))
    return ranges


def _scan_selected(paths: Sequence[Path]) -> pl.LazyFrame:
    if not paths:
        raise ValueError("At least one parquet input is required")

    frames: list[pl.LazyFrame] = []
    for path in paths:
        schema = pl.read_parquet_schema(path)
        missing = REQUIRED_COLUMNS.difference(schema)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frames.append(pl.scan_parquet(path).select(sorted(REQUIRED_COLUMNS)))
    return pl.concat(frames, how="vertical")


def build_airport_state(
    paths: Sequence[str | Path], *, interval: str = "15m"
) -> pl.DataFrame:
    """Aggregate movements into dense, regular airport-state sequences.

    Separate contiguous calendar ranges are assigned separate ``series_id``
    values.  This prevents the January/July ranking file from being silently
    filled with an artificial five-month run of zero traffic.
    """

    parquet_paths = [Path(path) for path in paths]
    movements = _scan_selected(parquet_paths).with_columns(
        pl.when(pl.col("PHASE_mvt") == "DEP")
        .then(pl.col("ADEP_mvt"))
        .otherwise(pl.col("ADES_mvt"))
        .alias("airport"),
        pl.col("MVT_TIME_UTC_mvt").dt.truncate(interval).alias("timestamp"),
        pl.col("SCHED_TIME_UTC_mvt").dt.truncate(interval).alias("scheduled_timestamp"),
    )

    observed = (
        movements.group_by("airport", "timestamp")
        .agg(
            (pl.col("PHASE_mvt") == "DEP").sum().cast(pl.UInt32).alias("departure_count"),
            (pl.col("PHASE_mvt") == "ARR").sum().cast(pl.UInt32).alias("arrival_count"),
            pl.col("TAXITIME_SEC_mvt")
            .filter(pl.col("PHASE_mvt") == "DEP")
            .mean()
            .alias("taxi_out_mean_sec"),
            pl.col("TAXITIME_SEC_mvt")
            .filter(pl.col("PHASE_mvt") == "DEP")
            .median()
            .alias("taxi_out_median_sec"),
            pl.col("TAXITIME_SEC_mvt")
            .filter(pl.col("PHASE_mvt") == "DEP")
            .quantile(0.9, interpolation="nearest")
            .alias("taxi_out_p90_sec"),
        )
        .collect(engine="streaming")
    )

    scheduled = (
        movements.filter(pl.col("PHASE_mvt") == "DEP")
        .group_by("airport", pl.col("scheduled_timestamp").alias("timestamp"))
        .agg(pl.len().cast(pl.UInt32).alias("scheduled_departure_count"))
        .collect(engine="streaming")
    )

    airports = observed.get_column("airport").drop_nulls().unique().sort()
    dates = observed.get_column("timestamp").dt.date().unique().to_list()
    ranges = _contiguous_date_ranges(dates)
    if not ranges:
        return observed.head(0)

    grids: list[pl.DataFrame] = []
    for period_id, (first_day, last_day) in enumerate(ranges):
        start = datetime.combine(first_day, time.min, tzinfo=timezone.utc)
        end = datetime.combine(last_day + timedelta(days=1), time.min, tzinfo=timezone.utc)
        timestamps = pl.datetime_range(
            start,
            end,
            interval=interval,
            closed="left",
            time_zone="UTC",
            eager=True,
        ).alias("timestamp")
        grids.append(
            pl.DataFrame({"airport": airports})
            .join(pl.DataFrame({"timestamp": timestamps}), how="cross")
            .with_columns(pl.lit(period_id, dtype=pl.UInt8).alias("period_id"))
        )

    return (
        pl.concat(grids)
        .join(observed, on=["airport", "timestamp"], how="left")
        .join(scheduled, on=["airport", "timestamp"], how="left")
        .with_columns(
            pl.col("departure_count").fill_null(0),
            pl.col("arrival_count").fill_null(0),
            pl.col("scheduled_departure_count").fill_null(0),
            (
                pl.col("airport")
                + pl.lit("__period_")
                + pl.col("period_id").cast(pl.String)
            ).alias("series_id"),
        )
        .select(
            "series_id",
            "airport",
            "period_id",
            "timestamp",
            "departure_count",
            "arrival_count",
            "scheduled_departure_count",
            "taxi_out_mean_sec",
            "taxi_out_median_sec",
            "taxi_out_p90_sec",
        )
        .sort("series_id", "timestamp")
    )
