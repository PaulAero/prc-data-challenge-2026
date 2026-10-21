"""Derived columns and precomputed aggregates for the data explorer cache.

The raw files are never modified. ``build_movements`` produces one flat table
(training 2025 plus ranking 2026) with derived columns: airport, local time,
off-block proxies, and quality flags. The aggregate builders turn it into the
small tables the app loads at start-up.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from prc.explore.common import AIRPORT_TZ, PROXIES, QUALITY_FLAGS, TIME_COLUMNS

RUNWAY_PATTERN = r"^\d{2}[LCR]?$"
STAND_HAS_DIGIT = r"\d"
AOBT_FAR_SEC = 900


def _seconds(end: str, start: str) -> pl.Expr:
    return (pl.col(end) - pl.col(start)).dt.total_seconds()


def _local_naive(col: str) -> pl.Expr:
    """Airport-local wall-clock time (naive datetime) of a UTC column."""

    expr: pl.Expr | None = None
    for airport, tz in AIRPORT_TZ.items():
        local = pl.col(col).dt.convert_time_zone(tz).dt.replace_time_zone(None)
        if expr is None:
            expr = pl.when(pl.col("airport") == airport).then(local)
        else:
            expr = expr.when(pl.col("airport") == airport).then(local)
    assert expr is not None
    return expr.otherwise(pl.col(col).dt.replace_time_zone(None))


def derive_movements(lf: pl.LazyFrame, dataset: str) -> pl.LazyFrame:
    """Add dataset label, airport, local time fields and off-block proxies."""

    lf = lf.with_columns(
        pl.lit(dataset).alias("dataset"),
        pl.when(pl.col("PHASE_mvt") == "DEP")
        .then(pl.col("ADEP_mvt"))
        .otherwise(pl.col("ADES_mvt"))
        .alias("airport"),
    )
    lf = lf.with_columns(_local_naive("MVT_TIME_UTC_mvt").alias("mvt_local"))
    lf = lf.with_columns(
        pl.col("mvt_local").dt.date().alias("local_date"),
        pl.col("mvt_local").dt.hour().cast(pl.Int8).alias("hour_local"),
        pl.col("mvt_local").dt.weekday().cast(pl.Int8).alias("weekday"),
        # Month follows the UTC file split (training files and the ranking
        # months are cut on UTC boundaries), not the local date.
        pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("month"),
    )
    is_dep = pl.col("PHASE_mvt") == "DEP"
    proxy_exprs = [
        pl.when(is_dep).then(_seconds("MVT_TIME_UTC_mvt", ref)).alias(name)
        for name, ref in PROXIES.items()
    ]
    return lf.with_columns(
        *proxy_exprs,
        pl.when(is_dep)
        .then(_seconds("BLOCK_TIME_UTC_mvt", "AOBT_3_flt"))
        .alias("blk_minus_aobt3"),
        pl.when(is_dep)
        .then(_seconds("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"))
        .alias("blk_minus_sched"),
    )


def add_quality_flags(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Boolean ``q_*`` columns, one per entry of ``QUALITY_FLAGS``.

    Flags are null-safe: a comparison involving a missing value is False,
    except ``q_no_nm_match`` which is about missingness itself.
    """

    is_dep = pl.col("PHASE_mvt") == "DEP"
    taxi = pl.col("TAXITIME_SEC_mvt")
    block_order = (
        pl.when(is_dep)
        .then(pl.col("BLOCK_TIME_UTC_mvt") > pl.col("MVT_TIME_UTC_mvt"))
        .otherwise(pl.col("BLOCK_TIME_UTC_mvt") < pl.col("MVT_TIME_UTC_mvt"))
    )
    airport_flt = pl.when(is_dep).then(pl.col("ADEP_flt")).otherwise(pl.col("ADES_flt"))
    stand = pl.col("STAND_mvt")
    exprs = {
        "q_taxi_le0": taxi <= 0,
        "q_taxi_gt1h": taxi > 3600,
        "q_taxi_gt2h": taxi > 7200,
        "q_block_order": block_order,
        "q_dup_mvt_id": pl.len().over("dataset", "MVT_ID_mvt") > 1,
        "q_dup_flight_day": pl.col("FLIGHT_mvt").is_not_null()
        & (pl.len().over("dataset", "airport", "PHASE_mvt", "FLIGHT_mvt", "local_date") > 1),
        "q_dup_nm_flight": pl.col("FLIGHT_ID_mvt").is_not_null()
        & (pl.len().over("dataset", "airport", "PHASE_mvt", "FLIGHT_ID_mvt") > 1),
        "q_no_nm_match": pl.col("FLIGHT_ID_mvt").is_null(),
        "q_airport_mismatch": airport_flt != pl.col("airport"),
        "q_diverted": pl.col("ADES_flt") != pl.col("ADES_FILED_flt"),
        "q_type_mismatch": pl.col("AIRCRAFT_TYPE_mvt") != pl.col("AIRCRAFT_TYPE_flt"),
        "q_runway_malformed": pl.col("RUNWAY_mvt").is_null()
        | ~pl.col("RUNWAY_mvt").str.contains(RUNWAY_PATTERN),
        "q_stand_malformed": stand.is_null()
        | (stand == "UNKNOWN")
        | ~stand.str.contains(STAND_HAS_DIGIT),
        "q_aobt_far": is_dep & (pl.col("blk_minus_aobt3").abs() > AOBT_FAR_SEC),
        "q_aobt_after_takeoff": is_dep & (pl.col("AOBT_3_flt") > pl.col("MVT_TIME_UTC_mvt")),
    }
    assert set(exprs) == set(QUALITY_FLAGS)
    return lf.with_columns(
        [expr.fill_null(False).alias(name) for name, expr in exprs.items()]
    )


def build_movements(train_paths: Sequence[Path], ranking_path: Path) -> pl.LazyFrame:
    """Training (2025) and ranking (2026) movements with all derived columns."""

    train = derive_movements(pl.scan_parquet(list(train_paths)), "train")
    ranking = derive_movements(pl.scan_parquet(ranking_path), "ranking")
    both = pl.concat([train, ranking], how="vertical")
    return add_quality_flags(both).sort("airport", "dataset", "PHASE_mvt", "MVT_TIME_UTC_mvt")


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------

GROUP = ("dataset", "PHASE_mvt", "airport")


def null_rates(lf: pl.LazyFrame, columns: Sequence[str]) -> pl.DataFrame:
    """Long table: dataset, phase, airport, column, n, null_rate, na_string_rate.

    ``na_string_rate`` counts the literal string "NA" (used in RUNWAY_mvt and
    FLIGHT_RULE_mvt for unknown values), which is not a null.
    """

    schema = lf.collect_schema()
    aggs: list[pl.Expr] = [pl.len().alias("n")]
    for col in columns:
        aggs.append(pl.col(col).null_count().alias(f"null::{col}"))
        if schema[col] == pl.String:
            aggs.append((pl.col(col) == "NA").sum().alias(f"na::{col}"))
    wide = lf.group_by(GROUP).agg(aggs).collect()
    long = wide.unpivot(index=[*GROUP, "n"], variable_name="key", value_name="count")
    long = long.with_columns(
        pl.col("key").str.split_exact("::", 1).struct.rename_fields(["kind", "column"])
    ).unnest("key")
    out = long.pivot(on="kind", index=[*GROUP, "n", "column"], values="count")
    if "na" not in out.columns:
        out = out.with_columns(pl.lit(None, dtype=pl.UInt32).alias("na"))
    return out.with_columns(
        (pl.col("null") / pl.col("n")).alias("null_rate"),
        (pl.col("na").fill_null(0) / pl.col("n")).alias("na_string_rate"),
    ).select(*GROUP, "column", "n", "null_rate", "na_string_rate")


def rounding_table(lf: pl.LazyFrame, columns: Sequence[str] = TIME_COLUMNS) -> pl.DataFrame:
    """Seconds-field diagnostics per column: share with seconds == 0, share
    with seconds a multiple of 5, and share with seconds in 7..53 ("mid").

    Uniform seconds give sec0 = 1/60 = 0.017, sec_mod5 = 0.2, sec_mid = 0.78.
    A minute-rounded column gives sec0 = 1. A minute-resolution column with a
    small added jitter (about +/-6 s) gives sec0 near 1/12 and sec_mid = 0.
    """

    aggs: list[pl.Expr] = []
    for col in columns:
        sec = pl.col(col).dt.second()
        aggs += [
            pl.col(col).count().alias(f"{col}::n"),
            (sec == 0).mean().alias(f"{col}::sec0"),
            (sec % 5 == 0).mean().alias(f"{col}::sec_mod5"),
            sec.is_between(7, 53).mean().alias(f"{col}::sec_mid"),
        ]
    wide = lf.group_by(GROUP).agg(aggs).collect()
    long = wide.unpivot(index=list(GROUP), variable_name="key", value_name="value")
    long = long.with_columns(
        pl.col("key").str.split_exact("::", 1).struct.rename_fields(["column", "stat"])
    ).unnest("key")
    return long.pivot(on="stat", index=[*GROUP, "column"], values="value").with_columns(
        pl.col("n").cast(pl.Int64)
    )


def quality_counts(lf: pl.LazyFrame) -> pl.DataFrame:
    """Long table: dataset, phase, airport, flag, n_rows, n_flagged, rate."""

    wide = (
        lf.group_by(GROUP)
        .agg(pl.len().alias("n_rows"), *[pl.col(f).sum().alias(f) for f in QUALITY_FLAGS])
        .collect()
    )
    return wide.unpivot(
        index=[*GROUP, "n_rows"], variable_name="flag", value_name="n_flagged"
    ).with_columns((pl.col("n_flagged") / pl.col("n_rows")).alias("rate"))


def daily_stats(lf: pl.LazyFrame) -> pl.DataFrame:
    """Per dataset, airport and local date: traffic counts and target stats."""

    is_dep = pl.col("PHASE_mvt") == "DEP"
    y = pl.col("TAXITIME_SEC_mvt").filter(is_dep)
    return (
        lf.group_by("dataset", "airport", "local_date")
        .agg(
            is_dep.sum().alias("n_dep"),
            (~is_dep).sum().alias("n_arr"),
            y.count().alias("n_target"),
            y.mean().alias("taxi_mean"),
            y.median().alias("taxi_median"),
            y.quantile(0.9).alias("taxi_p90"),
            pl.col("px_aobt3").filter(is_dep).mean().alias("px_aobt3_mean"),
        )
        .sort("dataset", "airport", "local_date")
        .collect()
    )


def hourly_counts(lf: pl.LazyFrame) -> pl.DataFrame:
    """Movements per dataset, airport, month, phase and local hour."""

    return (
        lf.group_by("dataset", "airport", "month", "PHASE_mvt", "hour_local")
        .agg(pl.len().alias("n"), pl.col("local_date").n_unique().alias("n_days"))
        .sort("dataset", "airport", "month", "PHASE_mvt", "hour_local")
        .collect()
    )


def runway_hourly(lf: pl.LazyFrame) -> pl.DataFrame:
    """Movements per dataset, airport, local date, hour, phase and runway."""

    return (
        lf.group_by("dataset", "airport", "local_date", "hour_local", "PHASE_mvt", "RUNWAY_mvt")
        .agg(pl.len().alias("n"))
        .sort("dataset", "airport", "local_date", "hour_local", "PHASE_mvt", "RUNWAY_mvt")
        .collect()
    )


def column_inventory(lf: pl.LazyFrame) -> pl.DataFrame:
    """dtype and n_unique per raw column, per dataset."""

    schema = lf.collect_schema()
    raw_cols = [c for c in schema if c.endswith("_mvt") or c.endswith("_flt")]
    uniq = (
        lf.group_by("dataset")
        .agg([pl.col(c).n_unique().alias(c) for c in raw_cols])
        .collect()
        .unpivot(index="dataset", variable_name="column", value_name="n_unique")
        .pivot(on="dataset", index="column", values="n_unique")
    )
    dtypes = pl.DataFrame({"column": raw_cols, "dtype": [str(schema[c]) for c in raw_cols]})
    return dtypes.join(uniq, on="column", how="left").rename(
        {"train": "n_unique_train", "ranking": "n_unique_ranking"}, strict=False
    )
