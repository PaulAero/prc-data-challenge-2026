"""Ranking-available per-flight features for the Q4 validation protocol."""

from __future__ import annotations

from collections.abc import Sequence
from math import pi
from pathlib import Path

import polars as pl


CATEGORICAL_FEATURES = [
    "airport",
    "destination",
    "runway",
    "stand",
    "aircraft_type",
    "market_segment",
    "wake_category",
    "operator",
    "flight_type",
    "flight_rule",
]

NUMERIC_FEATURES = [
    "aobt_proxy_sec",
    "mvt_vs_schedule_sec",
    "mvt_vs_lobt_sec",
    "mvt_vs_iobt_sec",
    "mvt_vs_eobt_sec",
    "aobt_vs_eobt_sec",
    "tod_sin",
    "tod_cos",
    "week_sin",
    "week_cos",
    "month",
]


def _seconds_between(left: str, right: str, alias: str) -> pl.Expr:
    return (pl.col(left) - pl.col(right)).dt.total_seconds().cast(pl.Float32).alias(alias)


def load_departure_features(paths: Sequence[str | Path]) -> pl.DataFrame:
    """Read official training departures and build test-available features."""

    selected = [
        "MVT_ID_mvt",
        "PHASE_mvt",
        "ADEP_mvt",
        "ADES_mvt",
        "MVT_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt",
        "TAXITIME_SEC_mvt",
        "RUNWAY_mvt",
        "STAND_mvt",
        "AIRCRAFT_TYPE_mvt",
        "MARKET_SEGMENT_flt",
        "WK_TBL_CAT_flt",
        "AIRCRAFT_OPERATOR_flt",
        "FLIGHT_TYPE_flt",
        "FLIGHT_RULE_flt",
        "LOBT_flt",
        "IOBT_flt",
        "EOBT_1_flt",
        "AOBT_3_flt",
    ]
    frame = (
        pl.scan_parquet([Path(path) for path in paths])
        .select(selected)
        .filter(pl.col("PHASE_mvt") == "DEP")
        .collect(engine="streaming")
        .rename(
            {
                "ADEP_mvt": "airport",
                "ADES_mvt": "destination",
                "MVT_TIME_UTC_mvt": "event_time",
                "TAXITIME_SEC_mvt": "target",
                "RUNWAY_mvt": "runway",
                "STAND_mvt": "stand",
                "AIRCRAFT_TYPE_mvt": "aircraft_type",
                "MARKET_SEGMENT_flt": "market_segment",
                "WK_TBL_CAT_flt": "wake_category",
                "AIRCRAFT_OPERATOR_flt": "operator",
                "FLIGHT_TYPE_flt": "flight_type",
                "FLIGHT_RULE_flt": "flight_rule",
            }
        )
        .with_columns(
            pl.col("event_time").dt.truncate("15m").alias("timestamp"),
            _seconds_between("event_time", "AOBT_3_flt", "aobt_proxy_sec"),
            _seconds_between("event_time", "SCHED_TIME_UTC_mvt", "mvt_vs_schedule_sec"),
            _seconds_between("event_time", "LOBT_flt", "mvt_vs_lobt_sec"),
            _seconds_between("event_time", "IOBT_flt", "mvt_vs_iobt_sec"),
            _seconds_between("event_time", "EOBT_1_flt", "mvt_vs_eobt_sec"),
            _seconds_between("AOBT_3_flt", "EOBT_1_flt", "aobt_vs_eobt_sec"),
            (
                pl.col("event_time").dt.hour() * 60 + pl.col("event_time").dt.minute()
            ).alias("minute_of_day"),
            pl.col("event_time").dt.weekday().alias("weekday"),
            pl.col("event_time").dt.month().cast(pl.Float32).alias("month"),
        )
        .with_columns(
            (2 * pi * pl.col("minute_of_day") / 1440).sin().cast(pl.Float32).alias("tod_sin"),
            (2 * pi * pl.col("minute_of_day") / 1440).cos().cast(pl.Float32).alias("tod_cos"),
            (2 * pi * pl.col("weekday") / 7).sin().cast(pl.Float32).alias("week_sin"),
            (2 * pi * pl.col("weekday") / 7).cos().cast(pl.Float32).alias("week_cos"),
        )
        .with_columns(
            pl.col(CATEGORICAL_FEATURES).fill_null("<NA>"),
            pl.col("target").cast(pl.Float32),
        )
    )
    return frame.select(
        "MVT_ID_mvt",
        "event_time",
        "timestamp",
        "target",
        *CATEGORICAL_FEATURES,
        *NUMERIC_FEATURES,
    )
