"""EXP-039: the D24 date artefact as a rule (LIRF only).

Mechanism (2025): the recorded off-block keeps the actual local time of day and takes the calendar
date nearest to the schedule. A departure whose real off-block is on a later local date than the
schedule, at a time of day within 12 h of the scheduled time of day, is therefore stored 24 h
early (target = 86,400 s + taxi). LIRF no-NM 2025, non-fills with the real off-block on a later date: |delta| < 12 h -> 11 of 11 D24;
night crossings (delta > 18 h) -> 23 normal, 1 D24 (ITY332). Matched LIRF and the other airports: 0 D24
in 984 comparable rows. Other airports date these correctly.

zone(frame, reg) -> bool array: LIRF, no NM, estimated off-block (take-off - reg) on a later local
date than the schedule, and |schedule tod - off-block tod| < 12 h.
"""
from __future__ import annotations

import numpy as np
import polars as pl

TZ = "Europe/Rome"


def zone(frame: pl.DataFrame, reg: np.ndarray, t_col: str = "event_time") -> np.ndarray:
    f = frame.select(
        pl.col(t_col).alias("t"),
        (pl.col(t_col) - pl.duration(seconds=pl.col("mvt_vs_schedule_sec").fill_null(0).cast(pl.Int64))).alias("s"),
        (pl.col("airport") == "LIRF").alias("lirf"), (pl.col("no_nm") == 1).alias("nonm"),
    ).with_columns(pl.Series("reg", np.clip(np.nan_to_num(reg, nan=900.0), 300.0, 1500.0)))  # robust taxi for the date test
    f = f.with_columns((pl.col("t") - pl.duration(seconds=pl.col("reg").cast(pl.Int64))).alias("ob"))
    loc = lambda c: pl.col(c).dt.convert_time_zone(TZ).dt.replace_time_zone(None)  # noqa: E731
    tod = lambda c: (loc(c).dt.hour().cast(pl.Int32) * 3600 + loc(c).dt.minute().cast(pl.Int32) * 60  # noqa: E731
                     + loc(c).dt.second().cast(pl.Int32))
    f = f.with_columns(later=loc("ob").dt.date() > loc("s").dt.date(), delta=(tod("s") - tod("ob")) / 3600.0)
    z = f.select(pl.col("lirf") & pl.col("nonm") & pl.col("later") & (pl.col("delta").abs() < 12))
    return z.to_series().fill_null(False).to_numpy()
