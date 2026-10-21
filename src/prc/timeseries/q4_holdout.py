"""Utilities for the fully masked 2025-Q4 forecasting holdout."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import polars as pl


Q4_START = datetime(2025, 10, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class ForecastCase:
    airport: str
    context_target: np.ndarray
    context_and_horizon_covariates: np.ndarray
    horizon_target: np.ndarray
    horizon_departure_count: np.ndarray


def make_q4_cases(state: pl.DataFrame) -> list[ForecastCase]:
    """Create one TimesFM-style case per airport with all of Q4 masked.

    The three covariate channels are departure, arrival, and scheduled
    departure counts. They are retained across the forecast horizon because
    the equivalent fields can be constructed from the ranking dataset.
    Missing context targets (mostly overnight bins) are interpolated only
    within the pre-Q4 context. Ground-truth Q4 values never enter the context.
    """

    cases: list[ForecastCase] = []
    for airport in state.get_column("airport").unique().sort():
        airport_state = state.filter(pl.col("airport") == airport).sort("timestamp")
        context = airport_state.filter(pl.col("timestamp") < Q4_START)
        horizon = airport_state.filter(pl.col("timestamp") >= Q4_START)
        if context.is_empty() or horizon.is_empty():
            raise ValueError(f"{airport} does not span both sides of the Q4 cutoff")

        context_target = (
            context.select(
                pl.col("taxi_out_mean_sec")
                .interpolate()
                .forward_fill()
                .backward_fill()
            )
            .to_series()
            .to_numpy()
            .astype(np.float32)
        )
        covariate_columns = [
            "departure_count",
            "arrival_count",
            "scheduled_departure_count",
        ]
        covariates = (
            pl.concat([context, horizon])
            .select(covariate_columns)
            .to_numpy()
            .T.astype(np.float32)
        )
        cases.append(
            ForecastCase(
                airport=airport,
                context_target=context_target,
                context_and_horizon_covariates=covariates,
                horizon_target=horizon.get_column("taxi_out_mean_sec")
                .to_numpy()
                .astype(np.float32),
                horizon_departure_count=horizon.get_column("departure_count")
                .to_numpy()
                .astype(np.float32),
            )
        )
    return cases


def weighted_bucket_rmse(
    actual: np.ndarray, predicted: np.ndarray, departure_count: np.ndarray
) -> float:
    """RMSE of bucket means weighted by the number of scored departures."""

    valid = np.isfinite(actual) & np.isfinite(predicted) & (departure_count > 0)
    if not valid.any():
        raise ValueError("No forecastable departure buckets")
    squared_error = np.square(actual[valid] - predicted[valid])
    return float(np.sqrt(np.average(squared_error, weights=departure_count[valid])))
