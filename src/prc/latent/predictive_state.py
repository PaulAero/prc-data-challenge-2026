"""Learn a compact airport state that is predictive of future taxi conditions.

The encoder is deliberately linear for the first experiment. Partial Least
Squares (PLS) chooses latent directions in observable traffic features that
maximise covariance with future taxi-state targets. Unlike PCA, the limited
latent capacity is therefore spent on information relevant to future airport
operations rather than on reconstructing every high-variance input.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import pi

import numpy as np
import polars as pl
from sklearn.cross_decomposition import PLSRegression


COUNT_COLUMNS = (
    "departure_count",
    "arrival_count",
    "scheduled_departure_count",
)


def observable_feature_frame(state: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Create ranking-available temporal features without reading taxi targets."""

    state = state.sort("airport", "timestamp")
    expressions: list[pl.Expr] = []
    feature_names: list[str] = []

    for column in COUNT_COLUMNS:
        feature_names.append(column)
        for lag in (1, 4, 12, 24, 96):
            name = f"{column}_lag_{lag}"
            expressions.append(pl.col(column).shift(lag).over("airport").alias(name))
            feature_names.append(name)
        for window in (4, 12, 24):
            name = f"{column}_mean_{window}"
            expressions.append(
                pl.col(column).rolling_mean(window_size=window).over("airport").alias(name)
            )
            feature_names.append(name)
        for lead in (4, 12, 24):
            name = f"{column}_lead_{lead}"
            expressions.append(pl.col(column).shift(-lead).over("airport").alias(name))
            feature_names.append(name)

    engineered = state.select("airport", "timestamp", *COUNT_COLUMNS).with_columns(expressions)
    engineered = engineered.with_columns(
        (pl.col("departure_count") - pl.col("arrival_count")).alias("traffic_balance"),
        (pl.col("scheduled_departure_count") - pl.col("departure_count")).alias(
            "schedule_pressure"
        ),
        (
            pl.col("timestamp").dt.hour() * 60 + pl.col("timestamp").dt.minute()
        ).alias("minute_of_day"),
        pl.col("timestamp").dt.weekday().alias("weekday"),
    ).with_columns(
        (2 * pi * pl.col("minute_of_day") / 1440).sin().alias("tod_sin"),
        (2 * pi * pl.col("minute_of_day") / 1440).cos().alias("tod_cos"),
        (2 * pi * pl.col("weekday") / 7).sin().alias("week_sin"),
        (2 * pi * pl.col("weekday") / 7).cos().alias("week_cos"),
    )
    feature_names.extend(
        ["traffic_balance", "schedule_pressure", "tod_sin", "tod_cos", "week_sin", "week_cos"]
    )

    airports = engineered.get_column("airport").unique().sort().to_list()
    airport_features = []
    for airport in airports:
        name = f"airport_{airport}"
        airport_features.append((pl.col("airport") == airport).cast(pl.Float32).alias(name))
        feature_names.append(name)

    return (
        engineered.with_columns(airport_features)
        .select("airport", "timestamp", *feature_names)
        .with_columns(pl.col(feature_names).fill_null(0).cast(pl.Float32)),
        feature_names,
    )


def _future_target_frame(
    state: pl.DataFrame, cutoff: datetime, horizons: tuple[int, ...]
) -> tuple[pl.DataFrame, list[str]]:
    """Build training targets strictly before the holdout cutoff."""

    context = (
        state.filter(pl.col("timestamp") < cutoff)
        .sort("airport", "timestamp")
        .with_columns(
            pl.col("taxi_out_mean_sec")
            .interpolate()
            .forward_fill()
            .backward_fill()
            .over("airport")
            .alias("taxi_state")
        )
    )
    target_names = [f"taxi_state_future_{h}" for h in horizons]
    targets = context.select("airport", "timestamp", "taxi_state").with_columns(
        [
            pl.col("taxi_state").shift(-horizon).over("airport").alias(name)
            for horizon, name in zip(horizons, target_names, strict=True)
        ]
    )
    return targets.drop_nulls(target_names), target_names


@dataclass
class PredictiveLatentEncoder:
    """Fitted linear encoder from observable traffic into predictive state."""

    model: PLSRegression
    feature_names: list[str]
    latent_names: list[str]
    horizons: tuple[int, ...]

    @classmethod
    def fit(
        cls,
        state: pl.DataFrame,
        *,
        cutoff: datetime,
        n_components: int = 6,
        horizons: tuple[int, ...] = (4, 12, 24),
    ) -> "PredictiveLatentEncoder":
        features, feature_names = observable_feature_frame(state)
        targets, target_names = _future_target_frame(state, cutoff, horizons)
        training = features.join(targets, on=["airport", "timestamp"], how="inner")
        x = training.select(feature_names).to_numpy()
        y = training.select(target_names).to_numpy()
        model = PLSRegression(n_components=n_components, scale=True, max_iter=1_000)
        model.fit(x, y)
        return cls(
            model=model,
            feature_names=feature_names,
            latent_names=[f"latent_z_{index}" for index in range(n_components)],
            horizons=horizons,
        )

    def transform(self, state: pl.DataFrame) -> pl.DataFrame:
        features, feature_names = observable_feature_frame(state)
        if feature_names != self.feature_names:
            raise ValueError("Observable feature schema differs from the fitted encoder")
        latent = self.model.transform(features.select(feature_names).to_numpy()).astype(np.float32)
        return features.select("airport", "timestamp").with_columns(
            [pl.Series(name, latent[:, index]) for index, name in enumerate(self.latent_names)]
        )
