"""Shared predictive latent geometry over airport-seasonal residuals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl
from sklearn.cross_decomposition import PLSRegression

from prc.latent.predictive_state import observable_feature_frame


def _with_week_slot(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        (
            (pl.col("timestamp").dt.weekday().cast(pl.Int16) - 1) * 96
            + pl.col("timestamp").dt.hour().cast(pl.Int16) * 4
            + pl.col("timestamp").dt.minute().cast(pl.Int16) // 15
        )
        .cast(pl.UInt16)
        .alias("week_slot")
    )


@dataclass
class SharedResidualPredictiveEncoder:
    """Encode deviations from each airport's normal time-of-week state."""

    model: PLSRegression
    raw_feature_names: list[str]
    residual_feature_names: list[str]
    latent_names: list[str]
    horizons: tuple[int, ...]
    feature_profile: pl.DataFrame
    feature_fallback: pl.DataFrame

    @staticmethod
    def _dynamic_features(all_names: list[str]) -> list[str]:
        excluded = {"tod_sin", "tod_cos", "week_sin", "week_cos"}
        return [
            name
            for name in all_names
            if not name.startswith("airport_") and name not in excluded
        ]

    @classmethod
    def fit(
        cls,
        state: pl.DataFrame,
        *,
        cutoff: datetime,
        n_components: int = 6,
        horizons: tuple[int, ...] = (4, 12, 24),
    ) -> "SharedResidualPredictiveEncoder":
        raw, all_names = observable_feature_frame(state)
        raw_feature_names = cls._dynamic_features(all_names)
        raw = _with_week_slot(raw)
        context_raw = raw.filter(pl.col("timestamp") < cutoff)
        expected_names = [f"expected_{name}" for name in raw_feature_names]
        feature_profile = context_raw.group_by("airport", "week_slot").agg(
            [
                pl.col(name).mean().alias(expected)
                for name, expected in zip(raw_feature_names, expected_names, strict=True)
            ]
        )
        feature_fallback = context_raw.group_by("airport").agg(
            [
                pl.col(name).mean().alias(f"fallback_{name}")
                for name in raw_feature_names
            ]
        )
        residual_features = cls._residualize(
            raw,
            raw_feature_names,
            feature_profile,
            feature_fallback,
        )

        context_target = (
            _with_week_slot(state.filter(pl.col("timestamp") < cutoff))
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
        target_profile = context_target.group_by("airport", "week_slot").agg(
            pl.col("taxi_state").mean().alias("expected_taxi_state")
        )
        context_target = context_target.join(
            target_profile, on=["airport", "week_slot"], how="left"
        ).with_columns(
            (pl.col("taxi_state") - pl.col("expected_taxi_state")).alias("taxi_residual")
        )
        target_names = [f"taxi_residual_future_{horizon}" for horizon in horizons]
        targets = context_target.select("airport", "timestamp", "taxi_residual").with_columns(
            [
                pl.col("taxi_residual")
                .shift(-horizon)
                .over("airport")
                .alias(name)
                for horizon, name in zip(horizons, target_names, strict=True)
            ]
        ).drop_nulls(target_names)

        residual_feature_names = [f"residual_{name}" for name in raw_feature_names]
        training = residual_features.join(targets, on=["airport", "timestamp"], how="inner")
        model = PLSRegression(n_components=n_components, scale=True, max_iter=1_000)
        model.fit(
            training.select(residual_feature_names).to_numpy(),
            training.select(target_names).to_numpy(),
        )
        return cls(
            model=model,
            raw_feature_names=raw_feature_names,
            residual_feature_names=residual_feature_names,
            latent_names=[f"latent_z_{index}" for index in range(n_components)],
            horizons=horizons,
            feature_profile=feature_profile,
            feature_fallback=feature_fallback,
        )

    @staticmethod
    def _residualize(
        raw: pl.DataFrame,
        raw_feature_names: list[str],
        feature_profile: pl.DataFrame,
        feature_fallback: pl.DataFrame,
    ) -> pl.DataFrame:
        joined = raw.join(
            feature_profile, on=["airport", "week_slot"], how="left"
        ).join(feature_fallback, on="airport", how="left")
        return joined.select(
            "airport",
            "timestamp",
            *[
                (
                    pl.col(name)
                    - pl.col(f"expected_{name}").fill_null(pl.col(f"fallback_{name}"))
                )
                .cast(pl.Float32)
                .alias(f"residual_{name}")
                for name in raw_feature_names
            ],
        )

    def transform(self, state: pl.DataFrame) -> pl.DataFrame:
        raw, all_names = observable_feature_frame(state)
        dynamic = self._dynamic_features(all_names)
        if dynamic != self.raw_feature_names:
            raise ValueError("Residual observable feature schema differs from fitted encoder")
        residual = self._residualize(
            _with_week_slot(raw),
            self.raw_feature_names,
            self.feature_profile,
            self.feature_fallback,
        )
        latent = self.model.transform(
            residual.select(self.residual_feature_names).to_numpy()
        ).astype(np.float32)
        return residual.select("airport", "timestamp").with_columns(
            [pl.Series(name, latent[:, index]) for index, name in enumerate(self.latent_names)]
        )
