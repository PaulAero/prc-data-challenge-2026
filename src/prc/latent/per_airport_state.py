"""Within-airport predictive latent states without static airport identity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl
from sklearn.cross_decomposition import PLSRegression

from prc.latent.predictive_state import _future_target_frame, observable_feature_frame


@dataclass
class FittedAirportEncoder:
    model: PLSRegression
    feature_names: list[str]
    latent_names: list[str]

    def transform(self, state: pl.DataFrame) -> pl.DataFrame:
        features, available = observable_feature_frame(state)
        missing = set(self.feature_names).difference(available)
        if missing:
            raise ValueError(f"Missing observable features: {sorted(missing)}")
        latent = self.model.transform(features.select(self.feature_names).to_numpy()).astype(
            np.float32
        )
        return features.select("airport", "timestamp").with_columns(
            [pl.Series(name, latent[:, index]) for index, name in enumerate(self.latent_names)]
        )


@dataclass
class PerAirportPredictiveLatentEncoder:
    """Fit independent dynamic-state axes for every airport."""

    encoders: dict[str, FittedAirportEncoder]
    latent_names: list[str]
    horizons: tuple[int, ...]

    @classmethod
    def fit(
        cls,
        state: pl.DataFrame,
        *,
        cutoff: datetime,
        n_components: int = 4,
        horizons: tuple[int, ...] = (4, 12, 24),
    ) -> "PerAirportPredictiveLatentEncoder":
        latent_names = [f"latent_z_{index}" for index in range(n_components)]
        encoders: dict[str, FittedAirportEncoder] = {}
        for airport in state.get_column("airport").unique().sort():
            airport_state = state.filter(pl.col("airport") == airport)
            features, all_feature_names = observable_feature_frame(airport_state)
            feature_names = [
                name for name in all_feature_names if not name.startswith("airport_")
            ]
            targets, target_names = _future_target_frame(airport_state, cutoff, horizons)
            training = features.join(targets, on=["airport", "timestamp"], how="inner")
            model = PLSRegression(
                n_components=n_components,
                scale=True,
                max_iter=1_000,
            )
            model.fit(
                training.select(feature_names).to_numpy(),
                training.select(target_names).to_numpy(),
            )
            encoders[airport] = FittedAirportEncoder(
                model=model,
                feature_names=feature_names,
                latent_names=latent_names,
            )
        return cls(encoders=encoders, latent_names=latent_names, horizons=horizons)

    def transform(self, state: pl.DataFrame) -> pl.DataFrame:
        present = set(state.get_column("airport").unique().to_list())
        expected = set(self.encoders)
        if present != expected:
            raise ValueError(
                f"Airport set differs from fitted encoders: missing={sorted(expected - present)}, "
                f"unexpected={sorted(present - expected)}"
            )
        return pl.concat(
            [
                encoder.transform(state.filter(pl.col("airport") == airport))
                for airport, encoder in sorted(self.encoders.items())
            ]
        ).sort("airport", "timestamp")
