"""Soft operating regimes learned over predictive airport-state coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass
class SoftRegimeEncoder:
    scaler: StandardScaler
    model: GaussianMixture
    feature_names: list[str]

    @classmethod
    def fit(
        cls,
        state: pl.DataFrame,
        *,
        cutoff: datetime,
        n_regimes: int = 8,
        seed: int = 20260925,
    ) -> "SoftRegimeEncoder":
        feature_names = [
            column
            for column in state.columns
            if column.startswith("raw_latent_z_")
            or column.startswith("residual_latent_z_")
        ]
        if not feature_names:
            raise ValueError("No combined latent columns found")
        context = state.filter(pl.col("timestamp") < cutoff).select(feature_names).to_numpy()
        scaler = StandardScaler().fit(context)
        model = GaussianMixture(
            n_components=n_regimes,
            covariance_type="diag",
            reg_covar=1e-4,
            n_init=3,
            max_iter=300,
            random_state=seed,
        ).fit(scaler.transform(context))
        return cls(scaler=scaler, model=model, feature_names=feature_names)

    def transform(self, state: pl.DataFrame) -> pl.DataFrame:
        values = self.scaler.transform(state.select(self.feature_names).to_numpy())
        probability = self.model.predict_proba(values).astype(np.float32)
        labels = probability.argmax(axis=1)
        entropy = -np.sum(probability * np.log(np.clip(probability, 1e-8, 1.0)), axis=1)
        surprise = -self.model.score_samples(values)
        return state.select("airport", "timestamp").with_columns(
            pl.Series("regime", [f"regime_{label}" for label in labels]),
            pl.Series("regime_entropy", entropy.astype(np.float32)),
            pl.Series("regime_surprise", surprise.astype(np.float32)),
            *[
                pl.Series(f"regime_probability_{index}", probability[:, index])
                for index in range(probability.shape[1])
            ],
        )
