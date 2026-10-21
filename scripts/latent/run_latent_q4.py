#!/usr/bin/env python3
"""Compare the same per-flight CatBoost model without and with latent state."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_error

from prc.models.flight_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    load_departure_features,
)


Q4_START = datetime(2025, 10, 1, tzinfo=timezone.utc)
EXPERIMENT_DIR = Path("experiments/exp-002-predictive-latent-q4")


def fit_model(
    train: pl.DataFrame,
    validation: pl.DataFrame,
    features: list[str],
    name: str,
) -> tuple[np.ndarray, float]:
    model = CatBoostRegressor(
        iterations=600,
        depth=9,
        learning_rate=0.08,
        loss_function="RMSE",
        random_seed=20260925,
        thread_count=12,
        verbose=100,
        allow_writing_files=False,
    )
    train_pd = train.select(features).to_pandas()
    validation_pd = validation.select(features).to_pandas()
    started = perf_counter()
    model.fit(
        train_pd,
        train.get_column("target").to_numpy(),
        cat_features=CATEGORICAL_FEATURES,
    )
    predictions = model.predict(validation_pd).astype(np.float32)
    elapsed = perf_counter() - started
    Path("artefacts/latent").mkdir(parents=True, exist_ok=True)
    model.save_model(f"artefacts/latent/{name}.cbm")
    return predictions, elapsed


def main() -> None:
    paths = sorted(Path("data/raw").glob("training_*.parquet"))
    flights = load_departure_features(paths)
    latent = pl.read_parquet("data/processed/latent/pls_predictive_q4.parquet")
    latent_columns = [column for column in latent.columns if column.startswith("latent_z_")]
    flights = flights.join(latent, on=["airport", "timestamp"], how="left")
    latent_nulls = flights.select(
        pl.sum_horizontal([pl.col(column).null_count() for column in latent_columns])
    ).item()
    if latent_nulls != 0:
        raise ValueError("Some departures did not receive a latent airport state")

    train = flights.filter(pl.col("event_time") < Q4_START)
    validation = flights.filter(pl.col("event_time") >= Q4_START)
    baseline_features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    latent_features = baseline_features + latent_columns

    baseline_prediction, baseline_seconds = fit_model(
        train, validation, baseline_features, "catboost_q4_baseline"
    )
    latent_prediction, latent_seconds = fit_model(
        train, validation, latent_features, "catboost_q4_predictive_latent"
    )
    actual = validation.get_column("target").to_numpy()
    baseline_rmse = root_mean_squared_error(actual, baseline_prediction)
    latent_rmse = root_mean_squared_error(actual, latent_prediction)

    scored = validation.select("MVT_ID_mvt", "airport", "event_time", "target").with_columns(
        pl.Series("baseline_prediction", baseline_prediction),
        pl.Series("latent_prediction", latent_prediction),
        pl.Series("baseline_squared_error", np.square(actual - baseline_prediction)),
        pl.Series("latent_squared_error", np.square(actual - latent_prediction)),
    )
    by_airport = scored.group_by("airport").agg(
        pl.len().alias("flights"),
        pl.col("baseline_squared_error").mean().sqrt().alias("baseline_rmse_sec"),
        pl.col("latent_squared_error").mean().sqrt().alias("latent_rmse_sec"),
    ).with_columns(
        (pl.col("baseline_rmse_sec") - pl.col("latent_rmse_sec")).alias("improvement_sec")
    ).sort("airport")

    metrics = {
        "protocol": "VAL-Q4-MASKED-V1 per-flight downstream test",
        "train_flights": train.height,
        "validation_flights": validation.height,
        "baseline_rmse_sec": float(baseline_rmse),
        "latent_rmse_sec": float(latent_rmse),
        "improvement_sec": float(baseline_rmse - latent_rmse),
        "latent_dimensions": latent_columns,
        "baseline_fit_predict_seconds": baseline_seconds,
        "latent_fit_predict_seconds": latent_seconds,
        "by_airport": by_airport.to_dicts(),
    }
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    scored.write_parquet(EXPERIMENT_DIR / "predictions.parquet")
    (EXPERIMENT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
