#!/usr/bin/env python3
"""EXP-003: compare per-airport predictive state with the saved EXP-002 baseline."""

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
EXPERIMENT_DIR = Path("experiments/exp-003-per-airport-latent-q4")


def main() -> None:
    paths = sorted(Path("data/raw").glob("training_*.parquet"))
    flights = load_departure_features(paths)
    latent = pl.read_parquet("data/processed/latent/pls_per_airport_q4.parquet")
    latent_columns = [column for column in latent.columns if column.startswith("latent_z_")]
    flights = flights.join(latent, on=["airport", "timestamp"], how="left")
    latent_nulls = flights.select(
        pl.sum_horizontal([pl.col(column).null_count() for column in latent_columns])
    ).item()
    if latent_nulls:
        raise ValueError("Some departures did not receive a per-airport latent state")

    train = flights.filter(pl.col("event_time") < Q4_START)
    validation = flights.filter(pl.col("event_time") >= Q4_START)
    baseline = pl.read_parquet(
        "experiments/exp-002-predictive-latent-q4/predictions.parquet"
    ).select("MVT_ID_mvt", "baseline_prediction")
    validation = validation.join(baseline, on="MVT_ID_mvt", how="left", validate="1:1")
    if validation.get_column("baseline_prediction").null_count():
        raise ValueError("Saved baseline predictions do not cover the full Q4 holdout")

    features = CATEGORICAL_FEATURES + NUMERIC_FEATURES + latent_columns
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
    started = perf_counter()
    model.fit(
        train.select(features).to_pandas(),
        train.get_column("target").to_numpy(),
        cat_features=CATEGORICAL_FEATURES,
    )
    prediction = model.predict(validation.select(features).to_pandas()).astype(np.float32)
    elapsed = perf_counter() - started
    model.save_model("artefacts/latent/catboost_q4_per_airport_latent.cbm")

    actual = validation.get_column("target").to_numpy()
    baseline_prediction = validation.get_column("baseline_prediction").to_numpy()
    scored = validation.select(
        "MVT_ID_mvt", "airport", "event_time", "target", "baseline_prediction"
    ).with_columns(
        pl.Series("latent_prediction", prediction),
        pl.Series("baseline_se", np.square(actual - baseline_prediction)),
        pl.Series("latent_se", np.square(actual - prediction)),
    )
    by_airport = scored.group_by("airport").agg(
        pl.len().alias("flights"),
        pl.col("baseline_se").mean().sqrt().alias("baseline_rmse_sec"),
        pl.col("latent_se").mean().sqrt().alias("latent_rmse_sec"),
    ).with_columns(
        (pl.col("baseline_rmse_sec") - pl.col("latent_rmse_sec")).alias("improvement_sec")
    ).sort("airport")

    daily = scored.with_columns(pl.col("event_time").dt.date().alias("date")).group_by(
        "date"
    ).agg(pl.col("baseline_se").sum(), pl.col("latent_se").sum(), pl.len().alias("n"))
    baseline_se = daily.get_column("baseline_se").to_numpy()
    latent_se = daily.get_column("latent_se").to_numpy()
    counts = daily.get_column("n").to_numpy()
    rng = np.random.default_rng(20260925)
    sampled = rng.integers(0, daily.height, size=(10_000, daily.height))
    boot_improvement = np.sqrt(
        baseline_se[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    ) - np.sqrt(latent_se[sampled].sum(axis=1) / counts[sampled].sum(axis=1))

    baseline_rmse = root_mean_squared_error(actual, baseline_prediction)
    latent_rmse = root_mean_squared_error(actual, prediction)
    importance = sorted(
        zip(model.feature_names_, model.get_feature_importance(), strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )
    metrics = {
        "protocol": "VAL-Q4-MASKED-V1 per-flight downstream test",
        "variant": "independent per-airport predictive PLS, no airport identity input",
        "train_flights": train.height,
        "validation_flights": validation.height,
        "baseline_rmse_sec": float(baseline_rmse),
        "latent_rmse_sec": float(latent_rmse),
        "improvement_sec": float(baseline_rmse - latent_rmse),
        "latent_fit_predict_seconds": elapsed,
        "by_airport": by_airport.to_dicts(),
        "paired_day_block_bootstrap": {
            "samples": 10_000,
            "seed": 20260925,
            "median_improvement_sec": float(np.median(boot_improvement)),
            "confidence_interval_95_sec": [
                float(value) for value in np.quantile(boot_improvement, [0.025, 0.975])
            ],
            "probability_improvement": float((boot_improvement > 0).mean()),
        },
        "feature_importance": {name: float(value) for name, value in importance},
    }
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    scored.write_parquet(EXPERIMENT_DIR / "predictions.parquet")
    (EXPERIMENT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
