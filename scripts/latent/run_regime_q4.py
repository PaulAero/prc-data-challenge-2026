#!/usr/bin/env python3
"""EXP-006: add soft operating regimes to the combined predictive state."""

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
EXPERIMENT_DIR = Path("experiments/exp-006-soft-regime-q4")


def renamed_latent(path: str, prefix: str) -> tuple[pl.DataFrame, list[str]]:
    latent = pl.read_parquet(path)
    source = [column for column in latent.columns if column.startswith("latent_z_")]
    renamed = [f"{prefix}_{column}" for column in source]
    return latent.rename(dict(zip(source, renamed, strict=True))), renamed


def run_experiment(
    *,
    experiment_dir: Path = EXPERIMENT_DIR,
    variant: str = "combined predictive state plus 8 soft GMM regimes",
    add_airport_regime: bool = False,
    model_path: str = "artefacts/latent/catboost_q4_soft_regimes.cbm",
) -> None:
    flights = load_departure_features(sorted(Path("data/raw").glob("training_*.parquet")))
    raw, raw_columns = renamed_latent(
        "data/processed/latent/pls_predictive_q4.parquet", "raw"
    )
    residual, residual_columns = renamed_latent(
        "data/processed/latent/pls_residual_shared_q4.parquet", "residual"
    )
    regimes = pl.read_parquet("data/processed/latent/gmm_regimes_q4.parquet")
    regime_numeric = [
        column
        for column in regimes.columns
        if column.startswith("regime_probability_")
        or column in {"regime_entropy", "regime_surprise"}
    ]
    state_columns = raw_columns + residual_columns + regime_numeric
    flights = (
        flights.join(raw, on=["airport", "timestamp"], how="left")
        .join(residual, on=["airport", "timestamp"], how="left")
        .join(regimes, on=["airport", "timestamp"], how="left")
    )
    if add_airport_regime:
        flights = flights.with_columns(
            pl.concat_str("airport", "regime", separator="__").alias("airport_regime")
        )
    if flights.select(
        pl.sum_horizontal([pl.col(column).null_count() for column in state_columns])
    ).item():
        raise ValueError("Some departures did not receive latent regime features")

    train = flights.filter(pl.col("event_time") < Q4_START)
    validation = flights.filter(pl.col("event_time") >= Q4_START)
    baseline = pl.read_parquet(
        "experiments/exp-002-predictive-latent-q4/predictions.parquet"
    ).select("MVT_ID_mvt", "baseline_prediction")
    validation = validation.join(baseline, on="MVT_ID_mvt", how="left", validate="1:1")
    categorical = CATEGORICAL_FEATURES + ["regime"]
    if add_airport_regime:
        categorical.append("airport_regime")
    features = categorical + NUMERIC_FEATURES + state_columns

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
        cat_features=categorical,
    )
    prediction = model.predict(validation.select(features).to_pandas()).astype(np.float32)
    elapsed = perf_counter() - started
    model.save_model(model_path)

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
    bse = daily.get_column("baseline_se").to_numpy()
    lse = daily.get_column("latent_se").to_numpy()
    counts = daily.get_column("n").to_numpy()
    rng = np.random.default_rng(20260925)
    sampled = rng.integers(0, daily.height, size=(10_000, daily.height))
    boot = np.sqrt(bse[sampled].sum(1) / counts[sampled].sum(1)) - np.sqrt(
        lse[sampled].sum(1) / counts[sampled].sum(1)
    )
    baseline_rmse = root_mean_squared_error(actual, baseline_prediction)
    latent_rmse = root_mean_squared_error(actual, prediction)
    importance = sorted(
        zip(model.feature_names_, model.get_feature_importance(), strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )
    metrics = {
        "protocol": "VAL-Q4-MASKED-V1 per-flight downstream test",
        "variant": variant,
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
            "median_improvement_sec": float(np.median(boot)),
            "confidence_interval_95_sec": [
                float(value) for value in np.quantile(boot, [0.025, 0.975])
            ],
            "probability_improvement": float((boot > 0).mean()),
        },
        "feature_importance": {name: float(value) for name, value in importance},
    }
    experiment_dir.mkdir(parents=True, exist_ok=True)
    scored.write_parquet(experiment_dir / "predictions.parquet")
    (experiment_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
