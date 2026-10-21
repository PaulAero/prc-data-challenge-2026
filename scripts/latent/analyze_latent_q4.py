#!/usr/bin/env python3
"""Add paired day-block uncertainty and feature importance to EXP-002."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostRegressor


EXPERIMENT_DIR = Path("experiments/exp-002-predictive-latent-q4")


def main() -> None:
    predictions = pl.read_parquet(EXPERIMENT_DIR / "predictions.parquet").with_columns(
        pl.col("event_time").dt.date().alias("date"),
        ((pl.col("target") - pl.col("baseline_prediction")) ** 2).alias("baseline_se"),
        ((pl.col("target") - pl.col("latent_prediction")) ** 2).alias("latent_se"),
    )
    daily = predictions.group_by("date").agg(
        pl.col("baseline_se").sum(),
        pl.col("latent_se").sum(),
        pl.len().alias("n"),
    ).sort("date")
    baseline_se = daily.get_column("baseline_se").to_numpy()
    latent_se = daily.get_column("latent_se").to_numpy()
    counts = daily.get_column("n").to_numpy()
    rng = np.random.default_rng(20260925)
    sampled = rng.integers(0, daily.height, size=(10_000, daily.height))
    baseline_rmse = np.sqrt(
        baseline_se[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    )
    latent_rmse = np.sqrt(latent_se[sampled].sum(axis=1) / counts[sampled].sum(axis=1))
    improvement = baseline_rmse - latent_rmse

    model = CatBoostRegressor()
    model.load_model("artefacts/latent/catboost_q4_predictive_latent.cbm")
    importance = sorted(
        zip(model.feature_names_, model.get_feature_importance(), strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )

    metrics_path = EXPERIMENT_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["paired_day_block_bootstrap"] = {
        "samples": 10_000,
        "seed": 20260925,
        "mean_improvement_sec": float(improvement.mean()),
        "median_improvement_sec": float(np.median(improvement)),
        "confidence_interval_95_sec": [
            float(value) for value in np.quantile(improvement, [0.025, 0.975])
        ],
        "probability_improvement": float((improvement > 0).mean()),
    }
    metrics["feature_importance"] = {
        name: float(value) for name, value in importance
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics["paired_day_block_bootstrap"], indent=2))
    print([(name, round(float(value), 3)) for name, value in importance[:15]])


if __name__ == "__main__":
    main()
