#!/usr/bin/env python3
"""Score saved Q4 airport-state predictions at the official per-flight level."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl


Q4_START = datetime(2025, 10, 1, tzinfo=timezone.utc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--experiment-dir", type=Path, default=Path("experiments/exp-001-timesfm-q4")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_files = sorted(args.raw_dir.glob("training_*.parquet"))
    departures = (
        pl.scan_parquet(training_files)
        .filter(pl.col("PHASE_mvt") == "DEP")
        .select(
            pl.col("ADEP_mvt").alias("airport"),
            pl.col("MVT_TIME_UTC_mvt").alias("time"),
            pl.col("MVT_TIME_UTC_mvt").dt.truncate("15m").alias("timestamp"),
            pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("actual"),
        )
        .collect(engine="streaming")
        .with_columns(
            (
                pl.col("time").dt.weekday() * 96
                + pl.col("time").dt.hour() * 4
                + pl.col("time").dt.minute() // 15
            ).alias("slot")
        )
    )
    context = departures.filter(pl.col("time") < Q4_START)
    holdout = departures.filter(pl.col("time") >= Q4_START)

    airport_mean = context.group_by("airport").agg(
        pl.col("actual").mean().alias("airport_mean")
    )
    weekly_profile = context.group_by("airport", "slot").agg(
        pl.col("actual").mean().alias("weekly_profile")
    )
    predictions = pl.read_parquet(args.experiment_dir / "predictions.parquet").select(
        "airport",
        "timestamp",
        pl.col("predicted_taxi_out_mean_sec").cast(pl.Float64).alias("timesfm"),
    )
    oracle = holdout.group_by("airport", "timestamp").agg(
        pl.col("actual").mean().alias("oracle_15m_bucket_mean")
    )
    scored = (
        holdout.join(predictions, on=["airport", "timestamp"])
        .join(airport_mean, on="airport")
        .join(weekly_profile, on=["airport", "slot"])
        .join(oracle, on=["airport", "timestamp"])
        .with_columns(
            ((pl.col("timesfm") - pl.col("actual")) ** 2).alias("timesfm_se"),
            ((pl.col("weekly_profile") - pl.col("actual")) ** 2).alias("profile_se"),
            ((pl.col("airport_mean") - pl.col("actual")) ** 2).alias("mean_se"),
            ((pl.col("oracle_15m_bucket_mean") - pl.col("actual")) ** 2).alias(
                "oracle_se"
            ),
        )
    )

    overall = scored.select(
        pl.len().alias("flights"),
        pl.col("timesfm_se").mean().sqrt().alias("timesfm_rmse_sec"),
        pl.col("profile_se").mean().sqrt().alias("weekly_profile_rmse_sec"),
        pl.col("mean_se").mean().sqrt().alias("airport_mean_rmse_sec"),
        pl.col("oracle_se").mean().sqrt().alias("oracle_15m_bucket_rmse_sec"),
    ).row(0, named=True)
    by_airport = scored.group_by("airport").agg(
        pl.len().alias("flights"),
        pl.col("timesfm_se").mean().sqrt().alias("timesfm_rmse_sec"),
        pl.col("profile_se").mean().sqrt().alias("weekly_profile_rmse_sec"),
        pl.col("mean_se").mean().sqrt().alias("airport_mean_rmse_sec"),
        pl.col("oracle_se").mean().sqrt().alias("oracle_15m_bucket_rmse_sec"),
    ).sort("airport")

    scored.select(
        "airport", "time", "actual", "timesfm", "weekly_profile", "airport_mean"
    ).write_parquet(args.experiment_dir / "flight_predictions.parquet")
    metrics_path = args.experiment_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["per_flight_rmse_sec"] = {
        key: float(value) if isinstance(value, float) else value
        for key, value in overall.items()
    }
    metrics["per_airport_per_flight"] = by_airport.to_dicts()
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(pl.DataFrame([overall]))
    print(by_airport)


if __name__ == "__main__":
    main()
