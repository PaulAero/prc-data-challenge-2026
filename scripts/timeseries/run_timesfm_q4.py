#!/usr/bin/env python3
"""Evaluate TimesFM 3 with the complete 2025-Q4 target masked.

This is an evaluation-only experiment. TimesFM 3 pretrained weights are not a
permitted dependency of the final competition solution unless their separate
licence is independently confirmed to allow that use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from prc.timeseries.q4_holdout import Q4_START, make_q4_cases, weighted_bucket_rmse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("data/processed/airport_state_15m_training.parquet"),
    )
    parser.add_argument("--checkpoint", default="google/timesfm-3.0-pytorch")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--context-length", type=int, default=15_360)
    parser.add_argument("--airport", action="append", help="Repeat to select airports")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/exp-001-timesfm-q4")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = pl.read_parquet(args.state)
    cases = make_q4_cases(state)
    if args.airport:
        selected = set(args.airport)
        cases = [case for case in cases if case.airport in selected]
        missing = selected.difference(case.airport for case in cases)
        if missing:
            raise ValueError(f"Unknown airports: {sorted(missing)}")

    horizon = len(cases[0].horizon_target)
    if any(len(case.horizon_target) != horizon for case in cases):
        raise ValueError("All selected airport horizons must have equal length")

    print(f"Q4 cutoff: {Q4_START.isoformat()}")
    print(f"airports: {[case.airport for case in cases]}")
    print(f"context used: {min(args.context_length, len(cases[0].context_target)):,} bins")
    print(f"fully masked horizon: {horizon:,} bins")
    print("future covariates: actual departure count, actual arrival count, scheduled departures")
    if args.dry_run:
        return

    try:
        from timesfm3 import ModelConfig, TimesFM3Evaluator
    except ImportError as exc:
        raise SystemExit(
            "TimesFM 3 is not installed. Install it only after accepting the "
            "checkpoint's separate non-commercial licence."
        ) from exc

    contexts: list[np.ndarray] = []
    past_future_covariates: list[np.ndarray] = []
    for case in cases:
        context_length = min(args.context_length, len(case.context_target))
        contexts.append(case.context_target[-context_length:][None, :])
        past_future_covariates.append(
            case.context_and_horizon_covariates[:, -(context_length + horizon) :]
        )

    config = ModelConfig(
        checkpoint_path=args.checkpoint,
        per_core_batch_size=1,
        device=args.device,
    )
    forecaster = TimesFM3Evaluator(config)
    started = perf_counter()
    outputs = list(
        forecaster.predict_batch(
            contexts=contexts,
            horizon=horizon,
            past_future_covariates=past_future_covariates,
            return_quantiles=True,
            use_symmetric_averaging=False,
        )
    )
    elapsed = perf_counter() - started

    prediction_frames: list[pl.DataFrame] = []
    scores: dict[str, float] = {}
    all_actual: list[np.ndarray] = []
    all_predicted: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    timestamps = (
        state.filter(pl.col("timestamp") >= Q4_START)
        .get_column("timestamp")
        .unique()
        .sort()
    )
    for case, output in zip(cases, outputs, strict=True):
        predicted = np.asarray(output.forecast).reshape(-1).astype(np.float32)
        if len(predicted) != horizon:
            raise ValueError(
                f"{case.airport}: expected {horizon} predictions, got {len(predicted)}"
            )
        scores[case.airport] = weighted_bucket_rmse(
            case.horizon_target, predicted, case.horizon_departure_count
        )
        all_actual.append(case.horizon_target)
        all_predicted.append(predicted)
        all_weights.append(case.horizon_departure_count)
        prediction_frames.append(
            pl.DataFrame(
                {
                    "airport": case.airport,
                    "timestamp": timestamps,
                    "actual_taxi_out_mean_sec": case.horizon_target,
                    "predicted_taxi_out_mean_sec": predicted,
                    "departure_count": case.horizon_departure_count,
                }
            )
        )

    scores["overall"] = weighted_bucket_rmse(
        np.concatenate(all_actual),
        np.concatenate(all_predicted),
        np.concatenate(all_weights),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pl.concat(prediction_frames).write_parquet(args.output_dir / "predictions.parquet")
    metrics = {
        "protocol": "mask all 2025-Q4 taxi-out targets",
        "interval": "15m",
        "context_length": args.context_length,
        "horizon": horizon,
        "airports": [case.airport for case in cases],
        "weighted_bucket_rmse_sec": scores,
        "inference_seconds": elapsed,
        "checkpoint": args.checkpoint,
        "submission_eligible": False,
        "submission_eligibility_reason": "TimesFM 3 checkpoint has a separate non-commercial licence",
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
