#!/usr/bin/env python3
"""Fit and materialize the first future-predictive airport latent state."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import polars as pl

from prc.latent.predictive_state import PredictiveLatentEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("data/processed/airport_state_15m_training.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/latent/pls_predictive_q4.parquet"),
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("artefacts/latent/pls_predictive_q4.pkl"),
    )
    parser.add_argument("--components", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cutoff = datetime(2025, 10, 1, tzinfo=timezone.utc)
    state = pl.read_parquet(args.state)
    started = perf_counter()
    encoder = PredictiveLatentEncoder.fit(
        state,
        cutoff=cutoff,
        n_components=args.components,
        horizons=(4, 12, 24),
    )
    latent = encoder.transform(state)
    elapsed = perf_counter() - started

    args.output.parent.mkdir(parents=True, exist_ok=True)
    latent.write_parquet(args.output)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.write_bytes(pickle.dumps(encoder))

    metadata = {
        "method": "PLSRegression predictive latent state",
        "fit_period": "2025-01-01 through 2025-09-30",
        "holdout_targets_used": False,
        "interval": "15m",
        "future_target_horizons_bins": list(encoder.horizons),
        "future_target_horizons_hours": [horizon / 4 for horizon in encoder.horizons],
        "latent_dimensions": encoder.latent_names,
        "observable_feature_count": len(encoder.feature_names),
        "rows": latent.height,
        "fit_and_transform_seconds": elapsed,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
