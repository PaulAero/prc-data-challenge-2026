#!/usr/bin/env python3
"""Fit independent future-predictive latent states for each airport."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import polars as pl

from prc.latent.per_airport_state import PerAirportPredictiveLatentEncoder


def main() -> None:
    state_path = Path("data/processed/airport_state_15m_training.parquet")
    output = Path("data/processed/latent/pls_per_airport_q4.parquet")
    model_output = Path("artefacts/latent/pls_per_airport_q4.pkl")
    state = pl.read_parquet(state_path)
    started = perf_counter()
    encoder = PerAirportPredictiveLatentEncoder.fit(
        state,
        cutoff=datetime(2025, 10, 1, tzinfo=timezone.utc),
        n_components=4,
        horizons=(4, 12, 24),
    )
    latent = encoder.transform(state)
    elapsed = perf_counter() - started

    output.parent.mkdir(parents=True, exist_ok=True)
    latent.write_parquet(output)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    model_output.write_bytes(pickle.dumps(encoder))
    metadata = {
        "method": "independent per-airport PLS predictive latent state",
        "fit_period": "2025-01-01 through 2025-09-30",
        "holdout_targets_used": False,
        "static_airport_identity_in_encoder": False,
        "future_target_horizons_hours": [1, 3, 6],
        "latent_dimensions": encoder.latent_names,
        "airports": sorted(encoder.encoders),
        "rows": latent.height,
        "fit_and_transform_seconds": elapsed,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
