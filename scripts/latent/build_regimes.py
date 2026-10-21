#!/usr/bin/env python3
"""Fit Q1-Q3 operating regimes over the combined predictive state."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import polars as pl

from prc.latent.regimes import SoftRegimeEncoder


def main() -> None:
    raw = pl.read_parquet("data/processed/latent/pls_predictive_q4.parquet")
    raw_columns = [column for column in raw.columns if column.startswith("latent_z_")]
    raw = raw.rename({column: f"raw_{column}" for column in raw_columns})
    residual = pl.read_parquet("data/processed/latent/pls_residual_shared_q4.parquet")
    residual_columns = [
        column for column in residual.columns if column.startswith("latent_z_")
    ]
    residual = residual.rename(
        {column: f"residual_{column}" for column in residual_columns}
    )
    combined = raw.join(residual, on=["airport", "timestamp"], how="inner")
    started = perf_counter()
    encoder = SoftRegimeEncoder.fit(
        combined,
        cutoff=datetime(2025, 10, 1, tzinfo=timezone.utc),
        n_regimes=8,
    )
    regimes = encoder.transform(combined)
    elapsed = perf_counter() - started
    output = Path("data/processed/latent/gmm_regimes_q4.parquet")
    model_output = Path("artefacts/latent/gmm_regimes_q4.pkl")
    output.parent.mkdir(parents=True, exist_ok=True)
    regimes.write_parquet(output)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    model_output.write_bytes(pickle.dumps(encoder))
    metadata = {
        "method": "8-component diagonal Gaussian mixture over combined latent state",
        "fit_period": "2025-01-01 through 2025-09-30",
        "holdout_targets_used": False,
        "rows": regimes.height,
        "fit_and_transform_seconds": elapsed,
        "regime_counts": regimes.group_by("regime").len().sort("regime").to_dicts(),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
