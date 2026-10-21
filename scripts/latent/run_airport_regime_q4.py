#!/usr/bin/env python3
"""EXP-007: condition soft operating regimes on airport identity."""

from pathlib import Path

from run_regime_q4 import run_experiment


if __name__ == "__main__":
    run_experiment(
        experiment_dir=Path("experiments/exp-007-airport-regime-q4"),
        variant="combined predictive state plus airport-conditioned soft regimes",
        add_airport_regime=True,
        model_path="artefacts/latent/catboost_q4_airport_regimes.cbm",
    )
