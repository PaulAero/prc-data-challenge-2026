"""Build the feature cache used by every VAL-MONTH-V2 experiment (train 2025 and ranking 2026).

Usage: uv run python scripts/protocol/build_features.py
Outputs data/processed/features/{train,ranking,labels}.parquet.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "12")

from prc.data import departure_labels, load_ranking, load_training, mask_departures
from prc.features.flight import build_features

OUT = Path(__import__("os").environ.get("PRC_FEATURES_DIR", "data/processed/features"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train = load_training()
    departure_labels(train).write_parquet(OUT / "labels.parquet")
    build_features(mask_departures(train)).write_parquet(OUT / "train.parquet")
    ranking = load_ranking()
    build_features(mask_departures(ranking)).write_parquet(OUT / "ranking.parquet")
    print(f"features built in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
