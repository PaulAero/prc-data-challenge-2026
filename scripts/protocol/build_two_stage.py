"""Build fold-dependent two-stage neighbour features for VAL-MONTH-V2.

Usage: uv run python scripts/protocol/build_two_stage.py [--round N] [folds...]
Env STAGE1_BLOCKS (default base,seq,stand) and TWO_STAGE_TAG (file suffix) select the stage-1 inputs.
Round 1 writes data/processed/two_stage/{fold}.parquet (estimates from blocks base+seq+stand).
Round N>1 re-estimates with the round N-1 neighbour features added and writes {fold}_rN.parquet
(prefix nbN), a fixed-point iteration on the estimated off-blocks.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "12")

import polars as pl

from prc.features.flight import block_columns
from prc.features.two_stage import neighbour_features, stage1_estimates
from prc.validation.protocol import FOLDS

FEAT = Path(__import__("os").environ.get("PRC_FEATURES_DIR", "data/processed/features"))
OUT = Path("data/processed/two_stage")


def main(folds: list[str], rnd: int = 1) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pl.read_parquet(FEAT / "train.parquet").join(pl.read_parquet(FEAT / "labels.parquet"), on="MVT_ID_mvt")
    b = block_columns(frame)
    blocks = os.environ.get("STAGE1_BLOCKS", "base,seq,stand").split(",")
    cols = [c for k in blocks for c in b[k]]
    tag = os.environ.get("TWO_STAGE_TAG", "")
    for name in folds:
        t0 = time.time()
        f, c = frame, cols
        if rnd > 1:
            prev = pl.read_parquet(OUT / (f"{name}.parquet" if rnd == 2 else f"{name}_r{rnd - 1}.parquet"))
            f = frame.join(prev, on="MVT_ID_mvt", how="left")
            c = cols + [x for x in prev.columns if x != "MVT_ID_mvt"]
        est = stage1_estimates(f, FOLDS[name].train(), c)
        suffix = ("" if rnd == 1 else f"_r{rnd}") + tag
        neighbour_features(frame, est, prefix="nb" if rnd == 1 else f"nb{rnd}").write_parquet(OUT / f"{name}{suffix}.parquet")
        print(f"two-stage {name}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    rnd = 1
    if args[:1] == ["--round"]:
        rnd, args = int(args[1]), args[2:]
    main(args or ["jan", "jul", "q4"], rnd)
