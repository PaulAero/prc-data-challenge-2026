"""Average the latent heads of several training seeds (EXP-015 seed averaging).

Usage: uv run python scripts/latent_set/seed_average.py --out _noarr_seedavg \
           --tags _noarr_fp32 _noarr_s1 _noarr_s2 --folds jan jul [final]

nn_pred, nn_q10/50/90 and nn_pfill are averaged over the seeds. The z columns come from the first
tag: z spaces of different seeds are not aligned, and EXP-010 showed z adds little beyond nn_pred.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "8")

import polars as pl

HEADS = ["nn_pred", "nn_q10", "nn_q50", "nn_q90", "nn_pfill"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--folds", nargs="+", default=["jan", "jul"])
    args = ap.parse_args()
    out_dir = Path(f"data/processed/latent_set{args.out}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in args.folds:
        frames = [pl.read_parquet(f"data/processed/latent_set{t}/{f}.parquet") for t in args.tags]
        base = frames[0]
        if any(x.height != base.height for x in frames):
            raise ValueError(f"{f}: seeds cover different rows")
        avg = (pl.concat([x.select("MVT_ID_mvt", *HEADS) for x in frames])
               .group_by("MVT_ID_mvt").agg([pl.col(c).mean() for c in HEADS]))
        out = base.drop(HEADS).join(avg, on="MVT_ID_mvt").select(base.columns)
        assert out.height == base.height
        out.write_parquet(out_dir / f"{f}.parquet")
        print(f"{out_dir / f}.parquet: {out.height} rows from {len(frames)} seeds")


if __name__ == "__main__":
    main()
