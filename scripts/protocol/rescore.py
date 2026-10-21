"""Rescore saved VAL-MONTH-V2 predictions and print a comparison table.

Usage: uv run python scripts/protocol/rescore.py v2-base v2-seqstand v2-twopart ...
"""

from __future__ import annotations

import sys

import polars as pl

from prc.validation.protocol import PRIMARY, score


def main(ids: list[str]) -> None:
    print(f"{'experiment':24s} {'fold':8s} {'all':>8s} {'core':>8s} {'clean':>8s}")
    for i in ids:
        p = pl.read_parquet(f"experiments/{i}/predictions.parquet")
        for fold, f in [(f, p.filter(pl.col("fold") == f)) for f in ["jan", "jul", "q4"]] + [
                ("jan+jul", p.filter(pl.col("fold").is_in(PRIMARY)))]:
            if f.height:
                s = score(f)
                print(f"{i:24s} {fold:8s} {s['all']:8.2f} {s['core']:8.2f} {s['clean']:8.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
