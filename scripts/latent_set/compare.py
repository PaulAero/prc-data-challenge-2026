"""Compare stacked experiments with a reference: all / core / clean, and paired day-block bootstraps.

Usage: uv run python scripts/latent_set/compare.py --ref v2-twopart-nn-fp32 v2-twopart-nn-s1 v2-twopart-nn-s2 ...

Only folds present in every file are used; the pooled numbers are over jan+jul when both exist.
Bootstrap = RMSE(ref) - RMSE(arm) over days; positive means the arm is better.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("POLARS_MAX_THREADS", "8")

import polars as pl

from prc.validation.protocol import PRIMARY, paired_day_bootstrap, score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("arms", nargs="+")
    args = ap.parse_args()
    ref = pl.read_parquet(f"experiments/{args.ref}/predictions.parquet").filter(pl.col("fold").is_in(PRIMARY))
    s = score(ref)
    print(f"{args.ref:32s} all {s['all']:.2f} core {s['core']:.2f} clean {s['clean']:.2f}")
    for arm in args.arms:
        b = pl.read_parquet(f"experiments/{arm}/predictions.parquet").select("MVT_ID_mvt", pl.col("pred").alias("b"))
        f = ref.join(b, on="MVT_ID_mvt", how="inner")
        if f.height != ref.height:
            print(f"{arm}: covers {f.height} of {ref.height} rows, skipped")
            continue
        s = score(f, "b")
        bc = paired_day_bootstrap(f, "pred", "b")
        bk = paired_day_bootstrap(f.filter(pl.col("target") <= 3 * 3600), "pred", "b", clean_only=False)
        print(f"{arm:32s} all {s['all']:.2f} core {s['core']:.2f} clean {s['clean']:.2f} | "
              f"clean {bc['median']:+.2f} [{bc['lo95']:+.2f}, {bc['hi95']:+.2f}] "
              f"core {bk['median']:+.2f} [{bk['lo95']:+.2f}, {bk['hi95']:+.2f}]")


if __name__ == "__main__":
    main()
