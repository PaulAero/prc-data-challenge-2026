"""Network-alone comparison of two latent sets on long taxis and snow days (EXP-029).

Usage: uv run python scripts/latent_set/long_taxi_check.py --a _noarr_wx_avg3 --b _noarr_wx_clip7200_avg3

Per holdout fold: clean RMSE of nn_pred, split by target band (<= 3,600 s, 3,600-7,200 s), bias in
the long band, and frozen-precipitation flights; on feb, the LTFM snow days' daily means.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "8")

import numpy as np
import polars as pl

from prc.validation.protocol import FOLDS, clean_mask

FEAT = Path(__import__("os").environ.get("PRC_FEATURES_DIR", "data/processed/features"))


def rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--folds", nargs="+", default=["jan", "jul", "feb"])
    args = ap.parse_args()
    base = (pl.read_parquet(FEAT / "train.parquet").select("MVT_ID_mvt", "event_time", "airport", "no_nm",
                                                             "wx_frozen_6h")
            .join(pl.read_parquet(FEAT / "labels.parquet"), on="MVT_ID_mvt"))
    for name in args.folds:
        f = base.filter(FOLDS[name].holdout())
        for tag, col in ((args.a, "a"), (args.b, "b")):
            f = f.join(pl.read_parquet(f"data/processed/latent_set{tag}/{name}.parquet").select(
                "MVT_ID_mvt", pl.col("nn_pred").alias(col)), on="MVT_ID_mvt")
        c = f.filter(pl.Series(clean_mask(f)))
        y = c["target"].to_numpy()
        ea, eb = c["a"].to_numpy() - y, c["b"].to_numpy() - y
        long_, frz = y > 3600, (c["wx_frozen_6h"] > 0).to_numpy()
        print(f"== {name}: clean {rmse(ea):.2f} -> {rmse(eb):.2f} | <=3600 s {rmse(ea[~long_]):.2f} -> "
              f"{rmse(eb[~long_]):.2f} | 3600-7200 s n={long_.sum()} {rmse(ea[long_]):.1f} -> {rmse(eb[long_]):.1f} "
              f"(bias {ea[long_].mean():.0f} -> {eb[long_].mean():.0f}) | frozen n={frz.sum()} "
              f"{rmse(ea[frz]):.1f} -> {rmse(eb[frz]):.1f} | max pred {c['a'].max():.0f} -> {c['b'].max():.0f}")
        if name == "feb":
            d = (f.filter((pl.col("airport") == "LTFM") & (pl.col("fill") == 0))
                 .group_by(pl.col("event_time").dt.date().alias("day"))
                 .agg(pl.col("target").mean().round(0).alias("truth"), pl.col("a").mean().round(0),
                      pl.col("b").mean().round(0)).sort("truth", descending=True).head(6))
            print(d)


if __name__ == "__main__":
    main()
