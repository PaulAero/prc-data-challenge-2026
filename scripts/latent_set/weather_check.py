"""Network-alone comparison of two latent sets on winter weather days (EXP-019).

Usage: uv run python scripts/latent_set/weather_check.py --a _fp32 --b _wx [--folds jan feb]

For each holdout fold: clean RMSE of nn_pred overall, on flights with frozen precipitation in the
previous 6 h (`wx_frozen_6h > 0`) and on the rest, per airport; then the daily mean prediction vs the
daily mean clipped target on the worst days (largest daily mean target) of the fold.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("POLARS_MAX_THREADS", "8")

import numpy as np
import polars as pl

from prc.validation.protocol import FOLDS, clean_mask

FEAT = __import__("pathlib").Path(__import__("os").environ.get("PRC_FEATURES_DIR", "data/processed/features"))


def rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="_fp32")
    ap.add_argument("--b", default="_wx")
    ap.add_argument("--folds", nargs="+", default=["jan", "jul", "feb"])
    args = ap.parse_args()
    base = (pl.read_parquet(FEAT / "train.parquet").select("MVT_ID_mvt", "event_time", "airport", "no_nm",
                                                             "wx_frozen_6h", "wx_snow_12h")
            .join(pl.read_parquet(FEAT / "labels.parquet"), on="MVT_ID_mvt"))
    for name in args.folds:
        f = base.filter(FOLDS[name].holdout())
        for tag, col in ((args.a, "a"), (args.b, "b")):
            x = pl.read_parquet(f"data/processed/latent_set{tag}/{name}.parquet").select(
                "MVT_ID_mvt", pl.col("nn_pred").alias(col))
            f = f.join(x, on="MVT_ID_mvt", how="inner")
        c = f.filter(pl.Series(clean_mask(f)))
        ea, eb = (c["a"] - c["target"]).to_numpy(), (c["b"] - c["target"]).to_numpy()
        frz = (c["wx_frozen_6h"] > 0).to_numpy()
        print(f"== {name}: clean n={c.height}  {args.a} {rmse(ea):.2f}  {args.b} {rmse(eb):.2f} | frozen n={frz.sum()} "
              f"{rmse(ea[frz]):.2f} -> {rmse(eb[frz]):.2f} | other {rmse(ea[~frz]):.2f} -> {rmse(eb[~frz]):.2f}")
        ap_ = c["airport"].to_numpy()
        for a in sorted(set(ap_)):
            m = ap_ == a
            mf = m & frz
            print(f"   {a} {rmse(ea[m]):.1f} -> {rmse(eb[m]):.1f}   frozen n={mf.sum()} "
                  f"{rmse(ea[mf]):.1f} -> {rmse(eb[mf]):.1f}")
        d = (f.filter(pl.col("fill") == 0)
             .group_by("airport", pl.col("event_time").dt.date().alias("day"))
             .agg(pl.len().alias("n"), pl.col("target").clip(0, 3600).mean().round(0).alias("target_mean"),
                  pl.col("a").mean().round(0).alias("pred_a"), pl.col("b").mean().round(0).alias("pred_b"),
                  pl.col("wx_snow_12h").max().round(2).alias("snow12h_max"),
                  pl.col("wx_frozen_6h").mean().round(2).alias("frozen_mean"))
             .sort("target_mean", descending=True).head(10))
        with pl.Config(tbl_rows=12, tbl_width_chars=160):
            print(d)


if __name__ == "__main__":
    main()
