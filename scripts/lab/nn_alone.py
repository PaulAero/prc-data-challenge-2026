"""Network-alone accuracy of latent fold files (EXP-047): ordinary rows (NM-matched non-fill <= 2 h),
jan and jul holdouts, plus a busy/quiet split by take-offs in the previous 30 min.

Usage: uv run python scripts/lab/nn_alone.py <latent tag> [<latent tag> ...]   e.g. v2_s20260926 aux02_s20260926
"""
import sys

import numpy as np
import polars as pl

from prc.validation.protocol import FOLDS

tr = pl.read_parquet("data/processed/features/train.parquet",
                     columns=["MVT_ID_mvt", "event_time", "no_nm", "airport", "dep_all_prev1800"]).join(
    pl.read_parquet("data/processed/features/labels.parquet"), on="MVT_ID_mvt")
for tag in sys.argv[1:]:
    res = {}
    allh = []
    for fold in ("jan", "jul"):
        h = tr.filter(FOLDS[fold].holdout()).join(
            pl.read_parquet(f"data/processed/latent_set_{tag}/{fold}.parquet", columns=["MVT_ID_mvt", "nn_pred"]),
            on="MVT_ID_mvt")
        h = h.filter((pl.col("no_nm") == 0) & (pl.col("fill") == 0) & (pl.col("target") <= 7200))
        res[fold] = float(np.sqrt(((h["nn_pred"] - h["target"]) ** 2).mean()))
        allh.append(h)
    h = pl.concat(allh)
    busy = h["dep_all_prev1800"] >= h["dep_all_prev1800"].median()
    e2 = ((h["nn_pred"] - h["target"]) ** 2).to_numpy()
    print(f"{tag:22s} jan {res['jan']:.2f} jul {res['jul']:.2f} pooled {np.sqrt(e2.mean()):.2f} "
          f"busy {np.sqrt(e2[busy.to_numpy()].mean()):.2f} quiet {np.sqrt(e2[~busy.to_numpy()].mean()):.2f}")
