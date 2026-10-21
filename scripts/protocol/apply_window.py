"""Apply the NM LOBT window (EXP-033) post hoc to saved VAL-MONTH-V2 predictions of an experiment.

Usage: uv run python scripts/protocol/apply_window.py <exp-id> <out-id>
Fits NMWindow on each fold's training rows (label-derived offset) and applies it to the holdout rows.
Only the window clip is applied; the fill-possible rule needs the model's parts (built into
TwoPartFill for new runs).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

from prc.models.gbdt import NMWindow
from prc.validation.protocol import FOLDS, PRIMARY, score

FEAT = Path(os.environ.get("PRC_FEATURES_DIR", "data/processed/features"))


def main(exp: str, out: str) -> None:
    feats = pl.read_parquet(FEAT / "train.parquet").select("MVT_ID_mvt", "event_time", "mvt_vs_lobt_sec",
                                                           "mvt_vs_schedule_sec").join(
        pl.read_parquet(FEAT / "labels.parquet"), on="MVT_ID_mvt")
    pred = pl.read_parquet(f"experiments/{exp}/predictions.parquet")
    parts = []
    for fold in pred["fold"].unique().sort().to_list():
        p = pred.filter(pl.col("fold") == fold).join(feats.drop("event_time", "target"), on="MVT_ID_mvt")
        train = feats.filter(FOLDS[fold].train()).join(
            pl.read_parquet(FEAT / "train.parquet").select("MVT_ID_mvt", "airport", "no_nm"), on="MVT_ID_mvt")
        w = NMWindow().fit(train)
        p = p.with_columns(pl.Series("pred", w.apply(p, p["pred"].to_numpy())))
        parts.append(p.select(pred.columns))
    res = pl.concat(parts)
    d = Path("experiments") / out
    d.mkdir(parents=True, exist_ok=True)
    res.write_parquet(d / "predictions.parquet")
    results = {f: score(res.filter(pl.col("fold") == f)) for f in res["fold"].unique().sort().to_list()}
    prim = res.filter(pl.col("fold").is_in(PRIMARY))
    if prim.height:
        results["primary_jan_jul"] = score(prim)
    (d / "metrics.json").write_text(json.dumps({"window_applied_to": exp, "results": results}, indent=1))
    for k, v in results.items():
        print(f"{out} {k}: all {v['all']:.2f} core {v['core']:.2f} clean {v['clean']:.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
