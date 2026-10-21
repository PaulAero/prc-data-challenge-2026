"""Run one experiment config under VAL-MONTH-V2.

Usage: uv run python scripts/protocol/run.py configs/<id>.yaml

Config keys:
  id: experiment id (output dir experiments/<id>/)
  model: lgbm | two_part
  blocks: feature blocks from the cache (base, seq, stand)
  extra_features: optional list of parquet paths joined on MVT_ID_mvt; "{fold}" is replaced by the
    fold name so fold-dependent (label-derived) features can be supplied per fold.
  drop: optional list of columns to remove
  folds: default [jan, jul, q4]
  rounds, seed, params: model settings
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "12")

import polars as pl
import yaml

from prc.features.flight import block_columns
from prc.models.gbdt import build_model
from prc.validation.protocol import FOLDS, PRIMARY, score

FEAT = Path(__import__("os").environ.get("PRC_FEATURES_DIR", "data/processed/features"))


def load_fold_frame(cfg: dict, fold: str) -> tuple[pl.DataFrame, list[str]]:
    frame = pl.read_parquet(FEAT / "train.parquet").join(pl.read_parquet(FEAT / "labels.parquet"), on="MVT_ID_mvt")
    blocks = block_columns(frame)
    cols = [c for b in cfg.get("blocks", ["base"]) for c in blocks[b]]
    for path in cfg.get("extra_features", []):
        extra = pl.read_parquet(path.format(fold=fold))
        frame = frame.join(extra, on="MVT_ID_mvt", how="left")
        cols += [c for c in extra.columns if c != "MVT_ID_mvt"]
    cols = [c for c in dict.fromkeys(cols) if c not in set(cfg.get("drop", []))]
    return frame, cols


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    out = Path("experiments") / cfg["id"]
    out.mkdir(parents=True, exist_ok=True)
    results, preds = {}, []
    for name in cfg.get("folds", ["jan", "jul", "q4"]):
        fold = FOLDS[name]
        t0 = time.time()
        frame, cols = load_fold_frame(cfg, name)
        train, hold = frame.filter(fold.train()), frame.filter(fold.holdout())
        model = build_model(cfg, cols)
        model.fit(train)
        hold = hold.with_columns(pl.Series("pred", model.predict(hold)), pl.lit(name).alias("fold"))
        if os.environ.get("PRC_SAVE_PARTS"):
            from prc.models.gbdt import mixture_parts
            pp, nf = mixture_parts(model, hold)
            hold = hold.with_columns(pl.Series("part_p", pp), pl.Series("part_nonfill", nf))
        if hasattr(model, "predict_parts"):
            p_fill, reg = model.predict_parts(hold)
            hold = hold.with_columns(pl.Series("p_fill", p_fill), pl.Series("pred_reg", reg))
        results[name] = score(hold) | {"seconds": round(time.time() - t0, 1), "n_features": len(cols),
                                       "top_gain": model.importance(15)}
        keep = ["MVT_ID_mvt", "fold", "airport", "event_time", "target", "no_nm", "pred"]
        keep += [c for c in ("p_fill", "pred_reg", "fill", "mvt_vs_schedule_sec", "part_p", "part_nonfill") if c in hold.columns]
        preds.append(hold.select(keep))
        print(f"{cfg['id']} {name}: all {results[name]['all']:.2f} core {results[name]['core']:.2f} clean {results[name]['clean']:.2f} "
              f"({results[name]['seconds']:.0f}s)", flush=True)
    allp = pl.concat(preds)
    prim = allp.filter(pl.col("fold").is_in(PRIMARY))
    if prim.height:
        results["primary_jan_jul"] = score(prim)
        print(f"{cfg['id']} PRIMARY jan+jul: all {results['primary_jan_jul']['all']:.2f} core {results['primary_jan_jul']['core']:.2f} "
              f"clean {results['primary_jan_jul']['clean']:.2f}", flush=True)
    (out / "metrics.json").write_text(json.dumps({"config": cfg, "protocol": "VAL-MONTH-V2", "results": results}, indent=1))
    allp.write_parquet(out / "predictions.parquet")


if __name__ == "__main__":
    main(sys.argv[1])
