"""Score the joint window network on its own (v2-nn-alone), per VAL-MONTH-V2 fold.

Usage: uv run python scripts/latent_set/score_nn.py [--tag _noctx]

Two readings of the network:
- nn_pred: the regression head, trained on non-fill rows only (the physical taxi-out);
- nn_twopart: p_fill * (take-off - schedule) + (1 - p_fill) * nn_pred, with the network's own fill head.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "8")

import polars as pl

from prc.validation.protocol import FOLDS, PRIMARY, score

FEAT = Path(__import__("os").environ.get("PRC_FEATURES_DIR", "data/processed/features"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    exp = Path(f"experiments/v2-nn-alone{args.tag.replace('_', '-')}")
    base = pl.read_parquet(FEAT / "train.parquet").select("MVT_ID_mvt", "event_time", "airport", "no_nm",
                                                          "mvt_vs_schedule_sec")
    base = base.join(pl.read_parquet(FEAT / "labels.parquet"), on="MVT_ID_mvt")
    results, preds = {}, []
    for name, fold in FOLDS.items():
        path = Path(f"data/processed/latent_set{args.tag}/{name}.parquet")
        if not path.exists():
            continue
        f = base.filter(fold.holdout()).join(pl.read_parquet(path), on="MVT_ID_mvt", how="left")
        assert f["nn_pred"].null_count() == 0, "missing holdout predictions"
        f = f.with_columns(pl.col("nn_pred").alias("pred"),
                           (pl.col("nn_pfill") * pl.col("mvt_vs_schedule_sec").fill_null(0)
                            + (1 - pl.col("nn_pfill")) * pl.col("nn_pred")).alias("pred_twopart"),
                           pl.lit(name).alias("fold"))
        results[name] = {"nn_pred": score(f), "nn_twopart": score(f, "pred_twopart")}
        preds.append(f.select("MVT_ID_mvt", "fold", "airport", "event_time", "target", "no_nm", "fill", "pred",
                              "pred_twopart"))
        r = results[name]
        print(f"{name}: nn_pred all {r['nn_pred']['all']:.2f} clean {r['nn_pred']['clean']:.2f} | "
              f"nn_twopart all {r['nn_twopart']['all']:.2f} clean {r['nn_twopart']['clean']:.2f}")
    allp = pl.concat(preds)
    prim = allp.filter(pl.col("fold").is_in(PRIMARY))
    if prim["fold"].n_unique() == len(PRIMARY):
        results["primary_jan_jul"] = {"nn_pred": score(prim), "nn_twopart": score(prim, "pred_twopart")}
        r = results["primary_jan_jul"]
        print(f"PRIMARY: nn_pred all {r['nn_pred']['all']:.2f} clean {r['nn_pred']['clean']:.2f} | "
              f"nn_twopart all {r['nn_twopart']['all']:.2f} clean {r['nn_twopart']['clean']:.2f}")
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "metrics.json").write_text(json.dumps({"protocol": "VAL-MONTH-V2", "results": results}, indent=1))
    allp.write_parquet(exp / "predictions.parquet")


if __name__ == "__main__":
    main()
