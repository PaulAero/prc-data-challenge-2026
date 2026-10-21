"""Blend saved VAL-MONTH-V2 predictions of several experiments (simple average or given weights).

Usage: uv run python scripts/protocol/blend.py <out-id> <exp1> <exp2> [--weights 0.5,0.5]
Writes experiments/<out-id>/predictions.parquet and metrics.json; prints the comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from prc.validation.protocol import PRIMARY, score


def main(argv: list[str]) -> None:
    w = None
    if "--weights" in argv:
        i = argv.index("--weights")
        w = [float(x) for x in argv[i + 1].split(",")]
        argv = argv[:i] + argv[i + 2:]
    out, exps = argv[0], argv[1:]
    w = w or [1 / len(exps)] * len(exps)
    base = pl.read_parquet(f"experiments/{exps[0]}/predictions.parquet").select(
        "MVT_ID_mvt", "fold", "airport", "event_time", "target", "no_nm")
    preds = [pl.read_parquet(f"experiments/{e}/predictions.parquet").select("MVT_ID_mvt", pl.col("pred").alias(f"p{i}"))
             for i, e in enumerate(exps)]
    for p in preds:
        base = base.join(p, on="MVT_ID_mvt", how="inner")
    base = base.with_columns(pl.sum_horizontal([pl.col(f"p{i}") * wi for i, wi in enumerate(w)]).alias("pred"))
    res = {f: score(base.filter(pl.col("fold") == f)) for f in base["fold"].unique().sort().to_list()}
    prim = base.filter(pl.col("fold").is_in(PRIMARY))
    if prim.height:
        res["primary_jan_jul"] = score(prim)
    d = Path("experiments") / out
    d.mkdir(parents=True, exist_ok=True)
    base.select("MVT_ID_mvt", "fold", "airport", "event_time", "target", "no_nm", "pred").write_parquet(d / "predictions.parquet")
    (d / "metrics.json").write_text(json.dumps({"blend_of": exps, "weights": w, "results": res}, indent=1))
    for k, v in res.items():
        print(f"{out} {k}: all {v['all']:.2f} core {v['core']:.2f} clean {v['clean']:.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
