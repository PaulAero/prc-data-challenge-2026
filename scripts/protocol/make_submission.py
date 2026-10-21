"""Train a config on all of 2025 and predict the ranking file; write a candidate submission + check report.

Usage: uv run python scripts/protocol/make_submission.py configs/<id>.yaml
Writes data/submissions/candidate_<id>.parquet and data/submissions/candidate_<id>.check.json.
Rename it to <team-name>_v<n>.parquet to upload it.

Supported configs: models lgbm / two_part, blocks from the feature cache, and the round-1 two-stage
block (`extra_features` naming data/processed/two_stage/{fold}.parquet), rebuilt here with stage 1
fitted on all 2025 training departures (out-of-fold on 2025, full model on 2026).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "12")

import numpy as np
import polars as pl
import yaml

from prc.features.flight import block_columns
from prc.features.two_stage import neighbour_features, stage1_estimates
from prc.models.gbdt import build_model

FEAT = Path(__import__("os").environ.get("PRC_FEATURES_DIR", "data/processed/features"))
OUT = Path("data/submissions")
TEMPLATE = Path("data/raw/submitting.parquet")


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    t0 = time.time()
    train = pl.read_parquet(FEAT / "train.parquet").join(pl.read_parquet(FEAT / "labels.parquet"), on="MVT_ID_mvt")
    rank = pl.read_parquet(FEAT / "ranking.parquet").with_columns(
        pl.lit(None, dtype=pl.Float64).alias("target"), pl.lit(None, dtype=pl.Int8).alias("fill"))
    b = block_columns(train)
    cols = [c for k in cfg.get("blocks", ["base"]) for c in b[k]]
    extras = cfg.get("extra_features", [])
    for e in [e for e in extras if "two_stage" not in e]:
        # fold-dependent learned features: "{fold}" -> "final" (out-of-fold on 2025, full models on 2026)
        extra = pl.read_parquet(e.format(fold="final"))
        missing = rank.join(extra, on="MVT_ID_mvt", how="anti").height
        if missing:
            raise ValueError(f"{e}: {missing} ranking departures have no values")
        train = train.join(extra, on="MVT_ID_mvt", how="left")
        rank = rank.join(extra, on="MVT_ID_mvt", how="left")
        cols += [c for c in extra.columns if c != "MVT_ID_mvt"]
    if any("two_stage" in e for e in extras):
        if any("two_stage" in e and e != "data/processed/two_stage/{fold}.parquet" for e in extras):
            raise ValueError("only the round-1 two-stage block is supported for submissions")
        both = pl.concat([train.with_columns(pl.lit(True).alias("_is_train")),
                          rank.with_columns(pl.lit(False).alias("_is_train"))], how="vertical_relaxed")
        est = stage1_estimates(both, pl.col("_is_train"), b["base"] + b["seq"] + b["stand"])
        nb = neighbour_features(both, est)  # train/ranking boundary (31 Dec -> 1 Jan) is real traffic
        train = train.join(nb, on="MVT_ID_mvt", how="left")
        rank = rank.join(nb, on="MVT_ID_mvt", how="left")
        cols += [c for c in nb.columns if c != "MVT_ID_mvt"]
    cols = [c for c in dict.fromkeys(cols) if c not in set(cfg.get("drop", []))]
    model = build_model(cfg, cols)
    model.fit(train)
    pred = np.clip(model.predict(rank), 0, None)
    if os.environ.get("PRC_SAVE_PARTS"):
        from prc.models.gbdt import mixture_parts
        pp, nf = mixture_parts(model, rank)
        pl.DataFrame({"MVT_ID_mvt": rank["MVT_ID_mvt"].cast(pl.Int64), "part_p": pp, "part_nonfill": nf}).write_parquet(
            OUT / f"parts_{cfg['id']}.parquet")
    # LEBL/LEMD targets are capped near 3,600 s in 2025
    capped = rank["airport"].is_in(["LEBL", "LEMD"]).to_numpy()
    pred = np.where(capped, np.minimum(pred, 3600), pred)
    sub = pl.DataFrame({"MVT_ID_mvt": rank["MVT_ID_mvt"].cast(pl.Float64),
                        "TAXITIME_SEC_mvt": np.rint(pred).astype(np.int32)})
    tmpl = pl.read_parquet(TEMPLATE)
    sub = tmpl.select("MVT_ID_mvt").join(sub, on="MVT_ID_mvt", how="left")  # template order
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"candidate_{cfg['id']}.parquet"
    sub.write_parquet(path)
    per_ap = (rank.select("airport").with_columns(pl.Series("pred", pred))
              .group_by("airport").agg(pl.len().alias("n"), pl.col("pred").mean().alias("pred_mean"),
                                       pl.col("pred").quantile(0.99).alias("pred_p99"), pl.col("pred").max().alias("pred_max"))
              .join(train.group_by("airport").agg(pl.col("target").mean().alias("train_target_mean")), on="airport")
              .sort("airport"))
    check = {
        "config": cfg, "file": str(path), "rows": sub.height, "template_rows": tmpl.height,
        "ids_match_template": bool(sub.height == tmpl.height and sub["TAXITIME_SEC_mvt"].null_count() == 0),
        "nulls": int(sub["TAXITIME_SEC_mvt"].null_count()), "dtype": str(sub.schema),
        "pred_min": int(sub["TAXITIME_SEC_mvt"].min()), "pred_max": int(sub["TAXITIME_SEC_mvt"].max()),
        "per_airport": per_ap.to_dicts(), "seconds": round(time.time() - t0, 1),
        "local_validation": json.loads((Path("experiments") / cfg["id"] / "metrics.json").read_text())["results"].get("primary_jan_jul", {}).get("all")
        if (Path("experiments") / cfg["id"] / "metrics.json").exists() else None,
    }
    (OUT / f"candidate_{cfg['id']}.check.json").write_text(json.dumps(check, indent=1, default=str))
    print(json.dumps({k: v for k, v in check.items() if k not in ("per_airport", "config")}, indent=1, default=str))
    print(per_ap)


if __name__ == "__main__":
    main(sys.argv[1])
