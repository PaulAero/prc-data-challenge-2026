"""Compare a reference run with its +latent run: paired day-block bootstrap and congestion quintiles.

Usage: uv run python scripts/latent_set/analyze.py [--ref v2-twopart] [--new v2-twopart-nn]

Joins the two prediction files on MVT_ID_mvt. Quintiles of `dep_between_aobt_to` and
`rw_prev1800` are cut on the pooled jan+jul clean rows (the congestion proxies from the hand
features; they are inputs, not labels). Writes experiments/<new>/analysis.json.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "8")

import numpy as np
import polars as pl

from prc.validation.protocol import PRIMARY, clean_mask, paired_day_bootstrap

FEAT = Path(__import__("os").environ.get("PRC_FEATURES_DIR", "data/processed/features"))


def rmse(x: pl.Series) -> float:
    return float(np.sqrt((x.to_numpy() ** 2).mean())) if len(x) else float("nan")


def by_quintile(f: pl.DataFrame, col: str) -> list[dict]:
    g = f.drop_nulls(col).filter(pl.col(col).is_not_nan())
    edges = np.quantile(g[col].to_numpy(), [0.2, 0.4, 0.6, 0.8])
    g = g.with_columns(pl.Series("q", np.searchsorted(edges, g[col].to_numpy(), "right")))
    out = []
    for q in range(5):
        s = g.filter(pl.col("q") == q)
        out.append({"quintile": q + 1, "n": s.height, "range": [float(s[col].min()), float(s[col].max())],
                    "rmse_ref": rmse(s["pred_ref"] - s["target"]), "rmse_new": rmse(s["pred_new"] - s["target"]),
                    "mean_target": float(s["target"].mean())})
        out[-1]["delta"] = out[-1]["rmse_new"] - out[-1]["rmse_ref"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="v2-twopart")
    ap.add_argument("--new", default="v2-twopart-nn")
    args = ap.parse_args()
    ref = pl.read_parquet(f"experiments/{args.ref}/predictions.parquet")
    new = pl.read_parquet(f"experiments/{args.new}/predictions.parquet").select(
        "MVT_ID_mvt", pl.col("pred").alias("pred_new"))
    f = ref.rename({"pred": "pred_ref"}).join(new, on="MVT_ID_mvt", how="inner")
    assert f.height == ref.height, "prediction files do not cover the same rows"
    cong = pl.read_parquet(FEAT / "train.parquet").select("MVT_ID_mvt", "dep_between_aobt_to", "rw_prev1800",
                                                          "rw_between_aobt_to")
    f = f.join(cong, on="MVT_ID_mvt", how="left")
    res = {}
    for label, sub in [("primary_jan_jul", f.filter(pl.col("fold").is_in(PRIMARY)))] + [
            (k, f.filter(pl.col("fold") == k)) for k in f["fold"].unique().sort().to_list()]:
        if not sub.height:
            continue
        clean = sub.filter(pl.Series(clean_mask(sub)))
        r = {"boot_clean": paired_day_bootstrap(sub, "pred_ref", "pred_new"),
             "boot_all": paired_day_bootstrap(sub, "pred_ref", "pred_new", clean_only=False),
             "clean_rmse_ref": rmse(clean["pred_ref"] - clean["target"]),
             "clean_rmse_new": rmse(clean["pred_new"] - clean["target"])}
        if label == "primary_jan_jul":
            for col in ("dep_between_aobt_to", "rw_prev1800", "rw_between_aobt_to"):
                r[f"quintiles_{col}"] = by_quintile(clean, col)
        res[label] = r
        b = r["boot_clean"]
        print(f"{label}: clean {r['clean_rmse_ref']:.2f} -> {r['clean_rmse_new']:.2f}; bootstrap ref-new median "
              f"{b['median']:.2f} [{b['lo95']:.2f}, {b['hi95']:.2f}] p(new better) {b['p_b_better']:.3f}")
    for col in ("dep_between_aobt_to", "rw_prev1800", "rw_between_aobt_to"):
        print(col)
        for q in res["primary_jan_jul"][f"quintiles_{col}"]:
            print(f"  Q{q['quintile']} n={q['n']} range={q['range'][0]:.0f}-{q['range'][1]:.0f} "
                  f"ref {q['rmse_ref']:.2f} new {q['rmse_new']:.2f} delta {q['delta']:+.2f}")
    Path(f"experiments/{args.new}/analysis.json").write_text(json.dumps(
        {"ref": args.ref, "new": args.new, "results": res}, indent=1))


if __name__ == "__main__":
    main()
