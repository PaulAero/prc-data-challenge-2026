"""Sanity checks for data/processed/latent_set{tag}/final.parquet.

Usage: uv run python scripts/latent_set/check_final.py [--tag ""]

Coverage of ranking departures, NaN counts, columns equal to the fold files, and the distribution
of nn_pred / nn_pfill per airport on ranking vs 2025 out-of-fold (and vs the jan/jul months of 2025).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "8")

import polars as pl

FEAT = Path(__import__("os").environ.get("PRC_FEATURES_DIR", "data/processed/features"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    d = Path(f"data/processed/latent_set{args.tag}")
    fin = pl.read_parquet(d / "final.parquet")
    ref_cols = pl.read_parquet(d / "jan.parquet").columns if (d / "jan.parquet").exists() else None
    rank = pl.read_parquet(FEAT / "ranking.parquet").select("MVT_ID_mvt", "airport", "event_time")
    train = pl.read_parquet(FEAT / "train.parquet").select("MVT_ID_mvt", "airport", "event_time")
    out = {"rows": fin.height, "columns_match_folds": ref_cols == fin.columns if ref_cols else None,
           "duplicate_ids": fin.height - fin["MVT_ID_mvt"].n_unique(),
           "nan_or_null": {c: int(fin[c].null_count() + fin[c].is_nan().sum()) for c in fin.columns if c != "MVT_ID_mvt"}}
    r = rank.join(fin, on="MVT_ID_mvt", how="left")
    t = train.join(fin, on="MVT_ID_mvt", how="left")
    out["ranking_departures"] = rank.height
    out["ranking_missing_nn_pred"] = int(r["nn_pred"].null_count())
    out["train_missing_nn_pred"] = int(t["nn_pred"].null_count())
    same_months = t.filter(pl.col("event_time").dt.month().is_in([1, 7]))

    def summ(f: pl.DataFrame, label: str) -> pl.DataFrame:
        return f.group_by("airport").agg(
            pl.len().alias(f"n_{label}"), pl.col("nn_pred").mean().round(1).alias(f"mean_{label}"),
            pl.col("nn_pred").quantile(0.1).round(0).alias(f"p10_{label}"),
            pl.col("nn_pred").quantile(0.9).round(0).alias(f"p90_{label}"),
            pl.col("nn_pfill").mean().round(4).alias(f"pfill_{label}"))

    tab = (summ(r, "rank").join(summ(t, "oof25"), on="airport").join(summ(same_months, "janjul25"), on="airport")
           .sort("airport"))
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=250):
        print(tab)
    print(json.dumps(out, indent=1))
    (d / "final_check.json").write_text(json.dumps(out | {"per_airport": tab.to_dicts()}, indent=1))


if __name__ == "__main__":
    main()
