"""February / June 2025 generalisation check (final-phase months), after scripts/run_febjun.sh.

Scores, on whole-month holdouts never used for any tuning: the eight-model ensemble with the regime
features (configs v1xr-*-fj), then the daily calibration to EUROCONTROL's published pre-departure delay
(uniform half shift, fill-aware at LIRF; EXP-056/056b) built from the published 2025 file. The monthly
calibration (EXP-055) is not scored here: on 2025 it would be fitted on the evaluated months themselves.
The ADS-B corrector is not scored either: no tracks were downloaded for February and June.

Writes experiments/febjun-eval/metrics.json and prints the table.
Usage: uv run python scripts/lab/febjun_eval.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import daily_delay as D  # noqa: E402

from prc.data import load_movements  # noqa: E402
from prc.validation.protocol import paired_day_bootstrap, score  # noqa: E402

MEMBERS = [f"{v}r-{m}-fj" for v in ("v10", "v11", "v12", "v15") for m in ("lgb", "cb")]


def main() -> None:
    ps = [pl.read_parquet(f"experiments/{m}/predictions.parquet").sort("MVT_ID_mvt") for m in MEMBERS]
    f = ps[0].drop("pred").with_columns(pl.Series("ens", np.mean([p["pred"].to_numpy() for p in ps], axis=0)))
    lat = pl.concat([pl.read_parquet(f"data/processed/latent_set_v3full_avg3/{k}.parquet").select(
        "MVT_ID_mvt", "nn_pred").with_columns(pl.lit(k).alias("fold")) for k in ("feb", "jun")])
    mv = pl.concat([load_movements(p) for p in sorted(glob.glob("data/raw/training_2025-0[26]-*.parquet"))],
                   how="diagonal_relaxed").filter(pl.col("PHASE_mvt") == "DEP").select(
        pl.col("MVT_ID_mvt").cast(pl.Int64),
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("ts"),
        pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day"))
    x = f.join(lat, on=["MVT_ID_mvt", "fold"], how="left").join(mv, on="MVT_ID_mvt").join(
        D.published(2025), on=["airport", "day"], how="left").with_columns(pl.col("pub").fill_null(0.0))
    sh = D.solve_shifts(x.filter(pl.col("pub") > 0), "ens")
    x = x.with_columns(pl.Series("daily", D.apply(x, "ens", sh, 0.5)))
    L = x.filter((pl.col("airport") == "LIRF") & (pl.col("pub") > 0))
    fc = D.fill_calibrate(L, "ens", 0.75)
    x = x.join(fc, on="MVT_ID_mvt", how="left").with_columns(pl.coalesce("fc", "daily").alias("calib"))
    res = {}
    for col in ("ens", "daily", "calib"):
        res[col] = {"pooled": score(x, col)} | {k: score(x.filter(pl.col("fold") == k), col) for k in ("feb", "jun")}
        r = res[col]
        print(f"{col:6s} pooled all {r['pooled']['all']:.2f} core {r['pooled']['core']:.2f} clean {r['pooled']['clean']:.2f}"
              f" | feb core {r['feb']['core']:.2f} | jun core {r['jun']['core']:.2f}"
              f" | LIRF all {r['pooled']['per_airport']['LIRF']['all']:.1f}")
    boot = paired_day_bootstrap(x, "ens", "calib", clean_only=False)
    print("ens -> calib, core, paired day bootstrap:", boot)
    out = Path("experiments/febjun-eval")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"results": res, "boot_core": boot}, indent=1, default=str))
    print("->", out / "metrics.json")


if __name__ == "__main__":
    main()
