"""EXP-052: ADS-B corrector on top of the four-stack ensemble.

Base = the ens4 out-of-fold predictions on January and July 2025 (mean of the eight members, as in
`candidate_ens4`). A LightGBM learns the residual (target - base, target clipped at 3 h) from the
ADS-B surface features plus base, airport, no_nm, schedule offset and hour. Trained on one month,
tested on the other (VAL-MONTH-V2 primary folds). Control: the same corrector without ADS-B
features. Only days with extracted ADS-B points are scored.

Usage: uv run python scripts/adsb/corrector.py
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from prc.validation.protocol import paired_day_bootstrap, score

import os
BASE_SUFFIX = os.environ.get("ADSB_BASE_SUFFIX", "")  # "r": members with the regime features (EXP-054)
MEMBERS = [f"{v}{BASE_SUFFIX}-{m}" for v in ("v10", "v11", "v12", "v15") for m in ("lgb", "cb")]
ADSB = ["adsb_cong_moving", "adsb_cong_still", "adsb_dt", "adsb_cs_ok", "adsb_type_ok", "adsb_still0",
        "adsb_n_stops", "adsb_stop_time", "adsb_lat0", "adsb_lon0", "adsb_gs0", "adsb_seen", "adsb_move0", "adsb_d0_stand", "adsb_leave_stand", "adsb_near_stand_t"]
SHAPE = ["adsb_path_len", "adsb_dist_to_lift", "adsb_mean_gs", "adsb_since_fast", "adsb_since_last_stop"]
BASE = ["base", "ap_code", "no_nm", "mvt_vs_schedule_sec", "hour"]
PARAMS = dict(objective="huber", alpha=900.0, learning_rate=0.02, num_leaves=15, min_child_samples=200,
              feature_fraction=0.8, n_estimators=500, num_threads=12, verbose=-1, seed=0,
              deterministic=True, force_row_wise=True)
MATCHED = pl.col("adsb_seen").is_not_null()
YCLIP = 10800.0


def load() -> pl.DataFrame:
    ps = [pl.read_parquet(f"experiments/{m}/predictions.parquet") for m in MEMBERS]
    base = np.mean([p["pred"].to_numpy() for p in ps], axis=0)
    f = ps[0].drop("pred").with_columns(pl.Series("base", base))
    a_cols = pl.read_parquet_schema("data/processed/adsb_features.parquet")
    a = pl.read_parquet("data/processed/adsb_features.parquet").with_columns(
        pl.col("MVT_ID_mvt").cast(pl.Int64), *[pl.col(c).cast(pl.Float64) for c in ADSB + [c for c in SHAPE if c in a_cols]])
    f = f.join(a, on="MVT_ID_mvt", how="inner")
    return f.sort("MVT_ID_mvt").with_columns(pl.col("airport").cast(pl.Categorical).to_physical().cast(pl.Float64).alias("ap_code"),
                          pl.col("event_time").dt.hour().cast(pl.Float64).alias("hour"),
                          pl.col("no_nm").cast(pl.Float64))


def fit_predict(tr: pl.DataFrame, te: pl.DataFrame, feats: list[str]) -> np.ndarray:
    """Trained on matched departures only; unmatched departures keep the base prediction."""
    tr = tr.filter(MATCHED)
    y = np.minimum(tr["target"].to_numpy().astype(float), YCLIP) - tr["base"].to_numpy()
    m = lgb.LGBMRegressor(**PARAMS).fit(tr.select(feats).to_numpy().astype(float), y)
    corr = m.predict(te.select(feats).to_numpy().astype(float))
    return te["base"].to_numpy() + np.where(te.select(MATCHED.alias("m"))["m"].to_numpy(), corr, 0.0)


def main() -> None:
    f = load()
    print("rows with ADS-B day:", f.group_by("fold").len().rows())
    outs = []
    for test in ("jan", "jul"):
        tr, te = f.filter(pl.col("fold") != test), f.filter(pl.col("fold") == test)
        outs.append(te.with_columns(pl.Series("ctrl", fit_predict(tr, te, BASE)),
                                    pl.Series("adsb", fit_predict(tr, te, BASE + ADSB))))
    o = pl.concat(outs)
    res = {}
    for col in ("base", "ctrl", "adsb"):
        s = score(o, col)
        res[col] = s
        print(f"{col:6s} all {s['all']:.2f} core {s['core']:.2f} clean {s['clean']:.2f}")
    print({a: {c: round(res[c]['per_airport'][a]['clean'], 1) for c in res} for a in sorted(res['base']['per_airport'])})
    boot = paired_day_bootstrap(o, "ctrl", "adsb")
    print("paired day bootstrap ctrl -> adsb (clean):", boot)
    Path("experiments/adsb-corrector").mkdir(parents=True, exist_ok=True)
    Path("experiments/adsb-corrector/metrics.json").write_text(json.dumps({"results": res, "boot": boot}, indent=1, default=str))
    o.select("MVT_ID_mvt", "fold", "airport", "event_time", "target", "no_nm", "base", "ctrl", "adsb").write_parquet(
        "experiments/adsb-corrector/predictions.parquet")


if __name__ == "__main__":
    main()
