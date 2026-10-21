"""EXP-052 final: apply the ADS-B corrector to a ranking candidate.

The corrector (generic: no airport code, no coordinates, no congestion counts; EXP-052) is trained on
the matched departures of January and July 2025, with the ens4 out-of-fold predictions as base, and
applied to the matched 2026 departures outside LIRF (no gain there in validation). Unmatched rows
keep the candidate's value.

Usage: uv run python scripts/adsb/apply_ranking.py <candidate.parquet> <out-name> [--v12] [--cb]
-> data/submissions/candidate_<out-name>.parquet (+ .check.json)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import corrector as C  # noqa: E402

from prc.data import load_movements  # noqa: E402

EXCLUDE = ("LIRF",)
GEN = [c for c in C.BASE if c != "ap_code"] + [c for c in C.ADSB if c not in ("adsb_lat0", "adsb_lon0")
                                                 and not c.startswith("adsb_cong")]
SEEDS = (0, 1, 2)
# v18 (EXP-053): + track-shape features, Huber 150, 31 leaves, 1000 rounds. `--v12` rebuilds v12.
FEATS_V18 = GEN + C.SHAPE
PARAMS_V18 = dict(C.PARAMS, num_leaves=31, min_child_samples=100, n_estimators=1000, alpha=150.0)


def main(cand: str, out: str, v12: bool = False, with_cb: bool = False) -> None:
    """with_cb (EXP-053, freeze bundle): average the LightGBM bag with a CatBoost corrector."""
    feats, params = (GEN, C.PARAMS) if v12 else (FEATS_V18, PARAMS_V18)
    tr = C.load().filter(C.MATCHED)
    y = np.minimum(tr["target"].to_numpy().astype(float), C.YCLIP) - tr["base"].to_numpy()
    X = tr.select(feats).to_numpy().astype(float)
    models = [lgb.LGBMRegressor(**{**params, "seed": s}).fit(X, y) for s in SEEDS]
    cb = None
    if with_cb:
        from catboost import CatBoostRegressor
        cb = CatBoostRegressor(loss_function="Huber:delta=150", iterations=1500, learning_rate=0.05, depth=7,
                               rsm=0.5, thread_count=12, verbose=0, random_seed=0).fit(X, y)

    sub = pl.read_parquet(cand)
    rk = load_movements("data/raw/ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP").select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("ADEP_mvt").alias("airport"),
        pl.col("FLIGHT_ID_mvt").is_null().cast(pl.Float64).alias("no_nm"),
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("mvt_vs_schedule_sec"),
        pl.col("MVT_TIME_UTC_mvt").dt.hour().cast(pl.Float64).alias("hour"))
    a = pl.read_parquet("data/processed/adsb_features.parquet").with_columns(
        pl.col("MVT_ID_mvt").cast(pl.Int64), *[pl.col(c).cast(pl.Float64) for c in C.ADSB + C.SHAPE])
    x = sub.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64).alias("_id")).join(
        rk.rename({"MVT_ID_mvt": "_id"}), on="_id", how="left").join(
        a.rename({"MVT_ID_mvt": "_id"}), on="_id", how="left").with_columns(
        pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("base"))
    assert x.height == sub.height and x["_id"].equals(sub["MVT_ID_mvt"].cast(pl.Int64))
    use = (x["adsb_seen"].is_not_null() & ~x["airport"].is_in(EXCLUDE)).to_numpy()
    Xr = x.select(feats).to_numpy().astype(float)
    corr = np.mean([m.predict(Xr) for m in models], axis=0)
    if cb is not None:
        corr = (corr + cb.predict(Xr)) / 2
    pred = x["base"].to_numpy() + np.where(use, corr, 0.0)
    pred = np.rint(np.maximum(pred, 0.0)).astype(np.int32)
    res = sub.with_columns(pl.Series("TAXITIME_SEC_mvt", pred))
    path = Path("data/submissions") / f"candidate_{out}.parquet"
    res.write_parquet(path)
    d = pred - x["base"].to_numpy()
    by_ap = x.with_columns(pl.Series("d", d), pl.Series("use", use)).group_by("airport").agg(
        pl.col("use").mean().alias("share_corrected"), pl.col("d").filter(pl.col("use")).mean().alias("mean_change"),
        (pl.col("d").filter(pl.col("use")) ** 2).mean().sqrt().alias("rms_change")).sort("airport")
    check = {"file": str(path), "base": cand, "rows": res.height, "nulls": int(res["TAXITIME_SEC_mvt"].null_count()),
             "corrected": int(use.sum()), "rms_change_all": float(np.sqrt((d ** 2).mean())),
             "per_airport": by_ap.to_dicts()}
    Path(str(path).replace(".parquet", ".check.json")).write_text(json.dumps(check, indent=1, default=float))
    print(json.dumps(check, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--v12" in sys.argv[3:], "--cb" in sys.argv[3:])
