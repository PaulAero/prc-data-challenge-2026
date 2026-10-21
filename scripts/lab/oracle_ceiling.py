"""EXP-046: ceiling of perfect neighbour reconstruction (what any latent architecture could reach).

Oracle features use the TRUE off-blocks of all OTHER departures (never the flight's own label):
aircraft still taxiing at my take-off, pushbacks in windows before my take-off, the queue of aircraft
that pushed back >= 10 min before my take-off but took off after me, and neighbours' true taxi times.
Arms (one LightGBM, regression clipped at 3 h, ff 0.5 + Huber 900, jan+jul; metric = ordinary rows,
NM-matched non-fill <= 2 h): ref = blocks + latent; +oracle; oracle without latent; blocks only.
Labels of other flights are not available on ranking: this is a ceiling, not a feature.

Usage: uv run python scripts/lab/oracle_ceiling.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "10")
import lightgbm as lgb
import numpy as np
import polars as pl

from prc.features.flight import block_columns
from prc.models.gbdt import DEFAULT_PARAMS, Frame, proxy_offset
from prc.validation.protocol import FOLDS

FEAT = Path("data/processed/features")
LAT = "data/processed/latent_set_noarr_wx_avg3/{fold}.parquet"
BLOCKS = ["base", "airline", "seq", "stand", "schedrel", "wx", "geo", "pxdev"]
THREADS = int(os.environ.get("LAB_THREADS", "10"))
OUT = Path("experiments/lab-oracle"); OUT.mkdir(parents=True, exist_ok=True)


def oracle_features(f: pl.DataFrame, col: str = "target", prefix: str = "or_") -> pl.DataFrame:
    """f: all 2025 departures with event_time (take-off), airport, runway and `col` (true taxi, or an
    estimate of it for the 'estimated oracle' arm, EXP-046 follow-up)."""
    out = []
    for (ap,), g in f.group_by(["airport"]):
        g = g.sort("event_time")
        t = g["event_time"].dt.epoch("s").to_numpy().astype(np.int64)
        y = np.clip(g[col].to_numpy(), 0, 10800)
        b = t - y.astype(np.int64)                      # true (recorded) off-block of each flight
        rw = g["runway"].fill_null("NA").to_numpy()
        n = len(t)
        bs = np.sort(b); ts = np.sort(t)
        o = {"MVT_ID_mvt": g["MVT_ID_mvt"].to_numpy()}
        # aircraft on the surface at my take-off (pushed back at or before t_i, take-off after t_i), minus self
        o["or_taxiing"] = (np.searchsorted(bs, t, "right") - np.searchsorted(ts, t, "right")).astype(float)
        for w in (300, 600, 1200, 1800):
            o[f"or_push_prev{w}"] = (np.searchsorted(bs, t, "left") - np.searchsorted(bs, t - w, "left")).astype(float)
        # queue: pushed back >= 10 min before my take-off but took off after me
        q = np.zeros(n)
        order_b = np.argsort(b)
        # for each i: count j with b_j <= t_i - 600 and t_j > t_i  ==  #(b_j <= t_i-600) - #(b_j <= t_i-600 and t_j <= t_i)
        # second term: flights that took off before me with b_j <= t_i - 600 -> use sorted-by-t prefix maximum trick
        # simple O(n log n) via cumulative: iterate in take-off order, maintain sorted list of b of earlier flights
        import bisect
        seen = []
        for i in range(n):  # g sorted by t
            thr = t[i] - 600
            before = bisect.bisect_right(seen, thr)        # earlier take-offs that pushed by thr
            total = np.searchsorted(bs, thr, "right")      # all flights that pushed by thr
            q[i] = total - before - (1 if b[i] <= thr else 0)
            bisect.insort(seen, b[i])
        o["or_queue_ahead"] = q
        # neighbours' true taxi times
        ycs = np.concatenate([[0.0], np.cumsum(y)])
        lo = np.searchsorted(t, t - 600, "left"); hi = np.searchsorted(t, t + 600, "right")
        cnt = hi - lo - 1
        o["or_taxi_pm10"] = np.where(cnt > 0, (ycs[hi] - ycs[lo] - y) / np.maximum(cnt, 1), np.nan)
        prev5 = np.full(n, np.nan); prev1 = np.full(n, np.nan); next1 = np.full(n, np.nan)
        or_tx_rw = np.full(n, np.nan)
        for r in np.unique(rw):
            idx = np.where(rw == r)[0]
            yr = y[idx]
            c = np.concatenate([[0.0], np.cumsum(yr)])
            k = np.arange(len(idx))
            lo5 = np.maximum(k - 5, 0)
            prev5[idx] = np.where(k > 0, (c[k] - c[lo5]) / np.maximum(k - lo5, 1), np.nan)
            prev1[idx] = np.concatenate([[np.nan], yr[:-1]])
            next1[idx] = np.concatenate([yr[1:], [np.nan]])
            tr, br = t[idx], np.sort(b[idx]); trs = np.sort(tr)
            or_tx_rw[idx] = np.searchsorted(br, tr, "right") - np.searchsorted(trs, tr, "right")
        o.update(or_taxi_prev5_rw=prev5, or_taxi_prev1_rw=prev1, or_taxi_next1_rw=next1, or_taxiing_rw=or_tx_rw)
        out.append(pl.DataFrame(o).rename(lambda c: c if c == "MVT_ID_mvt" else prefix + c[3:]))
    return pl.concat(out)


def followup():
    """est = oracle features computed from the latent's own estimates (nn_pred) of every flight;
    or_taxi / or_counts = the two oracle families separately, on top of the latent."""
    base = pl.read_parquet(FEAT / "train.parquet").join(pl.read_parquet(FEAT / "labels.parquet"), on="MVT_ID_mvt")
    orc = oracle_features(base.select("MVT_ID_mvt", "event_time", "airport", "runway", "target"))
    taxi_cols = [c for c in orc.columns if c.startswith("or_taxi_")]
    count_cols = [c for c in orc.columns if c not in taxi_cols and c != "MVT_ID_mvt"]
    b = block_columns(base)
    blocks = list(dict.fromkeys(c for k in BLOCKS for c in b[k]))
    res = json.loads((OUT / "metrics.json").read_text()) if (OUT / "metrics.json").exists() else {}
    for arm in ("est", "or_taxi", "or_counts"):
        sse = n = 0.0
        for fold in ("jan", "jul"):
            lat = pl.read_parquet(LAT.format(fold=fold))
            f = base.join(lat, on="MVT_ID_mvt", how="left")
            cols = blocks + [c for c in lat.columns if c != "MVT_ID_mvt"]
            if arm == "est":
                est = oracle_features(f.select("MVT_ID_mvt", "event_time", "airport", "runway",
                                               pl.col("nn_pred").fill_null(900.0).alias("est")), col="est", prefix="es_")
                f = f.join(est, on="MVT_ID_mvt", how="left"); cols += [c for c in est.columns if c != "MVT_ID_mvt"]
            else:
                f = f.join(orc, on="MVT_ID_mvt", how="left"); cols += taxi_cols if arm == "or_taxi" else count_cols
            F = FOLDS[fold]
            tr = f.filter(F.train() & (pl.col("fill") == 0)); ho = f.filter(F.holdout())
            fr = Frame(cols)
            prm = dict(DEFAULT_PARAMS, seed=20260926, objective="huber", alpha=900.0, feature_fraction=0.5,
                       num_threads=THREADS)
            m = lgb.train(prm, lgb.Dataset(fr.fit_transform(tr), np.minimum(tr["target"].to_numpy(), 10800),
                                           init_score=proxy_offset(tr)), 1500)
            pred = m.predict(fr.transform(ho)) + proxy_offset(ho)
            mask = ((ho["no_nm"] == 0) & (ho["fill"] == 0) & (ho["target"] <= 7200)).to_numpy()
            e = pred[mask] - ho["target"].to_numpy()[mask]
            sse += float((e ** 2).sum()); n += mask.sum()
        res[arm] = float(np.sqrt(sse / n))
        print(f"{arm:13s} ordinary RMSE {res[arm]:.2f}", flush=True)
    (OUT / "metrics.json").write_text(json.dumps(res, indent=1))


def main():
    base = pl.read_parquet(FEAT / "train.parquet").join(pl.read_parquet(FEAT / "labels.parquet"), on="MVT_ID_mvt")
    orc = oracle_features(base.select("MVT_ID_mvt", "event_time", "airport", "runway", "target"))
    ocols = [c for c in orc.columns if c != "MVT_ID_mvt"]
    print("oracle features built", orc.shape, flush=True)
    b = block_columns(base)
    blocks = list(dict.fromkeys(c for k in BLOCKS for c in b[k]))
    res = {}
    for arm, use_lat, use_or in (("ref", True, False), ("oracle", True, True), ("oracle_nolat", False, True),
                                 ("blocks", False, False)):
        sse = n = 0.0
        for fold in ("jan", "jul"):
            f = base.join(orc, on="MVT_ID_mvt", how="left")
            cols = list(blocks)
            if use_lat:
                lat = pl.read_parquet(LAT.format(fold=fold)); f = f.join(lat, on="MVT_ID_mvt", how="left")
                cols += [c for c in lat.columns if c != "MVT_ID_mvt"]
            if use_or:
                cols += ocols
            F = FOLDS[fold]
            tr = f.filter(F.train() & (pl.col("fill") == 0)); ho = f.filter(F.holdout())
            fr = Frame(cols)
            prm = dict(DEFAULT_PARAMS, seed=20260926, objective="huber", alpha=900.0, feature_fraction=0.5,
                       num_threads=THREADS)
            m = lgb.train(prm, lgb.Dataset(fr.fit_transform(tr), np.minimum(tr["target"].to_numpy(), 10800),
                                           init_score=proxy_offset(tr)), 1500)
            pred = m.predict(fr.transform(ho)) + proxy_offset(ho)
            mask = ((ho["no_nm"] == 0) & (ho["fill"] == 0) & (ho["target"] <= 7200)).to_numpy()
            e = pred[mask] - ho["target"].to_numpy()[mask]
            sse += float((e ** 2).sum()); n += mask.sum()
            if use_or and fold == "jan":
                imp = sorted(zip(m.feature_importance("gain"), cols), reverse=True)[:10]
                print(arm, "top", [c for _, c in imp], flush=True)
        res[arm] = float(np.sqrt(sse / n))
        print(f"{arm:13s} ordinary RMSE {res[arm]:.2f}", flush=True)
    (OUT / "metrics.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    import sys
    followup() if "followup" in sys.argv else main()
