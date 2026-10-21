"""Stake-weighted calibration of P(fill) on full-year out-of-fold parts (EXP-038).

For each two-month block, a calibrator is fitted on the other five blocks' out-of-fold rows and
applied to the block. The calibrator is a weighted logistic regression of the fill label on
logit(p) and its interactions with segment indicators (LIRF, no-NM) and log stake, with weights
min((ms - reg)^2, cap): the mixture's squared error is a Brier score weighted by that stake.

Scores the mixture p*ms + (1-p)*reg (+ D24 and NM window) on all / core / clean over 12 months,
per two-month block, and on the LIRF no-NM subset.

Usage: uv run python scripts/lab/calib.py <parts.parquet> [p_column ...]
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def design(d: pl.DataFrame, pcol: str) -> np.ndarray:
    lp = logit(d[pcol].to_numpy())
    lirf = (d["airport"].to_numpy() == "LIRF").astype(float)
    nonm = d["no_nm"].to_numpy().astype(float)
    st = np.log1p(np.abs(d["mvt_vs_schedule_sec"].fill_null(0).to_numpy() - d["reg"].to_numpy())) / 10
    seg = [lirf * nonm, lirf * (1 - nonm), (1 - lirf) * nonm]
    cols = [lp, st, lp * st] + seg + [s * lp for s in seg] + [s * st for s in seg]
    return np.column_stack(cols)


def mixture(d: pl.DataFrame, p: np.ndarray) -> np.ndarray:
    ms = d["mvt_vs_schedule_sec"].fill_null(0).to_numpy().astype(float)
    reg = d["reg"].to_numpy(); q = d["q24"].to_numpy()
    mix = p * ms + (1 - p) * reg
    mix = q * (86400.0 + reg) + (1 - q) * mix
    lo, hi = d["win_lo"].to_numpy(), d["win_hi"].to_numpy()
    m = ~np.isnan(lo)
    mix[m] = np.clip(mix[m], lo[m], hi[m])
    return mix


def score(d: pl.DataFrame, pred: np.ndarray) -> dict:
    y = d["target"].to_numpy(); se = (pred - y) ** 2
    core = y <= 10800; clean = (d["no_nm"].to_numpy() == 0) & (y <= 7200)
    ln = (d["airport"].to_numpy() == "LIRF") & (d["no_nm"].to_numpy() == 1)
    N = len(y)
    return dict(all=np.sqrt(se.mean()), core=np.sqrt(se[core].mean()), clean=np.sqrt(se[clean].mean()),
                lirf_nonm_mse=se[ln].sum() / N, core_mse=se[core].sum() / N)


def calibrate(d: pl.DataFrame, pcol: str, cap: float) -> np.ndarray:
    out = np.zeros(d.height)
    blocks = d["block"].to_numpy()
    ms = d["mvt_vs_schedule_sec"].fill_null(0).to_numpy().astype(float)
    w = np.minimum((ms - d["reg"].to_numpy()) ** 2, cap)
    X = design(d, pcol); f = d["fill"].to_numpy()
    possible = d[pcol].to_numpy() > 0
    for b in np.unique(blocks):
        tr = (blocks != b) & possible
        lr = LogisticRegression(C=10.0, max_iter=2000)
        lr.fit(X[tr], f[tr], sample_weight=w[tr] / w[tr].mean())
        te = blocks == b
        out[te] = np.where(possible[te], lr.predict_proba(X[te])[:, 1], 0.0)
    return out


def main():
    d = pl.read_parquet(sys.argv[1])
    pcols = sys.argv[2:] or [c for c in d.columns if c.startswith("p_")]
    fmt = lambda r: " ".join(f"{k} {v:,.1f}" for k, v in r.items())  # noqa: E731
    for pc in pcols:
        p = d[pc].to_numpy()
        print(f"{pc:8s} raw         {fmt(score(d, mixture(d, p)))}")
        for cap in (3600.0 ** 2, 14400.0 ** 2, 86400.0 ** 2):
            pcal = calibrate(d, pc, cap)
            print(f"{pc:8s} cal cap {np.sqrt(cap):>6.0f} {fmt(score(d, mixture(d, pcal)))}")
    f = d["fill"].to_numpy(); ms = d["mvt_vs_schedule_sec"].fill_null(0).to_numpy()
    orc = np.where(f == 1, ms, d["reg"].to_numpy()).astype(float)
    print(f"oracle            {fmt(score(d, mixture(d, f.astype(float))))}")


if __name__ == "__main__":
    main()
