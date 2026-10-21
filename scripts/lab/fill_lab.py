"""Fast lab for the two-part fill mixture (EXP-036): regression target handling and stake-weighted
fill classifiers. One LightGBM seed per arm, jan + jul folds of VAL-MONTH-V2, final-lgb blocks.

The mixture's squared error is, to first order, sum_i (ms_i - reg_i)^2 (p_i - fill_i)^2 plus the
non-fill regression error: a Brier score weighted by the stake (ms - reg)^2. An unweighted
classifier spends its capacity on on-time flights where the fill label costs nothing.

Usage: uv run python scripts/lab/fill_lab.py <out_dir>
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "12")
import lightgbm as lgb
import numpy as np
import polars as pl

from prc.features.flight import block_columns
from prc.models.gbdt import DEFAULT_PARAMS, D24, Frame, NMWindow, proxy_offset
from prc.validation.protocol import FOLDS

FEAT = Path("data/processed/features")
LATENT = "data/processed/latent_set_noarr_wx_avg3/{fold}.parquet"
BLOCKS = ["base", "airline", "seq", "stand", "schedrel", "wx", "geo", "pxdev"]
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/lab-fill")
OUT.mkdir(parents=True, exist_ok=True)
THREADS = int(os.environ.get("LAB_THREADS", "12"))


def load(fold: str):
    f = pl.read_parquet(FEAT / "train.parquet").join(pl.read_parquet(FEAT / "labels.parquet"), on="MVT_ID_mvt")
    b = block_columns(f)
    cols = [c for k in BLOCKS for c in b[k]]
    ex = pl.read_parquet(LATENT.format(fold=fold))
    f = f.join(ex, on="MVT_ID_mvt", how="left")
    cols += [c for c in ex.columns if c != "MVT_ID_mvt"]
    return f, list(dict.fromkeys(cols))


def reg_fit_predict(train, hold, cols, target_clip=None, drop_over=None, seed=20260926, rounds=1500):
    tr = train.filter(pl.col("fill") == 0)
    if drop_over is not None:
        tr = tr.filter(pl.col("target") <= drop_over)
    y = tr["target"].to_numpy()
    if target_clip is not None:
        y = np.minimum(y, target_clip)
    fr = Frame(cols)
    X = fr.fit_transform(tr)
    params = dict(DEFAULT_PARAMS, seed=seed, objective="regression", num_threads=THREADS)
    m = lgb.train(params, lgb.Dataset(X, y, init_score=proxy_offset(tr)), rounds)
    out = {}
    for name, fr_ in (("hold", hold), ("train", train)):
        out[name] = m.predict(fr.transform(fr_)) + proxy_offset(fr_)
    return out


def clf_fit_predict(train, hold, cols, weight=None, seed=20260926, rounds=800, **over):
    fr = Frame(cols)
    X = fr.fit_transform(train)
    params = dict(DEFAULT_PARAMS, seed=seed, objective="binary", num_threads=THREADS)
    params.update(over)
    m = lgb.train(params, lgb.Dataset(X, train["fill"].to_numpy(), weight=weight), rounds)
    return m.predict(fr.transform(hold))


def main():
    arms = {}
    rows = []
    for fold in ("jan", "jul"):
        t0 = time.time()
        frame, cols = load(fold)
        F = FOLDS[fold]
        train, hold = frame.filter(F.train()), frame.filter(F.holdout())
        win = NMWindow().fit(train)
        d24 = D24().fit(train)
        q24 = d24.prob(hold)
        possible = win.fill_possible(hold)
        ms_h = hold["mvt_vs_schedule_sec"].fill_null(0).to_numpy().astype(float)
        ms_t = train["mvt_vs_schedule_sec"].fill_null(0).to_numpy().astype(float)
        # regression arms
        regs = {
            "raw": reg_fit_predict(train, hold, cols)["hold"],
            "clip3h": reg_fit_predict(train, hold, cols, target_clip=10800.0)["hold"],
            "drop3h": reg_fit_predict(train, hold, cols, drop_over=10800.0)["hold"],
        }
        print(fold, "regs done", round(time.time() - t0), flush=True)
        # stake proxy on training rows: the latent's out-of-fold non-fill estimate
        nn_t = train["nn_pred"].fill_null(900.0).to_numpy().astype(float)
        stake = (ms_t - nn_t) ** 2
        clfs = {
            "base": clf_fit_predict(train, hold, cols),
            "reg63": clf_fit_predict(train, hold, cols, num_leaves=63, min_data_in_leaf=200, rounds=500),
            "w_abs": clf_fit_predict(train, hold, cols, weight=np.sqrt(stake) / np.sqrt(stake).mean()),
            "w_sq_cap": clf_fit_predict(train, hold, cols, weight=np.minimum(stake, 7200.0**2) / np.minimum(stake, 7200.0**2).mean()),
            "w_sq": clf_fit_predict(train, hold, cols, weight=np.minimum(stake, 86400.0**2) / np.minimum(stake, 86400.0**2).mean(),
                                    num_leaves=63, min_data_in_leaf=200, rounds=500),
        }
        print(fold, "clfs done", round(time.time() - t0), flush=True)
        keep = hold.select("MVT_ID_mvt", "airport", "event_time", "target", "fill", "no_nm", "mvt_vs_schedule_sec")
        for rn, reg in regs.items():
            keep = keep.with_columns(pl.Series(f"reg_{rn}", reg))
        for cn, p in clfs.items():
            keep = keep.with_columns(pl.Series(f"p_{cn}", np.where(possible, p, 0.0)))
        keep = keep.with_columns(pl.Series("q24", q24), pl.lit(fold).alias("fold"))
        rows.append(keep)
    allp = pl.concat(rows)
    allp.write_parquet(OUT / "parts.parquet")
    # score every combination
    y = allp["target"].to_numpy(); ms = allp["mvt_vs_schedule_sec"].fill_null(0).to_numpy().astype(float)
    core = y <= 10800; clean = (allp["no_nm"].to_numpy() == 0) & (y <= 7200)
    q = allp["q24"].to_numpy()
    lo_hi = None
    res = {}
    for rn in ("raw", "clip3h", "drop3h"):
        reg = allp[f"reg_{rn}"].to_numpy()
        for cn in ("base", "reg63", "w_abs", "w_sq_cap", "w_sq"):
            p = allp[f"p_{cn}"].to_numpy()
            mix = p * ms + (1 - p) * reg
            mix = q * (86400.0 + reg) + (1 - q) * mix
            se = (mix - y) ** 2
            res[f"{rn}|{cn}"] = dict(all=float(np.sqrt(se.mean())), core=float(np.sqrt(se[core].mean())),
                                    clean=float(np.sqrt(se[clean].mean())))
            print(f"{rn:7s} {cn:9s} all {res[f'{rn}|{cn}']['all']:.2f} core {res[f'{rn}|{cn}']['core']:.2f} clean {res[f'{rn}|{cn}']['clean']:.2f}", flush=True)
    (OUT / "metrics.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
