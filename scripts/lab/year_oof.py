"""Full-year out-of-fold parts of the two-part model (EXP-038).

Six two-month holdouts over 2025 (3-day buffers), features = final-lgb blocks + the final latent file
(out-of-fold for every 2025 row) + optional extra parquet files. For each holdout: the non-fill
regression and one or more fill classifiers. Output: <out>/parts.parquet with, for every 2025
departure, target, fill, ms, reg and p_<variant>. Used to (1) score the tail on 12 months instead of
2, (2) fit calibrators and mixture stackers out of fold.

Usage: uv run python scripts/lab/year_oof.py <out_dir> [--extra path ...] [--yclip 10800]
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "12")
import lightgbm as lgb
import numpy as np
import polars as pl

from prc.features.flight import block_columns
from prc.models.gbdt import DEFAULT_PARAMS, D24, Frame, NMWindow, proxy_offset

FEAT = Path("data/processed/features")
LATENT = "data/processed/latent_set_noarr_wx_avg3/final.parquet"
BLOCKS = ["base", "airline", "seq", "stand", "schedrel", "wx", "geo", "pxdev"]
THREADS = int(os.environ.get("LAB_THREADS", "12"))


def utc(y, m):
    return datetime(y, m, 1, tzinfo=timezone.utc)


BLOCKS_2M = [(utc(2025, m), utc(2025, m + 2) if m + 2 <= 12 else utc(2026, 1)) for m in (1, 3, 5, 7, 9, 11)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--yclip", type=float, default=10800.0)
    ap.add_argument("--rounds", type=int, default=1500)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    f = pl.read_parquet(FEAT / "train.parquet").join(pl.read_parquet(FEAT / "labels.parquet"), on="MVT_ID_mvt")
    b = block_columns(f)
    cols = [c for k in BLOCKS for c in b[k]]
    for e in [LATENT, *a.extra]:
        ex = pl.read_parquet(e)
        f = f.join(ex, on="MVT_ID_mvt", how="left")
        cols += [c for c in ex.columns if c != "MVT_ID_mvt"]
    cols = list(dict.fromkeys(cols))
    rows = []
    for i, (s, e) in enumerate(BLOCKS_2M):
        t0 = time.time()
        buf = timedelta(days=3)
        hold = f.filter((pl.col("event_time") >= s) & (pl.col("event_time") < e))
        train = f.filter((pl.col("event_time") < s - buf) | (pl.col("event_time") >= e + buf))
        win = NMWindow().fit(train)
        d24 = D24().fit(train)
        tr = train.filter(pl.col("fill") == 0)
        fr = Frame(cols)
        prm = dict(DEFAULT_PARAMS, seed=20260926, objective="regression", num_threads=THREADS)
        m = lgb.train(prm, lgb.Dataset(fr.fit_transform(tr), np.minimum(tr["target"].to_numpy(), a.yclip),
                                       init_score=proxy_offset(tr)), a.rounds)
        reg_h = m.predict(fr.transform(hold)) + proxy_offset(hold)
        fc = Frame(cols)
        Xc = fc.fit_transform(train)
        nn_t = train["nn_pred"].fill_null(900.0).to_numpy().astype(float)
        ms_t = train["mvt_vs_schedule_sec"].fill_null(0).to_numpy().astype(float)
        stake = (ms_t - nn_t) ** 2
        variants = {
            "base": (None, {}, 800),
            "wsq": (np.minimum(stake, 86400.0 ** 2), dict(num_leaves=63, min_data_in_leaf=200), 500),
            "wabs": (np.sqrt(stake), {}, 800),
        }
        ph = {}
        for name, (w, over, rounds) in variants.items():
            prm = dict(DEFAULT_PARAMS, seed=20260926, objective="binary", num_threads=THREADS, **over)
            ww = None if w is None else w / w.mean()
            c = lgb.train(prm, lgb.Dataset(Xc, train["fill"].to_numpy(), weight=ww), rounds)
            ph[name] = np.where(win.fill_possible(hold), c.predict(fc.transform(hold)), 0.0)
        _, lo, hi, _ = win.arrays(hold)
        matched = win.arrays(hold)[0]
        keep = hold.select("MVT_ID_mvt", "airport", "event_time", "target", "fill", "no_nm", "mvt_vs_schedule_sec",
                           "mvt_vs_lobt_sec", "local_hour", "nn_pred", "nn_pfill")
        keep = keep.with_columns(pl.Series("reg", reg_h), pl.Series("q24", d24.prob(hold)),
                                 pl.Series("win_lo", np.where(matched, np.maximum(np.nan_to_num(lo, nan=0), 0), np.nan)),
                                 pl.Series("win_hi", np.where(matched, hi, np.nan)), pl.lit(i).alias("block"),
                                 *[pl.Series(f"p_{k}", v) for k, v in ph.items()])
        rows.append(keep)
        print(f"block {i} done {time.time() - t0:.0f}s", flush=True)
        pl.concat(rows).write_parquet(out / "parts.parquet")


if __name__ == "__main__":
    main()
