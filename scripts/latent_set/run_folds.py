"""Joint window transformer, out-of-fold outputs per VAL-MONTH-V2 fold (EXP-010 and variants).

Usage:
  uv run --group timesfm python scripts/latent_set/run_folds.py --folds jan jul q4 [options]

The scheme is `prc.latent_set.oof.run_oof`: training days of the fold (fold.train(), 3-day buffer)
split into k groups of 7-day blocks; the holdout month gets the mean of the k models.

Outputs: data/processed/latent_set{tag}/{fold}.parquet, experiments/v2-nn-alone{tag}/{fold}_meta.json,
checkpoints in artefacts/latent_set/. The adopted EXP-010 run is tag "" (lr 1e-3, 16 epochs,
patience 2); it is never overwritten unless --overwrite is given.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "8")

import numpy as np
import torch

from prc.latent_set.oof import label_arrays, load_2025_events, run_oof
from prc.latent_set.tokens import CONTEXT, build_windows, subset_events
from prc.latent_set.train import Device, TrainConfig
from prc.validation.protocol import FOLDS

torch.set_num_threads(8)


def fold_days(fold, days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    day0 = lambda dt: int(dt.timestamp() // 86400)  # noqa: E731
    b = timedelta(days=fold.buffer_days)
    s, e = day0(fold.start), day0(fold.end)
    sb, eb = day0(fold.start - b), day0(fold.end + b)
    days = np.unique(days)
    return days[(days < sb) | (days >= eb)], days[(days >= s) & (days < e)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=["jan", "jul", "q4"])
    ap.add_argument("--tag", default=None, help="output suffix, e.g. _lr3e4; default derived from flags")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--no-stand-bias", action="store_true")
    ap.add_argument("--no-runway-bias", action="store_true")
    ap.add_argument("--weather", action="store_true", help="hourly weather inputs on every token")
    ap.add_argument("--airline", action="store_true", help="carrier from the flight number as a token field (EXP-045)")
    ap.add_argument("--huber", type=float, default=0.0, help="Huber delta (s) for the mean head; 0 = squared error")
    ap.add_argument("--aux-weight", type=float, default=0.0, help="surface-state auxiliary loss weight (EXP-047)")
    ap.add_argument("--no-arrivals", action="store_true", help="drop arrival tokens")
    ap.add_argument("--context-min", type=int, default=CONTEXT // 60, help="window context each side")
    ap.add_argument("--y-clip", type=float, default=3600.0, help="clip of the non-fill training target (s)")
    ap.add_argument("--max-epochs", type=int, default=16)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--only-group", type=int, default=None, help="fit a single group model (ablation)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    tag = args.tag if args.tag is not None else ("_noctx" if args.no_context else "")
    out_dir = Path(f"data/processed/latent_set{tag}")
    exp_dir = Path(f"experiments/v2-nn-alone{tag.replace('_', '-')}")
    ck_dir = Path("artefacts/latent_set")
    for name in args.folds:
        if (out_dir / f"{name}.parquet").exists() and not args.overwrite:
            raise SystemExit(f"{out_dir / name}.parquet exists; pass --overwrite or a new --tag")
    for p in (out_dir, exp_dir, ck_dir):
        p.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ev, _ = load_2025_events(args.weather, airline=args.airline)
    if args.no_arrivals:
        ev = subset_events(ev, ev.is_dep)
    target, fill = label_arrays(ev)
    win = build_windows(ev, max_tokens=480, context=args.context_min * 60)
    aux = None
    if args.aux_weight > 0:  # EXP-047: surface-state targets from other departures' labels (training only)
        from prc.latent_set.aux import compute_aux
        ta = time.time()
        aux = compute_aux(ev, target)
        print(f"aux targets: {int((~np.isnan(aux[:, 0])).sum())} departures, {time.time() - ta:.0f}s", flush=True)
    D = Device(ev, target, fill, y_clip=args.y_clip, aux=aux)
    print(f"events {ev.n}, windows {win.n}, max tokens {int((win.hi - win.lo).max())}, prep {time.time() - t0:.0f}s",
          flush=True)

    cfg = TrainConfig(context=not args.no_context, stand_bias=not args.no_stand_bias, huber=args.huber, aux_weight=args.aux_weight,
                      runway_bias=not args.no_runway_bias, max_epochs=args.max_epochs, patience=args.patience,
                      lr=args.lr, seed=args.seed)
    dep_idx = np.where(ev.is_dep)[0]
    for name in args.folds:
        tf = time.time()
        train_days, hold_days = fold_days(FOLDS[name], ev.day[dep_idx])
        frame, meta = run_oof(D, win, ev, train_days, np.isin(ev.day, hold_days), cfg, args.k,
                              ck_dir / f"{name}{tag}", log=lambda s: print(f"[{name}] {s}", flush=True),
                              only_group=args.only_group)
        frame.write_parquet(out_dir / f"{name}.parquet")
        (exp_dir / f"{name}_meta.json").write_text(json.dumps(
            {"fold": name, "config": asdict(cfg), "args": vars(args), **meta, "seconds": round(time.time() - tf, 1),
             "rows": frame.height, "n_windows": win.n}, indent=1, default=float))
        print(f"[{name}] done in {time.time() - tf:.0f}s, rows {frame.height}", flush=True)


if __name__ == "__main__":
    main()
