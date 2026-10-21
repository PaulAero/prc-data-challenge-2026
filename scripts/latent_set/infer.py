"""Inference only: rebuild the latent outputs from saved checkpoints, on CPU or GPU.

Usage:
  uv run --group timesfm python scripts/latent_set/infer.py --target final --device cpu
  uv run --group timesfm python scripts/latent_set/infer.py --target jan --device cpu --scope target

--target final: 2025 + ranking events (ranking rows are the target), checkpoints final{tag}_g*.pt.
--target jan|jul|q4: 2025 events, the fold's holdout month is the target, checkpoints {fold}{tag}_g*.pt.
--scope full: every output row (2025 out-of-fold + target), as the training run wrote them. Each group
  model predicts only the windows of its own out-of-fold days plus the target windows.
--scope target: target rows only (nn_pred, quantiles, pfill; z columns are omitted because their
  PCA is fitted on the out-of-fold rows).
--amp auto: bf16 autocast on CUDA, fp32 on CPU (training default). --amp off: fp32 everywhere.

Writes data/processed/latent_set{tag}/{target}_{device}{suffix}.parquet and a timing JSON next to it;
never overwrites the adopted {target}.parquet.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "8")

import numpy as np
import polars as pl
import torch

from prc.data import load_ranking, mask_departures
from prc.latent_set.oof import FEAT, assemble, label_arrays, load_2025_events, load_weather_table
from prc.latent_set.tokens import CONTEXT, Windows, build_events, build_windows, concat_events, subset_events
from prc.latent_set.train import Device, TrainConfig, model_from_state, oof_groups, predict, window_days
from prc.validation.protocol import FOLDS


def fold_days(fold, days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    day0 = lambda dt: int(dt.timestamp() // 86400)  # noqa: E731
    b = timedelta(days=fold.buffer_days)
    s, e = day0(fold.start), day0(fold.end)
    sb, eb = day0(fold.start - b), day0(fold.end + b)
    days = np.unique(days)
    return days[(days < sb) | (days >= eb)], days[(days >= s) & (days < e)]


def sub_windows(win: Windows, keep: np.ndarray) -> Windows:
    return Windows(win.lo[keep], win.hi[keep], win.c0[keep], win.airport[keep])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["final", "jan", "jul", "q4", "feb"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--amp", default="auto", choices=["auto", "off"])
    ap.add_argument("--scope", default="full", choices=["full", "target"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--out", default=None, help="explicit output parquet path")
    ap.add_argument("--no-stand-bias", action="store_true")
    ap.add_argument("--no-runway-bias", action="store_true")
    ap.add_argument("--weather", action="store_true", help="hourly weather inputs on every token")
    ap.add_argument("--airline", action="store_true", help="carrier from the flight number as a token field (EXP-045)")
    ap.add_argument("--huber", type=float, default=0.0, help="Huber delta (s) for the mean head; 0 = squared error")
    ap.add_argument("--no-arrivals", action="store_true")
    ap.add_argument("--context-min", type=int, default=CONTEXT // 60)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    amp = None if args.amp == "auto" else False
    batch = args.batch or (256 if args.device.startswith("cuda") else 64)
    out_dir = Path(f"data/processed/latent_set{args.tag}")
    out = Path(args.out) if args.out else out_dir / f"{args.target}_{args.device.split(':')[0]}{args.suffix}.parquet"
    adopted = {(out_dir / f"{t}.parquet").resolve() for t in ("final", "jan", "jul", "q4")}
    if out.resolve() in adopted:
        raise SystemExit(f"refusing to overwrite the training-run output {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    ck = Path("artefacts/latent_set")

    t0 = time.time()
    ev, enc = load_2025_events(args.weather, airline=args.airline)
    if args.target == "final":
        ev26, _ = build_events(mask_departures(load_ranking()), pl.read_parquet(FEAT / "ranking.parquet"), enc,
                              weather=load_weather_table() if args.weather else None)
        rank_ids = ev26.mvt_id[ev26.is_dep]
        ev = concat_events(ev, ev26)
    if args.no_arrivals:
        ev = subset_events(ev, ev.is_dep)
    target, fill = label_arrays(ev)
    if args.target == "final":
        is_target = np.isin(ev.mvt_id, rank_ids) & ev.is_dep
        train_days = np.unique(ev.day[ev.is_dep & ~is_target])
        target_days = np.unique(ev.day[is_target])
    else:
        train_days, target_days = fold_days(FOLDS[args.target], ev.day[ev.is_dep])
        is_target = np.isin(ev.day, target_days) & ev.is_dep
    win = build_windows(ev, max_tokens=480, context=args.context_min * 60)
    D = Device(ev, target, fill, dev=args.device)
    prep = time.time() - t0
    groups = oof_groups(train_days, 3)
    cfg = TrainConfig(stand_bias=not args.no_stand_bias, runway_bias=not args.no_runway_bias)
    wd = window_days(win)
    outs, zs, timing = {}, {}, {"prep_sec": round(prep, 1)}
    for g in range(3):
        state = torch.load(ck / f"{args.target}{args.tag}_g{g}.pt", map_location=args.device)
        model = model_from_state(state, ev.num.shape[1], cfg, args.device)
        days = target_days if args.scope == "target" else np.concatenate([target_days, groups[g]])
        w = sub_windows(win, np.isin(wd, days))
        wt = sub_windows(win, np.isin(wd, target_days))
        t1 = time.time()
        if args.scope == "full":
            outs[g], zs[g] = predict(model, D, w, batch=batch, amp=amp)
        else:
            outs[g], zs[g] = predict(model, D, wt, batch=batch, amp=amp)
        timing[f"g{g}_sec"] = round(time.time() - t1, 1)
        timing[f"g{g}_windows"] = int(w.n if args.scope == "full" else wt.n)
        print(f"model {g}: {timing[f'g{g}_windows']} windows in {timing[f'g{g}_sec']:.0f}s on {args.device}",
              flush=True)
    if args.scope == "full":
        frame, align = assemble(ev, outs, zs, groups, train_days, is_target)
    else:
        m = np.mean([o[is_target] for o in outs.values()], 0)
        frame = pl.DataFrame({"MVT_ID_mvt": ev.mvt_id[is_target], "nn_pred": m[:, 0], "nn_q10": m[:, 1],
                              "nn_q50": m[:, 2], "nn_q90": m[:, 3], "nn_pfill": m[:, 4]}).with_columns(
            pl.all().exclude("MVT_ID_mvt").cast(pl.Float32))
        align = []
    frame.write_parquet(out)
    timing |= {"total_sec": round(time.time() - t0, 1), "target_departures": int(is_target.sum()),
               "device": args.device, "threads": args.threads, "amp": args.amp, "scope": args.scope,
               "batch": batch, "align_r2": align, "rows": frame.height}
    out.with_suffix(".json").write_text(json.dumps(timing, indent=1))
    print(json.dumps(timing), flush=True)


if __name__ == "__main__":
    main()
