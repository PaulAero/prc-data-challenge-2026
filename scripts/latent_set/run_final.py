"""Latent outputs for the submission: 2025 out-of-fold + ranking (Jan and Jul 2026).

Usage: uv run --group timesfm python scripts/latent_set/run_final.py [--tag ""] [training flags]

Same scheme as the folds (`prc.latent_set.oof.run_oof`): 3 group models over all 2025 departure
days (7-day blocks, 10% early-stopping slice of the fitting days). 2025 rows get out-of-fold
outputs; ranking rows get the mean of the 3 models, z aligned on ranking inputs, PCA fitted on
2025 rows.

Ranking events come from mask_departures(load_ranking()) plus the ranking feature cache, encoded
with the 2025 encoding (unseen categories -> 0). 2025 and 2026 events are merged before windows
are cut, so windows at 1 January 2026 read 31 December 2025 as context.

Writes data/processed/latent_set{tag}/final.parquet and experiments/v2-nn-alone{tag}/final_meta.json.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "8")

import numpy as np
import polars as pl
import torch

from prc.data import load_ranking, mask_departures
from prc.latent_set.oof import FEAT, label_arrays, load_2025_events, load_weather_table, run_oof
from prc.latent_set.tokens import CONTEXT, build_events, build_windows, concat_events, subset_events
from prc.latent_set.train import Device, TrainConfig

torch.set_num_threads(8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-stand-bias", action="store_true")
    ap.add_argument("--no-runway-bias", action="store_true")
    ap.add_argument("--weather", action="store_true", help="hourly weather inputs on every token")
    ap.add_argument("--airline", action="store_true", help="carrier from the flight number as a token field (EXP-045)")
    ap.add_argument("--huber", type=float, default=0.0, help="Huber delta (s) for the mean head; 0 = squared error")
    ap.add_argument("--aux-weight", type=float, default=0.0, help="surface-state auxiliary loss weight (EXP-047)")
    ap.add_argument("--no-arrivals", action="store_true")
    ap.add_argument("--context-min", type=int, default=CONTEXT // 60)
    ap.add_argument("--y-clip", type=float, default=3600.0, help="clip of the non-fill training target (s)")
    ap.add_argument("--max-epochs", type=int, default=16)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out = Path(f"data/processed/latent_set{args.tag}/final.parquet")
    exp_dir = Path(f"experiments/v2-nn-alone{args.tag.replace('_', '-')}")
    ck_dir = Path("artefacts/latent_set")
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite")
    for p in (out.parent, exp_dir, ck_dir):
        p.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ev25, enc = load_2025_events(args.weather, airline=args.airline)
    rank = mask_departures(load_ranking())
    ev26, _ = build_events(rank, pl.read_parquet(FEAT / "ranking.parquet"), enc,
                              weather=load_weather_table() if args.weather else None)
    unseen = {c: float((ev26.cats[:, k] == 0).mean()) for k, c in enumerate(["airport", "runway", "stand",
                                                                             "aircraft_type", "operator", "wake",
                                                                             "segment"])}
    unseen25 = {c: float((ev25.cats[:, k] == 0).mean()) for k, c in enumerate(unseen)}
    ev = concat_events(ev25, ev26)
    if args.no_arrivals:
        ev = subset_events(ev, ev.is_dep)
    target, fill = label_arrays(ev)
    rank_ids = set(rank.filter(pl.col("PHASE_mvt") == "DEP")["MVT_ID_mvt"].to_list())
    is_rank = np.isin(ev.mvt_id, np.fromiter(rank_ids, dtype=np.int64))
    assert np.isnan(target[is_rank]).all(), "ranking departures must have no labels"
    win = build_windows(ev, max_tokens=480, context=args.context_min * 60)
    aux = None
    if args.aux_weight > 0:  # EXP-047: surface-state targets from other departures' labels (training only)
        from prc.latent_set.aux import compute_aux
        ta = time.time()
        aux = compute_aux(ev, target)
        print(f"aux targets: {int((~np.isnan(aux[:, 0])).sum())} departures, {time.time() - ta:.0f}s", flush=True)
    D = Device(ev, target, fill, y_clip=args.y_clip, aux=aux)
    with open(ck_dir / f"encoding_final{args.tag}.pkl", "wb") as f:
        pickle.dump(enc, f)
    print(f"events {ev.n} (ranking deps {int(is_rank.sum())}), windows {win.n}, prep {time.time() - t0:.0f}s",
          flush=True)

    cfg = TrainConfig(stand_bias=not args.no_stand_bias, runway_bias=not args.no_runway_bias, huber=args.huber, aux_weight=args.aux_weight,
                      max_epochs=args.max_epochs, patience=args.patience, lr=args.lr, seed=args.seed)
    train_days = np.unique(ev.day[ev.is_dep & ~is_rank])
    tf = time.time()
    frame, meta = run_oof(D, win, ev, train_days, is_rank, cfg, 3, ck_dir / f"final{args.tag}",
                          log=lambda s: print(f"[final] {s}", flush=True))
    frame.write_parquet(out)
    (exp_dir / "final_meta.json").write_text(json.dumps(
        {"fold": "final", "config": asdict(cfg), "args": vars(args), **meta, "seconds": round(time.time() - tf, 1),
         "rows": frame.height, "n_windows": win.n, "unseen_share_ranking": unseen,
         "unseen_share_2025": unseen25}, indent=1, default=float))
    print(f"[final] done in {time.time() - tf:.0f}s, rows {frame.height}", flush=True)


if __name__ == "__main__":
    main()
