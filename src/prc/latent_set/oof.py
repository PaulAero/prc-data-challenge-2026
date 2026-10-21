"""Out-of-fold scheme shared by the VAL-MONTH-V2 folds and the final (ranking) run.

Fitting days are split into k groups of contiguous 7-day blocks. Model g is fitted on the other
groups (10% of those days held out for early stopping) and predicts group g (out-of-fold). The
target rows (holdout month, or the ranking file) get the mean of the k models. Embeddings z of
models 1..k-1 are mapped onto model 0 by an affine least-squares fit on the target rows (inputs
only), then reduced by a PCA fitted on the fitting-day rows only.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from prc.latent_set.tokens import Events, Windows
from prc.latent_set.train import Device, TrainConfig, align_linear, fit, oof_groups, predict, split_val

N_PCA = 8


def run_oof(D: Device, win: Windows, ev: Events, train_days: np.ndarray, target_rows: np.ndarray,
            cfg: TrainConfig, k: int = 3, ckpt_prefix: Path | None = None, log=print,
            only_group: int | None = None) -> tuple[pl.DataFrame, dict]:
    """Fit the k group models, predict everything, assemble. Returns (frame, metadata)."""
    groups = oof_groups(train_days, k)
    outs, zs, metas = {}, {}, []
    for g in ([only_group] if only_group is not None else range(k)):
        fit_days, val_days = split_val(np.setdiff1d(train_days, groups[g]), 0.1, seed=g)
        log(f"group {g}: fit days {len(fit_days)}, val days {len(val_days)}, oof days {len(groups[g])}")
        tg = time.time()
        model, meta = fit(D, win, fit_days, val_days, cfg, log=log)
        outs[g], zs[g] = predict(model, D, win, amp=False)  # fp32 inference: device-independent outputs
        if ckpt_prefix is not None:
            torch.save(model.state_dict(), f"{ckpt_prefix}_g{g}.pt")
        meta["seconds"] = round(time.time() - tg, 1)
        metas.append(meta)
        del model
        torch.cuda.empty_cache()
    frame, align = assemble(ev, outs, zs, groups, train_days, target_rows)
    meta = {"k": k, "groups": {int(g): len(d) for g, d in groups.items()}, "align_r2": align, "models": metas}
    return frame, meta


def assemble(ev: Events, outs: dict, zs: dict, groups: dict, train_days: np.ndarray,
             target_rows: np.ndarray) -> tuple[pl.DataFrame, list[float]]:
    """Out-of-fold rows from their group model, target rows as the mean of the models, z aligned on
    the target rows to the first model, PCA fitted on fitting-day rows. outs/zs: {group: full arrays}."""
    ev_train = np.isin(ev.day, train_days) & ev.is_dep
    target_rows = target_rows & ev.is_dep
    oof = np.full((ev.n, 5), np.nan, dtype=np.float32)
    oof_z = np.full((ev.n, next(iter(zs.values())).shape[1]), np.nan, dtype=np.float32)
    tgt_out, tgt_z, align = [], [], []
    for g in sorted(outs):
        out, z = outs[g], zs[g]
        rows = np.isin(ev.day, groups[g]) & ev.is_dep
        if tgt_z:
            M, r2 = align_linear(z[target_rows], tgt_z[0])
            z = np.c_[z, np.ones(len(z))] @ M
            align.append(r2)
        oof[rows], oof_z[rows] = out[rows], z[rows]
        tgt_out.append(out[target_rows]); tgt_z.append(z[target_rows])
    res, zz = oof.copy(), oof_z.copy()
    res[target_rows] = np.mean(tgt_out, 0)
    zz[target_rows] = np.mean(tgt_z, 0)
    keep = ev_train | target_rows
    tr_rows = ev_train & ~np.isnan(zz[:, 0])
    mu = zz[tr_rows].mean(0)
    _, _, vt = np.linalg.svd(zz[tr_rows] - mu, full_matrices=False)
    pcs = (zz - mu) @ vt[:N_PCA].T
    cols = {"MVT_ID_mvt": ev.mvt_id[keep], "nn_pred": res[keep, 0], "nn_q10": res[keep, 1],
            "nn_q50": res[keep, 2], "nn_q90": res[keep, 3], "nn_pfill": res[keep, 4]}
    cols |= {f"nn_z{j}": pcs[keep, j] for j in range(N_PCA)}
    frame = pl.DataFrame(cols).drop_nulls("nn_pred").with_columns(pl.all().exclude("MVT_ID_mvt").cast(pl.Float32))
    return frame, align


FEAT = Path(__import__("os").environ.get("PRC_FEATURES_DIR", "data/processed/features"))


def load_weather_table() -> pl.DataFrame:
    from prc.features.flight import load_weather
    w = load_weather()
    if w is None:
        raise FileNotFoundError("no weather files in data/external/weather")
    return w


def load_2025_events(weather: bool = False, airline: bool = False):
    """Events and encoding for all 2025 movements (the encoding is fitted here)."""
    from prc.data import load_training, mask_departures
    from prc.latent_set.tokens import build_events
    masked = mask_departures(load_training())
    ev, enc = build_events(masked, pl.read_parquet(FEAT / "train.parquet"),
                           weather=load_weather_table() if weather else None, airline=airline)
    return ev, enc


def label_arrays(ev: Events) -> tuple[np.ndarray, np.ndarray]:
    """Target and fill aligned to events; NaN for arrivals and for rows without labels (ranking)."""
    labels = pl.read_parquet(FEAT / "labels.parquet")
    lab = pl.DataFrame({"MVT_ID_mvt": ev.mvt_id}).join(labels, on="MVT_ID_mvt", how="left")
    target = lab["target"].cast(pl.Float64).fill_null(float("nan")).to_numpy().copy()
    fill = lab["fill"].cast(pl.Float64).fill_null(float("nan")).to_numpy().copy()
    target[~ev.is_dep] = np.nan
    return target, fill
