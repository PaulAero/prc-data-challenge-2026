"""Training, out-of-fold prediction and embedding alignment for the joint window model.

Labels enter only through the loss, on departures whose take-off day is in the fitting day set.
Every token's inputs come from `tokens.build_events`, i.e. from a masked movement table.
"""

from __future__ import annotations

import contextlib
import copy
import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from prc.latent_set.model import WindowEncoder, loss_fn
from prc.latent_set.tokens import CENTRAL, Events, Windows

Y_SCALE = 600.0
Y_CLIP = 3600.0


@dataclass
class TrainConfig:
    d: int = 128
    heads: int = 8
    layers: int = 4
    ff: int = 256
    z_dim: int = 16
    drop: float = 0.1
    context: bool = True
    stand_bias: bool = True
    runway_bias: bool = True
    batch: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.01
    max_epochs: int = 14
    patience: int = 2
    edge_weight: float = 0.25   # loss weight of labelled departures outside the central slice
    fill_weight: float = 0.2
    huber: float = 0.0          # seconds; 0 = squared error for the mean head (EXP-045)
    aux_weight: float = 0.0     # weight of the surface-state auxiliary loss (EXP-047); 0 = off
    seed: int = 20260926


class Device:
    """Event arrays and labels resident on the GPU."""

    def __init__(self, ev: Events, target: np.ndarray, fill: np.ndarray, dev: str = "cuda", y_clip: float = Y_CLIP,
                 aux: np.ndarray | None = None):
        t = lambda a, dt=None: torch.as_tensor(a, dtype=dt, device=dev)  # noqa: E731
        self.dev = dev
        self.num = t(ev.num, torch.float16)
        self.cats = t(ev.cats.astype(np.int64))
        self.is_dep = t(ev.is_dep)
        self.t_ref = t(ev.t_ref)
        self.t_stand = t(ev.t_stand)
        self.rw = t(ev.rw_pair)
        self.st = t(ev.st_pair)
        self.day = t(ev.day.astype(np.int64))
        self.day0 = int(ev.day.min())
        self.n_days = int(ev.day.max()) - self.day0 + 1
        self.has_y = t(~np.isnan(target) & ev.is_dep)
        self.y_clip = y_clip  # training target clip (s); also the ceiling of the early-stopping metric
        self.y = t(np.clip(np.nan_to_num(target), 0, y_clip) / Y_SCALE, torch.float32)
        self.fill = t(np.nan_to_num(fill), torch.float32)
        self.n = ev.n
        self.aux = self.aux_ok = None
        if aux is not None:  # EXP-047: standardised auxiliary targets, NaN = invalid
            self.aux = t(np.nan_to_num(aux), torch.float32)
            self.aux_ok = t(~np.isnan(aux))
            span = 3 * 3600  # targets reach +-3 h: both ends must be fitting days (see prc.latent_set.aux)
            self.day_lo = t(((ev.t_ref - span) // 86400).astype(np.int64))
            self.day_hi = t(((ev.t_ref + span) // 86400).astype(np.int64))

    def day_mask(self, days: np.ndarray) -> torch.Tensor:
        m = torch.zeros(self.n_days, dtype=torch.bool, device=self.dev)
        m[torch.as_tensor(days - self.day0, device=self.dev)] = True
        return m

    def batch(self, win: Windows, w: np.ndarray):
        lo = torch.as_tensor(win.lo[w], device=self.dev)
        ln = torch.as_tensor(win.hi[w] - win.lo[w], device=self.dev)
        c0 = torch.as_tensor(win.c0[w], device=self.dev)
        L = int(ln.max())
        ar = torch.arange(L, device=self.dev)
        pad = ar[None, :] >= ln[:, None]
        idx = torch.where(pad, lo[:, None], lo[:, None] + ar[None, :])
        centre = c0 + CENTRAL // 2
        t_ref = self.t_ref[idx]
        rel = t_ref - centre[:, None]
        central = (t_ref >= c0[:, None]) & (t_ref < c0[:, None] + CENTRAL) & ~pad & self.is_dep[idx]
        inputs = dict(num=self.num[idx].float(), cats=self.cats[idx], is_dep=self.is_dep[idx] & ~pad,
                      t_rel=rel.float() / 3600.0, t_ref=rel, t_stand=self.t_stand[idx] - centre[:, None],
                      rw=self.rw[idx], st=self.st[idx], pad=pad)
        return inputs, idx, central, pad


def _weights(D: Device, idx, central, pad, day_ok: torch.Tensor, edge_weight: float):
    lab = D.has_y[idx] & ~pad & day_ok[D.day[idx] - D.day0]
    w = lab.float() * torch.where(central, 1.0, edge_weight)
    w_reg = w * (D.fill[idx] == 0).float()
    return w_reg, w


def window_days(win: Windows) -> np.ndarray:
    return (win.c0 // 86400).astype(np.int64)


def fit(D: Device, win: Windows, fit_days: np.ndarray, val_days: np.ndarray, cfg: TrainConfig,
        log=print) -> tuple[WindowEncoder, dict]:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    wd = window_days(win)
    tr_w = np.where(np.isin(wd, fit_days))[0]
    va_w = np.where(np.isin(wd, val_days))[0]
    fit_ok, val_ok = D.day_mask(fit_days), D.day_mask(val_days)
    vocab = [int(D.cats[:, k].max()) for k in range(D.cats.shape[1])]
    use_aux = cfg.aux_weight > 0 and D.aux is not None
    model = WindowEncoder(D.num.shape[1], vocab, cfg.d, cfg.heads, cfg.layers, cfg.ff, cfg.z_dim, cfg.drop,
                          cfg.context, cfg.stand_bias, cfg.runway_bias,
                          n_aux=D.aux.shape[1] if use_aux else 0).to(D.dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(tr_w) / cfg.batch)
    total = steps_per_epoch * cfg.max_epochs
    warm = min(500, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(cfg.max_epochs):
        t0 = time.time()
        model.train()
        perm = rng.permutation(tr_w)
        tot = 0.0
        for s in range(steps_per_epoch):
            w = perm[s * cfg.batch:(s + 1) * cfg.batch]
            inp, idx, central, pad = D.batch(win, w)
            w_reg, w_fill = _weights(D, idx, central, pad, fit_ok, cfg.edge_weight)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out, z = model(**inp)
                aux_pred = model.aux_head(z) if use_aux else None
            m = w_fill > 0
            loss = loss_fn(out[m].float(), D.y[idx][m], D.fill[idx][m], w_reg[m], w_fill[m], cfg.fill_weight,
                           cfg.huber / Y_SCALE)
            if use_aux:
                win_ok = fit_ok[(D.day_lo[idx] - D.day0).clamp(0, D.n_days - 1)] & \
                    fit_ok[(D.day_hi[idx] - D.day0).clamp(0, D.n_days - 1)]
                am = (D.aux_ok[idx] & (win_ok & m)[..., None]).float() * w_fill[..., None]
                se = (aux_pred.float() - D.aux[idx]) ** 2
                loss = loss + cfg.aux_weight * (se * am).sum() / am.sum().clamp(min=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
        val = evaluate(model, D, win, va_w, val_ok)
        hist.append({"epoch": ep, "train_loss": tot / steps_per_epoch, **val, "sec": round(time.time() - t0, 1)})
        log(f"  epoch {ep}: loss {tot / steps_per_epoch:.4f} val rmse(nonfill, clip) {val['val_rmse']:.1f} "
            f"auc-ish bce {val['val_bce']:.4f} ({time.time() - t0:.0f}s)")
        if val["val_rmse"] < best - 0.05:
            best, best_state, bad = val["val_rmse"], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model, {"best_val_rmse": best, "history": hist, "n_fit_windows": len(tr_w), "n_val_windows": len(va_w)}


def _amp(D: Device, amp: bool | None):
    """bf16 autocast on CUDA (training default), plain fp32 elsewhere or when amp=False."""
    use = (D.dev.startswith("cuda") if amp is None else amp) and D.dev.startswith("cuda")
    return torch.autocast("cuda", dtype=torch.bfloat16) if use else contextlib.nullcontext()


@torch.no_grad()
def evaluate(model, D: Device, win: Windows, va_w: np.ndarray, val_ok: torch.Tensor, batch: int = 256) -> dict:
    model.eval()
    se, n, bce = 0.0, 0.0, 0.0
    for s in range(0, len(va_w), batch):
        inp, idx, central, pad = D.batch(win, va_w[s:s + batch])
        with _amp(D, None):
            out, _ = model(**inp)
        out = out.float()
        lab = central & D.has_y[idx] & val_ok[D.day[idx] - D.day0]
        nf = lab & (D.fill[idx] == 0)
        se += float((((out[..., 0] - D.y[idx]) * Y_SCALE) ** 2)[nf].sum())
        n += float(nf.sum())
        bce += float(torch.nn.functional.binary_cross_entropy_with_logits(out[..., 4][lab], D.fill[idx][lab],
                                                                           reduction="sum"))
    return {"val_rmse": math.sqrt(se / max(n, 1)), "val_bce": bce / max(n, 1)}


@torch.no_grad()
def predict(model, D: Device, win: Windows, batch: int = 256, amp: bool | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Outputs for every departure token (from the window where it is central).
    Returns out [n_events, 5] (seconds for regression heads, probability for fill) and z [n_events, z].
    amp=None: bf16 autocast on CUDA, fp32 on CPU; amp=False forces fp32 everywhere."""
    model.eval()
    out_all = torch.full((D.n, 5), float("nan"), device=D.dev)
    z_all = torch.full((D.n, model.to_z.out_features), float("nan"), device=D.dev)
    for s in range(0, win.n, batch):
        w = np.arange(s, min(s + batch, win.n))
        inp, idx, central, pad = D.batch(win, w)
        with _amp(D, amp):
            out, z = model(**inp)
        out = out.float()
        out = torch.cat([out[..., :4] * Y_SCALE, torch.sigmoid(out[..., 4:5])], -1)
        out_all[idx[central]] = out[central]
        z_all[idx[central]] = z.float()[central]
    return out_all.cpu().numpy(), z_all.cpu().numpy()


def oof_groups(days: np.ndarray, k: int = 3, block_days: int = 7, seed: int = 20260926) -> dict[int, np.ndarray]:
    """Assign days to k groups by contiguous 7-day blocks (random block order). Returns {group: days}."""
    days = np.unique(days)
    blocks = days // block_days
    ub = np.unique(blocks)
    rng = np.random.default_rng(seed)
    g = rng.permutation(len(ub)) % k
    gmap = dict(zip(ub, g))
    grp = np.array([gmap[b] for b in blocks])
    return {i: days[grp == i] for i in range(k)}


def split_val(days: np.ndarray, frac: float = 0.1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    days = np.unique(days)
    v = rng.random(len(days)) < frac
    return days[~v], days[v]


def align_linear(z_src: np.ndarray, z_ref: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares affine map from one model's z to the reference model's z (fitted on rows both
    models predict, inputs only). Returns the [z+1, z] matrix and the R^2 of the fit."""
    X = np.c_[z_src, np.ones(len(z_src))]
    M, *_ = np.linalg.lstsq(X, z_ref, rcond=None)
    r2 = 1 - ((X @ M - z_ref) ** 2).sum() / ((z_ref - z_ref.mean(0)) ** 2).sum()
    return M, float(r2)


def model_from_state(state: dict, n_num: int, cfg: TrainConfig, device: str = "cpu") -> WindowEncoder:
    """Rebuild a WindowEncoder from a saved state dict (vocabulary sizes read from the weights)."""
    n_cat = len([k for k in state if k.startswith("cat_emb.") and k.endswith(".weight")])
    vocab = [state[f"cat_emb.{k}.weight"].shape[0] - 1 for k in range(n_cat)]
    n_aux = state["aux_head.3.weight"].shape[0] if "aux_head.3.weight" in state else 0
    model = WindowEncoder(n_num, vocab, cfg.d, cfg.heads, cfg.layers, cfg.ff, cfg.z_dim, cfg.drop, cfg.context,
                          cfg.stand_bias, cfg.runway_bias, n_aux=n_aux)
    model.load_state_dict(state)
    return model.to(device).eval()
