"""Auxiliary surface-state targets for the window network (EXP-047).

EXP-046 measured that the true surface state at each take-off (aircraft still taxiing, pushbacks just
before, queue ahead, neighbours' taxi times) is worth -12.9 s on ordinary flights on top of the
latent, and that features rebuilt from the network's own estimates add nothing. Here those oracle
quantities become training-only auxiliary targets: the network must encode the surface state in
its bottleneck z. They are computed from the labels of OTHER departures and are never inputs.

Leakage guards:
- a departure's targets are valid only if every departure within +-3 h at its airport is labelled
  (so nothing is computed from unknown or ranking rows);
- the training loop further restricts them to departures whose +-3 h span lies in the model's own
  fitting days (`train.fit`), so out-of-fold and holdout labels never reach a fitted model.
"""

from __future__ import annotations

import bisect

import numpy as np

from prc.latent_set.tokens import Events

AUX_NAMES = ["taxiing", "taxiing_rw", "push_prev600", "push_prev1800", "queue_ahead", "taxi_pm10",
             "taxi_prev1_rw", "taxi_next1_rw"]
SPAN = 3 * 3600  # seconds: reach of every target (off-blocks are clipped at 3 h before take-off)
Y_CLIP = 10800.0


def _one_airport(t: np.ndarray, y: np.ndarray, rw: np.ndarray) -> np.ndarray:
    """t take-off (epoch s, sorted), y true taxi (s, clipped), rw runway ids -> [n, 8] raw targets."""
    n = len(t)
    b = t - y.astype(np.int64)
    bs, ts = np.sort(b), t  # t is sorted
    out = np.full((n, len(AUX_NAMES)), np.nan)
    out[:, 0] = np.searchsorted(bs, t, "right") - np.searchsorted(ts, t, "right")
    out[:, 2] = np.searchsorted(bs, t, "left") - np.searchsorted(bs, t - 600, "left")
    out[:, 3] = np.searchsorted(bs, t, "left") - np.searchsorted(bs, t - 1800, "left")
    q = np.zeros(n)
    seen: list[int] = []
    for i in range(n):  # queue ahead: pushed back >= 10 min before my take-off, took off after me
        thr = t[i] - 600
        q[i] = np.searchsorted(bs, thr, "right") - bisect.bisect_right(seen, thr) - (1 if b[i] <= thr else 0)
        bisect.insort(seen, int(b[i]))
    out[:, 4] = q
    c = np.concatenate([[0.0], np.cumsum(y)])
    lo, hi = np.searchsorted(t, t - 600, "left"), np.searchsorted(t, t + 600, "right")
    cnt = hi - lo - 1
    out[:, 5] = np.where(cnt > 0, (c[hi] - c[lo] - y) / np.maximum(cnt, 1), np.nan)
    for r in np.unique(rw):
        idx = np.where(rw == r)[0]
        tr, yr = t[idx], y[idx]
        out[idx, 1] = np.searchsorted(np.sort(b[idx]), tr, "right") - np.searchsorted(tr, tr, "right")
        prev_ok = np.r_[False, np.diff(tr) <= 3600]
        out[idx, 6] = np.where(prev_ok, np.r_[np.nan, yr[:-1]], np.nan)
        next_ok = np.r_[np.diff(tr) <= 3600, False]
        out[idx, 7] = np.where(next_ok, np.r_[yr[1:], np.nan], np.nan)
    return out


def compute_aux(ev: Events, target: np.ndarray) -> np.ndarray:
    """Standardised auxiliary targets [ev.n, 8] for departure tokens; NaN where invalid or for arrivals."""
    raw = np.full((ev.n, len(AUX_NAMES)), np.nan)
    dep = np.where(ev.is_dep)[0]
    for ap in np.unique(ev.airport[dep]):
        idx = dep[ev.airport[dep] == ap]
        idx = idx[np.argsort(ev.t_ref[idx], kind="stable")]
        t, yv = ev.t_ref[idx].astype(np.int64), target[idx]
        lab = ~np.isnan(yv)
        if lab.sum() < 10:
            continue
        li = idx[lab]
        vals = _one_airport(t[lab], np.clip(yv[lab], 0, Y_CLIP), ev.rw_pair[li])
        # invalid if an unlabelled departure of this airport lies within SPAN
        unl = np.sort(t[~lab])
        if len(unl):
            k = np.searchsorted(unl, t[lab])
            d_lo = np.abs(t[lab] - unl[np.clip(k - 1, 0, len(unl) - 1)])
            d_hi = np.abs(unl[np.clip(k, 0, len(unl) - 1)] - t[lab])
            vals[np.minimum(d_lo, d_hi) < SPAN] = np.nan
        raw[li] = vals
    # counts on a log scale, taxi times in units of 10 min; then standardise over valid rows
    raw[:, :5] = np.log1p(raw[:, :5])
    raw[:, 5:] = raw[:, 5:] / 600.0
    mu, sd = np.nanmean(raw, 0), np.nanstd(raw, 0) + 1e-6
    return ((raw - mu) / sd).astype(np.float32)
