"""Two-stage neighbour features: estimate every departure's taxi-out, then describe each flight's
neighbourhood with those estimates (the blanked off-blocks are estimated, never read).

Stage-1 estimates must have the same quality on training and evaluation rows: training rows get
out-of-fold predictions (day-grouped K-fold inside the fold's training months); every other row
gets the prediction of a model fitted on all training rows.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from prc.models.gbdt import LGBMRegressor


def stage1_estimates(frame: pl.DataFrame, train_mask: pl.Expr, cols: list[str], k: int = 4, rounds: int = 600,
                     seed: int = 20260926) -> pl.DataFrame:
    """`frame`: all departures with features, target, fill, event_time. Returns MVT_ID_mvt, est."""
    frame = frame.with_columns(train_mask.alias("_train"), pl.col("event_time").dt.date().alias("_day"))
    train = frame.filter("_train")
    days = train.select(pl.col("_day").unique().sort())
    rng = np.random.default_rng(seed)
    days = days.with_columns(pl.Series("_k", rng.permutation(days.height) % k, dtype=pl.Int32))
    train = train.join(days, on="_day", how="left")
    fit_rows = lambda f: f.filter(pl.col("fill") == 0)  # physical taxi-out, not the schedule-fill artefact
    parts = []
    for j in range(k):
        m = LGBMRegressor(cols, rounds=rounds, seed=seed).fit(fit_rows(train.filter(pl.col("_k") != j)))
        held = train.filter(pl.col("_k") == j)
        parts.append(held.select("MVT_ID_mvt", pl.Series("est", m.predict(held))))
    rest = frame.filter(~pl.col("_train"))
    m = LGBMRegressor(cols, rounds=rounds, seed=seed).fit(fit_rows(train))
    parts.append(rest.select("MVT_ID_mvt", pl.Series("est", m.predict(rest))))
    return pl.concat(parts)


def _overtaken(t: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each i: number of j with t_j < t_i and b_j > b_i (pushed back after me, took off before me)."""
    order = np.argsort(t, kind="stable")
    ranks = np.searchsorted(np.sort(b), b, "left") + 1
    n = len(t)
    tree = np.zeros(n + 2, dtype=np.int64)
    out = np.zeros(n, dtype=np.int64)
    inserted = 0
    for i in order:
        r = ranks[i]
        s, x = 0, r
        while x > 0:
            s += tree[x]
            x -= x & -x
        out[i] = inserted - s  # inserted with rank > r
        x = r
        while x <= n:
            tree[x] += 1
            x += x & -x
        inserted += 1
    return out


def neighbour_features(frame: pl.DataFrame, est: pl.DataFrame, prefix: str = "nb") -> pl.DataFrame:
    """`frame`: MVT_ID_mvt, airport, runway, event_time. `est`: MVT_ID_mvt, est (seconds)."""
    f = frame.select("MVT_ID_mvt", "airport", "runway", pl.col("event_time").dt.epoch("s").alias("t")).join(
        est, on="MVT_ID_mvt", how="inner")
    out = []
    for (ap,), g in f.group_by(["airport"]):
        g = g.sort("t")
        t = g["t"].to_numpy().astype(float)
        e = np.clip(g["est"].to_numpy(), 60, 7200)
        b = t - e
        bs, ts = np.sort(b), np.sort(t)
        r: dict[str, np.ndarray] = {"MVT_ID_mvt": g["MVT_ID_mvt"].to_numpy(), f"{prefix}_self_est": e}
        # aircraft still taxiing out when I take off / when I (estimatedly) push back; self excluded
        r[f"{prefix}_taxiing_at_to"] = np.searchsorted(bs, t, "right") - np.searchsorted(ts, t, "right") - 1
        r[f"{prefix}_taxiing_at_ob"] = np.searchsorted(bs, b, "right") - np.searchsorted(ts, b, "right") - 1
        r[f"{prefix}_to_between"] = np.searchsorted(ts, t, "left") - np.searchsorted(ts, b, "left")
        r[f"{prefix}_overtaken"] = _overtaken(t, b)
        cs = np.concatenate([[0.0], np.cumsum(e)])
        lo, hi = np.searchsorted(t, t - 600, "left"), np.searchsorted(t, t + 600, "right")
        nn = hi - lo - 1
        r[f"{prefix}_pm600_mean"] = np.where(nn > 0, (cs[hi] - cs[lo] - e) / np.maximum(nn, 1), np.nan)
        rw = g["runway"].to_numpy()
        r[f"{prefix}_rw_prev5_mean"] = np.full(len(t), np.nan)
        r[f"{prefix}_rw_taxiing_at_to"] = np.full(len(t), np.nan)
        for w in np.unique(rw):
            idx = np.where(rw == w)[0]
            ei = e[idx]
            c = np.concatenate([[0.0], np.cumsum(ei)])
            j = np.arange(len(idx))
            lo5 = np.maximum(j - 5, 0)
            r[f"{prefix}_rw_prev5_mean"][idx] = np.where(j > 0, (c[j] - c[lo5]) / np.maximum(j - lo5, 1), np.nan)
            bw, tw = np.sort(b[idx]), np.sort(t[idx])
            r[f"{prefix}_rw_taxiing_at_to"][idx] = np.searchsorted(bw, t[idx], "right") - np.searchsorted(tw, t[idx], "right") - 1
        out.append(pl.DataFrame(r))
    res = pl.concat(out, how="vertical_relaxed")
    return res.with_columns([pl.col(c).cast(pl.Float32) for c in res.columns if c != "MVT_ID_mvt"])
