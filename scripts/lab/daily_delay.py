"""EXP-056: daily calibration to EUROCONTROL's published pre-departure delay (DLY_ALL_PRE_2).

Used under the open-data rule (public EUROCONTROL data, same source and terms as the monthly taxi-out
averages the organisers approved). DLY_ALL_PRE_2 (all_pre_departure_delays_<year>.csv, ansperformance.eu)
equals, per airport and take-off day, sum over departures of max(0, off-block - schedule) in minutes
(2025 check: median ratio 0.9999, corr 0.9997). With take-off T and schedule S known, a prediction p
implies off-block T - p; for each airport-day we solve for the shift d such that
sum max(0, T - S - (p + d)) / 60 = published, and move the predictions by lam x d (rows <= 7,200 s).
"""
from __future__ import annotations
import numpy as np
import polars as pl

APS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]


def published(year: int) -> pl.DataFrame:
    f = pl.read_csv(f"data/external/pru/all_pre_departure_delays_{year}.csv", infer_schema_length=0)
    return f.filter(pl.col("APT_ICAO").is_in(APS)).select(
        pl.col("APT_ICAO").alias("airport"), pl.col("FLT_DATE").str.slice(0, 10).str.to_date(strict=False).alias("day"),
        pl.col("DLY_ALL_PRE_2").cast(pl.Float64, strict=False).alias("pub")).drop_nulls()


def _expected_pos(a: np.ndarray, sigma: float) -> np.ndarray:
    """E[max(0, a + e)], e ~ N(0, sigma^2); sigma = 0 gives max(0, a)."""
    if sigma <= 0:
        return np.maximum(0.0, a)
    from scipy.stats import norm
    z = a / sigma
    return a * norm.cdf(z) + sigma * norm.pdf(z)


def solve_shifts(x: pl.DataFrame, pred_col: str, lo: float = -900.0, hi: float = 900.0, sigma: float = 0.0) -> pl.DataFrame:
    """sigma: per-flight prediction uncertainty (s) used for the expected contribution (EXP-056b)."""
    out = []
    for (ap, day), g in x.group_by(["airport", "day"]):
        ts = g["ts"].to_numpy(); p = g[pred_col].to_numpy(); target = g["pub"][0] * 60.0
        ok = p <= 7200
        f = lambda d: _expected_pos(ts - (p + np.where(ok, d, 0.0)), sigma).sum()
        a, b = lo, hi
        if f(a) < target:
            d = a
        elif f(b) > target:
            d = b
        else:
            for _ in range(50):
                m = (a + b) / 2
                if f(m) > target:
                    a = m
                else:
                    b = m
            d = (a + b) / 2
        out.append((ap, day, d))
    return pl.DataFrame(out, schema=["airport", "day", "shift"], orient="row")


def apply(x: pl.DataFrame, pred_col: str, shifts: pl.DataFrame, lam: float) -> np.ndarray:
    y = x.join(shifts, on=["airport", "day"], how="left").with_columns(pl.col("shift").fill_null(0.0))
    p = y[pred_col].to_numpy()
    return np.where(p <= 7200, p + lam * y["shift"].to_numpy(), p)


def fill_calibrate(g: pl.DataFrame, pred_col: str, lam: float, kmax: float = 6.0) -> pl.DataFrame:
    """EXP-056b, LIRF: fill-aware daily calibration. A schedule-filled record contributes 0 to the
    published sum (off-block = schedule); a real late off-block contributes T - S - taxi. Per row the
    fill weight is read from the prediction, q = (p - r) / (ms - r) with r the latent network's
    non-fill estimate (nn_pred) and ms = T - S; per day a common logit shift k is solved so that
    sum (1 - q) max(0, ms - r) matches the published total, and lam x k is applied:
    p' = p + (q' - q) (ms - r). Rows with ms - r <= 600 s are left alone.
    g: MVT_ID_mvt, day, ts, nn_pred, pub, pred_col. Returns MVT_ID_mvt, fc."""
    out = []
    for _, h in g.group_by("day"):
        p = h[pred_col].to_numpy(); ms = h["ts"].to_numpy()
        r = np.clip(h["nn_pred"].fill_null(900.0).to_numpy(), 300, 3600)
        span = ms - r
        ok = span > 600
        ph = np.where(ok, np.clip((p - r) / np.where(ok, span, 1.0), 0.001, 0.999), 0.5)
        lg = np.log(ph / (1 - ph))
        target = h["pub"][0] * 60.0

        def E(k):
            q = np.where(ok, 1 / (1 + np.exp(-(lg + k))), 0.0)
            return ((1 - q) * np.maximum(0, ms - r)).sum()

        a, b = -kmax, kmax
        if E(a) < target:
            k = a
        elif E(b) > target:
            k = b
        else:
            for _ in range(40):
                m = (a + b) / 2
                if E(m) > target:
                    a = m
                else:
                    b = m
            k = (a + b) / 2
        q = 1 / (1 + np.exp(-(lg + lam * k)))
        new = np.where(ok, p + (q - ph) * span, p)
        out.append(h.select("MVT_ID_mvt").with_columns(pl.Series("fc", new)))
    return pl.concat(out)
