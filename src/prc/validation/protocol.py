"""VAL-MONTH-V2: whole-month holdouts that mimic the ranking file.

Features are built once from a movement table in which *every* departure's off-block and taxi time
is blanked (the ranking condition), so they are identical whichever fold is held out. Only
label-derived components (models, target encodings, stage-1 estimates) depend on the fold, and they
must be fitted on the fold's training rows only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl


@dataclass(frozen=True)
class Fold:
    name: str
    start: datetime
    end: datetime
    buffer_days: int = 3

    def holdout(self) -> pl.Expr:
        return (pl.col("event_time") >= self.start) & (pl.col("event_time") < self.end)

    def train(self) -> pl.Expr:
        b = timedelta(days=self.buffer_days)
        return (pl.col("event_time") < self.start - b) | (pl.col("event_time") >= self.end + b)


def _utc(y: int, m: int, d: int = 1) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


FOLDS = {
    "jan": Fold("jan", _utc(2025, 1), _utc(2025, 2)),
    "jul": Fold("jul", _utc(2025, 7), _utc(2025, 8)),
    "q4": Fold("q4", _utc(2025, 10), _utc(2026, 1)),
    # stress fold: the only 2025 winter disruption (LTFM snow, February) unseen in training
    "feb": Fold("feb", _utc(2025, 2), _utc(2025, 3)),
    # final phase months (Feb and Jun 2026 added by the organisers): generalisation check
    "jun": Fold("jun", _utc(2025, 6), _utc(2025, 7)),
}
PRIMARY = ("jan", "jul")
CLEAN_MAX_TARGET = 7200.0
CORE_MAX_TARGET = 10800.0  # all flights except > 3 h, whose error is decided by a handful of overnight records


def clean_mask(frame: pl.DataFrame) -> np.ndarray:
    return ((frame["no_nm"] == 0) & (frame["target"] <= CLEAN_MAX_TARGET)).to_numpy()


def _rmse(se: np.ndarray) -> float:
    return float(np.sqrt(se.mean())) if len(se) else float("nan")


def score(frame: pl.DataFrame, pred_col: str = "pred") -> dict:
    """RMSE on all flights, clean flights, and per airport. `frame` needs target, no_nm, airport."""
    se = ((frame[pred_col] - frame["target"]) ** 2).to_numpy()
    clean = clean_mask(frame)
    ap = frame["airport"].to_numpy()
    core = (frame["target"] <= CORE_MAX_TARGET).to_numpy()
    out = {"n": int(len(se)), "all": _rmse(se), "core": _rmse(se[core]), "clean": _rmse(se[clean]),
           "n_clean": int(clean.sum()), "n_core": int(core.sum()), "per_airport": {}}
    for a in sorted(set(ap)):
        m = ap == a
        out["per_airport"][a] = {"all": _rmse(se[m]), "clean": _rmse(se[m & clean])}
    return out


def paired_day_bootstrap(frame: pl.DataFrame, a: str, b: str, n_boot: int = 2000, seed: int = 0,
                         clean_only: bool = True) -> dict:
    """RMSE(a) - RMSE(b) resampled over days; positive means b is better."""
    f = frame.filter(pl.Series(clean_mask(frame))) if clean_only else frame
    f = f.with_columns(pl.col("event_time").dt.date().alias("_d"),
                       ((pl.col(a) - pl.col("target")) ** 2).alias("_sa"),
                       ((pl.col(b) - pl.col("target")) ** 2).alias("_sb"))
    days = f.group_by("_d").agg(pl.col("_sa").sum(), pl.col("_sb").sum(), pl.len())
    sa, sb, n = (days[c].to_numpy() for c in ("_sa", "_sb", "len"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n), size=(n_boot, len(n)))
    diff = np.sqrt(sa[idx].sum(1) / n[idx].sum(1)) - np.sqrt(sb[idx].sum(1) / n[idx].sum(1))
    return {"median": float(np.median(diff)), "lo95": float(np.quantile(diff, 0.025)),
            "hi95": float(np.quantile(diff, 0.975)), "p_b_better": float((diff > 0).mean())}
