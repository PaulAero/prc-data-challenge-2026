"""Record-level post-processing rules on identifiable rows (EXP-049).

Each rule is confined to rows identifiable label-free in the ranking file, rests on 2025 evidence
measured out of fold, and has a mechanism. They apply after any model or blend.

1. `lobt_floor`: NM-matched departures taking off more than 2 h after LOBT. The NM join puts the
   off-block within 60 min of LOBT; in 2025 it sits about 15 min after LOBT (median), while the model
   drifts to the far window edge. pred = max(pred, take-off - LOBT - 1,200 s). Full-year out of fold:
   143 rows, squared error -65%, all six two-month blocks improve (2 h is the threshold: from 1.5 h
   it hurts).
2. `twin`: duplicate departure records (same airport, flight number and schedule, take-offs within
   60 s), one NM-matched and one not, at LFPG and EDDM. The matched twin holds the first off-block
   (pushed back, returned, pushed again): target = take-off - LOBT - 300 s (median). 2025: 90 twins,
   RMSE 2,451 s (model) -> 1,116 s (rule).
3. `winter_nonm`: no-NM departures outside LIRF on airport-days with more than 6% no-NM departures,
   frozen precipitation and a minimum temperature below 3 C. On de-icing days LOBT is re-filed while
   the aircraft queues after pushback, so the off-block falls more than 60 min before LOBT and the NM
   match fails: there, no-NM selects long taxis. pred *= factor (fitted on 2025, 1.29 in sample;
   leave-one-day-out squared error -37%; non-frozen disruption days: no effect).
"""

from __future__ import annotations

import numpy as np
import polars as pl

LOBT_FLOOR_MIN = 7200.0
LOBT_FLOOR_GAP = 1200.0
TWIN_AIRPORTS = ("LFPG", "EDDM")
TWIN_GAP = 300.0
WINTER_NONM_SHARE = 0.06
WINTER_TMIN = 3.0
WINTER_FROZEN_SHARE = 0.2


def rule_frame(movements: pl.DataFrame, feats: pl.DataFrame) -> pl.DataFrame:
    """Per departure: the label-free quantities the rules need. movements: raw table (masked is fine),
    feats: feature cache rows (MVT_ID_mvt, airport, no_nm, mvt_vs_lobt_sec, wx_frozen_6h, wx_snow_12h,
    wx_tmin_6h, event_time)."""
    dep = movements.filter(pl.col("PHASE_mvt") == "DEP").select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), "ADEP_mvt", "FLIGHT_mvt", "SCHED_TIME_UTC_mvt",
        pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").alias("_t"), pl.col("FLIGHT_ID_mvt").is_not_null().alias("_nm"))
    a = dep.drop_nulls("FLIGHT_mvt")
    j = a.join(a, on=["ADEP_mvt", "FLIGHT_mvt", "SCHED_TIME_UTC_mvt"], suffix="_b").filter(
        (pl.col("MVT_ID_mvt") != pl.col("MVT_ID_mvt_b")) & ((pl.col("_t") - pl.col("_t_b")).abs() <= 60)
        & pl.col("_nm") & ~pl.col("_nm_b") & pl.col("ADEP_mvt").is_in(TWIN_AIRPORTS))
    twins = j.select("MVT_ID_mvt").unique().with_columns(pl.lit(True).alias("twin"))
    f = feats.select("MVT_ID_mvt", "airport", "no_nm", "mvt_vs_lobt_sec", "event_time",
                     ((pl.col("wx_frozen_6h") > 0) | (pl.col("wx_snow_12h") > 0)).alias("_frozen"), "wx_tmin_6h")
    f = f.with_columns(pl.col("event_time").dt.date().alias("_day"))
    day = f.group_by("airport", "_day").agg(pl.col("no_nm").mean().alias("_share"),
                                             pl.col("_frozen").mean().alias("_fz"), pl.col("wx_tmin_6h").min().alias("_tmin"))
    f = f.join(day, on=["airport", "_day"], how="left").join(twins, on="MVT_ID_mvt", how="left")
    winter = ((pl.col("airport") != "LIRF") & (pl.col("no_nm") == 1) & (pl.col("_share") > WINTER_NONM_SHARE)
              & (pl.col("_fz") > WINTER_FROZEN_SHARE) & (pl.col("_tmin") < WINTER_TMIN))
    return f.select("MVT_ID_mvt", "no_nm", "mvt_vs_lobt_sec", pl.col("twin").fill_null(False),
                    winter.fill_null(False).alias("winter"))


def apply_rules(rf: pl.DataFrame, pred: np.ndarray, winter_factor: float, rules=("lobt_floor", "twin", "winter_nonm")) -> np.ndarray:
    """rf aligned row by row with pred."""
    out = pred.astype(float).copy()
    tl = rf["mvt_vs_lobt_sec"].cast(pl.Float64).to_numpy()
    matched = (rf["no_nm"].to_numpy() == 0) & ~np.isnan(tl)
    if "winter_nonm" in rules:
        out = np.where(rf["winter"].to_numpy(), out * winter_factor, out)
    if "lobt_floor" in rules:
        fl = matched & (tl > LOBT_FLOOR_MIN)
        out = np.where(fl, np.maximum(out, tl - LOBT_FLOOR_GAP), out)
    if "twin" in rules:
        tw = rf["twin"].to_numpy() & matched
        out = np.where(tw, np.maximum(tl - TWIN_GAP, 0.0), out)
    return out


def fit_winter_factor(rf: pl.DataFrame, pred: np.ndarray, target: np.ndarray) -> float:
    """Least-squares multiplicative factor on the winter rows (target clipped at 3 h)."""
    w = rf["winter"].to_numpy() & ~np.isnan(target) & (target <= 10800)
    if w.sum() < 30:
        return 1.0
    return float((target[w] * pred[w]).sum() / (pred[w] ** 2).sum())


THY_P_FILL = 0.9


def thy_regime(movements: pl.DataFrame, feats: pl.DataFrame) -> pl.DataFrame:
    """EXP-050: Turkish Airlines' LIRF records switched to substitute block times on 2025-10-17
    (departures 376/378 fills to year end; arrivals' substitute signature 15-23% -> 77-79%). January
    2026 arrivals still show it (62%), July 2026 not (25%). Returns, for THY LIRF departures in
    January 2026, the fill value (take-off - schedule, or 600 s for an early take-off)."""
    dep = movements.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF")
                           & (pl.col("FLIGHT_mvt").str.starts_with("THY") | pl.col("FLIGHT_mvt").str.starts_with("TK"))
                           & (pl.col("MVT_TIME_UTC_mvt").dt.year() == 2026) & (pl.col("MVT_TIME_UTC_mvt").dt.month() == 1))
    ms = (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds()
    return dep.select(pl.col("MVT_ID_mvt").cast(pl.Int64),
                      pl.when(ms >= 0).then(ms).otherwise(600).cast(pl.Float64).alias("thy_fill_value"))


def apply_thy(pred: np.ndarray, ids: np.ndarray, thy: pl.DataFrame, nonfill_est: dict[int, float]) -> np.ndarray:
    """pred aligned with ids; THY rows become 0.9 x fill value + 0.1 x non-fill estimate."""
    out = pred.astype(float).copy()
    pos = {int(i): k for k, i in enumerate(ids)}
    for mid, fv in zip(thy["MVT_ID_mvt"].to_list(), thy["thy_fill_value"].to_list()):
        k = pos.get(int(mid))
        if k is not None:
            out[k] = THY_P_FILL * fv + (1 - THY_P_FILL) * nonfill_est.get(int(mid), 900.0)
    return out
