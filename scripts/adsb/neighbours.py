"""Neighbour queue features from other departures' ADS-B tracks (EXP-053).

EXP-046 measured that true neighbour taxi-outs are worth about -13 s on ordinary flights. ADS-B shows
part of them for matched departures. For every departure (matched or not): aggregates over the other
matched departures of the same airport taking off within +-15 min (and in the 15 min before):
count, mean visible taxi (adsb_seen), mean stop time, mean time since the last stop, mean time since
leaving the stand where seen. Label-free: built from tracks and take-off times only.

Usage: uv run python scripts/adsb/neighbours.py -> data/processed/adsb_neighbours.parquet
"""

from __future__ import annotations

import glob

import numpy as np
import polars as pl

from prc.data import load_movements

WIN = 900.0
COLS = ["adsb_seen", "adsb_stop_time", "adsb_since_last_stop", "adsb_leave_stand"]


def main() -> None:
    raw = sorted(glob.glob("data/raw/training_2025-0[17]-*.parquet")) + ["data/raw/ranking.parquet"]
    mv = pl.concat([load_movements(p) for p in raw], how="diagonal_relaxed").filter(pl.col("PHASE_mvt") == "DEP").select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("ADEP_mvt").alias("airport"),
        pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("t"))
    a = pl.read_parquet("data/processed/adsb_features.parquet").select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), *[pl.col(c).cast(pl.Float64) for c in COLS])
    x = mv.join(a, on="MVT_ID_mvt", how="inner")  # departures on ADS-B days
    out = []
    for (ap,), g in x.sort("t").group_by(["airport"], maintain_order=True):
        t = g["t"].to_numpy(); ok = ~np.isnan(g["adsb_seen"].to_numpy())
        vals = {c: g[c].to_numpy() for c in COLS}
        res = {"MVT_ID_mvt": g["MVT_ID_mvt"].to_numpy()}
        for lab, lo_off, hi_off in (("pm15", -WIN, WIN), ("b15", -WIN, 0.0)):
            lo = np.searchsorted(t, t + lo_off, "left"); hi = np.searchsorted(t, t + hi_off, "right")
            n = np.zeros(len(t)); sums = {c: np.zeros(len(t)) for c in COLS}; cnts = {c: np.zeros(len(t)) for c in COLS}
            cs_ok = np.concatenate([[0], np.cumsum(ok)])
            for c in COLS:
                v = vals[c]; m = ~np.isnan(v)
                cs = np.concatenate([[0.0], np.cumsum(np.where(m, v, 0.0))]); cc = np.concatenate([[0], np.cumsum(m)])
                s = cs[hi] - cs[lo] - np.where(m, v, 0.0)  # exclude self
                k = cc[hi] - cc[lo] - m
                res[f"nb_{lab}_{c.removeprefix('adsb_')}"] = np.where(k > 0, s / np.maximum(k, 1), np.nan)
            res[f"nb_{lab}_n"] = (cs_ok[hi] - cs_ok[lo] - ok).astype(float)
        out.append(pl.DataFrame(res))
    pl.concat(out).write_parquet("data/processed/adsb_neighbours.parquet")
    print("->", "data/processed/adsb_neighbours.parquet", sum(o.height for o in out))


if __name__ == "__main__":
    main()
