"""Stand positions from arrival tracks (label-free: arrivals' stand and in-block are inputs).

For every landing seen in the adsb.lol points (airborne point followed by a ground point within
120 s), follow the ground session (gaps < 180 s) to its last point. Arrivals are matched by
destination airport, landing time (within 120 s) and callsign (as in `features.py`). An arrival
gives a stand fix when its track ends within 180 s of the recorded in-block time at <= 3 kt.
Stand position = median of the fixes; spread = median distance of fixes to it.

Usage: uv run python scripts/adsb/stands.py   -> data/processed/adsb_stands.parquet
"""

from __future__ import annotations

import datetime as dt
import glob
import os

import numpy as np
import polars as pl

from prc.data import load_movements

PTS = os.environ.get("ADSB_ROOT", "data/adsb") + "/points"
OUT = "data/processed/adsb_stands.parquet"
SESSION_GAP, GAP, TOL = 180.0, 120.0, 120.0


def landings(points: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for (icao, ap), g in points.sort(["icao", "airport", "t", "lat", "lon"]).group_by(["icao", "airport"], maintain_order=True):
        t = g["t"].to_numpy(); gr = g["ground"].to_numpy(); gs = np.nan_to_num(g["gs"].to_numpy(), nan=-1.0)
        lat = g["lat"].to_numpy(); lon = g["lon"].to_numpy(); cs = g["callsign"].to_list()
        for i in np.where(~gr[:-1] & gr[1:] & (np.diff(t) < GAP))[0]:
            j = int(i) + 1
            while j + 1 < len(t) and gr[j + 1] and t[j + 1] - t[j] < SESSION_GAP:
                j += 1
            rows.append({"icao": icao, "airport": ap, "t_land": float(t[i + 1]), "adsb_cs": cs[j] or cs[int(i)] or "",
                         "t_last": float(t[j]), "lat_last": float(lat[j]), "lon_last": float(lon[j]),
                         "gs_last": float(gs[j])})
    return pl.DataFrame(rows)


def main() -> None:
    raw = sorted(glob.glob("data/raw/training_2025-0[17]-*.parquet")) + ["data/raw/ranking.parquet"]
    mv = pl.concat([load_movements(p) for p in raw], how="diagonal_relaxed")
    arr = mv.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("STAND_mvt").is_not_null()).select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("ADES_mvt").alias("airport"), "STAND_mvt", "CALLSIGN_flt",
        pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("t_mvt"),
        pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("t_block"),
        pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day"))
    fixes = []
    for p in sorted(glob.glob(f"{PTS}/*.parquet")):
        day = dt.date.fromisoformat(os.path.basename(p)[:10])
        ld = landings(pl.read_parquet(p))
        if not ld.height:
            continue
        a = arr.filter(pl.col("day") == day)
        c = a.join(ld, on="airport").filter((pl.col("t_mvt") - pl.col("t_land")).abs() <= TOL).with_columns(
            ((pl.col("t_mvt") - pl.col("t_land")).abs()
             + 300.0 * (pl.col("CALLSIGN_flt").is_not_null() & (pl.col("CALLSIGN_flt") != pl.col("adsb_cs")))).alias("score"))
        c = c.sort(["score", "MVT_ID_mvt", "icao", "t_land"]).unique("MVT_ID_mvt", keep="first", maintain_order=True).unique(
            ["icao", "t_land"], keep="first", maintain_order=True)
        good = c.filter(((pl.col("t_last") - pl.col("t_block")).abs() <= 180) & (pl.col("gs_last") <= 3))
        fixes.append(good.select("airport", "STAND_mvt", "lat_last", "lon_last"))
        print(day, "landings", ld.height, "matched", c.height, "stand fixes", good.height, flush=True)
    fx = pl.concat(fixes)
    st = fx.group_by("airport", "STAND_mvt").agg(pl.len().alias("n_fix"), pl.col("lat_last").median().alias("stand_lat"),
                                                 pl.col("lon_last").median().alias("stand_lon"))
    fx = fx.join(st, on=["airport", "STAND_mvt"])
    spread = fx.with_columns((np.hypot((pl.col("lat_last") - pl.col("stand_lat")) * 111000.0,
                                       (pl.col("lon_last") - pl.col("stand_lon")) * 111000.0
                                       * (pl.col("stand_lat").radians().cos()))).alias("d")).group_by(
        "airport", "STAND_mvt").agg(pl.col("d").median().alias("spread_m"))
    st = st.join(spread, on=["airport", "STAND_mvt"])
    st.write_parquet(OUT)
    print(st.group_by("airport").agg(pl.len().alias("stands"), pl.col("n_fix").sum(), pl.col("spread_m").median()).sort("airport"))


if __name__ == "__main__":
    main()
