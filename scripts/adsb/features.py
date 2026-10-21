"""Per-departure ADS-B surface features from the extracted adsb.lol airport points.

Built the same way on train (2025) and ranking (2026): only the take-off time, airport, callsign
and aircraft type of the movement record are used, never the off-block.

Matching: take-offs are detected in the tracks (ground point followed by an airborne point within
120 s). Each departure is matched to the take-off at its airport within 120 s with the lowest
score = |dt| + 300 x callsign mismatch (when the NM callsign is known) + 60 x type mismatch,
one-to-one, greedy by score.

Features (NaN when unmatched):
- adsb_seen: take-off minus first ground point of the final ground session (gaps < 180 s);
- adsb_still0: first point stationary (gs <= 1 kt), i.e. seen at the stand or a holding point;
- adsb_move0: take-off minus the first moving point after an initial stationary run (pushback);
- adsb_stop_time, adsb_n_stops: stationary time and stops (>= 20 s) in the visible segment;
- adsb_lat0, adsb_lon0, adsb_gs0: where and how fast the aircraft was first seen;
- adsb_dt, adsb_cs_ok, adsb_type_ok: match quality.
Congestion (all departures, from all tracks at the airport): distinct aircraft moving on the ground
(gs > 3 kt) and stationary on the ground in the 10 min before take-off.

Usage: uv run python scripts/adsb/features.py   -> data/processed/adsb_features.parquet
"""

from __future__ import annotations

import datetime as dt
import glob
import os

import numpy as np
import polars as pl

from prc.data import load_movements

PTS = os.environ.get("ADSB_ROOT", "data/adsb") + "/points"
OUT = os.environ.get("ADSB_OUT", "data/processed/adsb_features.parquet")
SESSION_GAP, LIFTOFF_GAP, MATCH_TOL = 180.0, 120.0, 120.0
STILL_KT, MOVE_KT = 1.0, 3.0
GROUND_KT = float(os.environ.get("ADSB_GROUND_KT", "0"))  # 0 = ground flag only (v12)


def _segments(points: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for (icao, ap), g in points.sort(["icao", "airport", "t", "lat", "lon"]).group_by(["icao", "airport"], maintain_order=True):
        t = g["t"].to_numpy(); gs = np.nan_to_num(g["gs"].to_numpy(), nan=-1.0)
        gr = g["ground"].to_numpy() | ((gs >= 0) & (gs < GROUND_KT))  # many transponders never set the flag
        lat = g["lat"].to_numpy(); lon = g["lon"].to_numpy(); cs = g["callsign"].to_list(); ty = g["type"].to_list()
        for i in np.where(gr[:-1] & ~gr[1:] & (np.diff(t) < LIFTOFF_GAP))[0]:
            j = int(i)
            while j > 0 and gr[j - 1] and t[j] - t[j - 1] < SESSION_GAP:
                j -= 1
            seg = np.arange(j, i + 1)
            st = gs[seg] <= STILL_KT
            still0 = bool(st[0])
            move0 = np.nan
            if still0:
                k = np.argmax(~st) if (~st).any() else None
                if k is not None:
                    move0 = t[seg[k]]
            # stops >= 20 s
            n_stops, stop_time, s = 0, 0.0, None
            for q in range(len(seg)):
                if st[q] and s is None:
                    s = q
                if s is not None and (not st[q] or q == len(seg) - 1):
                    dur = t[seg[q]] - t[seg[s]]
                    if dur >= 20 and not (still0 and s == 0):
                        n_stops += 1; stop_time += dur
                    s = None
            ls, lo_ = lat[seg], lon[seg]
            step = np.hypot(np.diff(ls) * 111000.0, np.diff(lo_) * 111000.0 * np.cos(np.radians(ls[:-1])))
            fast = np.where(gs[seg] > 10.0)[0]
            mv = gs[seg] > STILL_KT
            last_stop_end = np.nan
            for q in range(len(seg) - 1, 0, -1):  # end of the last stop >= 20 s
                if st[q - 1] and not st[q]:
                    k0 = q - 1
                    while k0 > 0 and st[k0 - 1]:
                        k0 -= 1
                    if t[seg[q - 1]] - t[seg[k0]] >= 20:
                        last_stop_end = float(t[seg[q]])
                        break
            shape = {"path_len": float(step.sum()) if len(step) else 0.0,
                     "dist_to_lift": float(np.hypot((lat[i + 1] - lat[j]) * 111000.0,
                                                    (lon[i + 1] - lon[j]) * 111000.0 * np.cos(np.radians(lat[j])))),
                     "fast_t": float(t[seg[fast[0]]]) if len(fast) else np.nan,
                     "mean_gs": float(gs[seg][mv].mean()) if mv.any() else np.nan,
                     "last_stop_end_t": last_stop_end}
            rows.append({**shape, "icao": icao, "airport": ap, "t_takeoff": float(t[i + 1]),
                         "adsb_cs": (cs[int(i) + 1] or cs[int(i)] or ""), "adsb_type": ty[int(i)] or "",
                         "first_ground": float(t[j]), "move0_t": float(move0), "adsb_still0": still0,
                         "adsb_n_stops": n_stops, "adsb_stop_time": stop_time, "adsb_lat0": float(lat[j]),
                         "adsb_lon0": float(lon[j]), "adsb_gs0": float(gs[j]),
                         "seg_t": t[seg].tolist(), "seg_lat": lat[seg].tolist(), "seg_lon": lon[seg].tolist(),
                         "seg_gs": gs[seg].tolist()})
    return pl.DataFrame(rows)


def _congestion(points: pl.DataFrame, deps: pl.DataFrame) -> pl.DataFrame:
    g = points.filter(pl.col("ground")).with_columns((pl.col("t") // 60).cast(pl.Int64).alias("minute"),
                                                     (pl.col("gs") > MOVE_KT).alias("moving"))
    per_min = g.group_by("airport", "minute", "moving").agg(pl.col("icao").unique())
    out = []
    for ap, d in deps.group_by("airport"):
        pm = per_min.filter(pl.col("airport") == ap[0])
        mv = {r[0]: set(r[1]) for r in pm.filter(pl.col("moving")).select("minute", "icao").iter_rows()}
        sti = {r[0]: set(r[1]) for r in pm.filter(~pl.col("moving")).select("minute", "icao").iter_rows()}
        ids, a, b = [], [], []
        for mid, t in d.select("MVT_ID_mvt", "t_mvt").iter_rows():
            m0 = int(t // 60)
            sm, ss = set(), set()
            for m in range(m0 - 10, m0 + 1):
                sm |= mv.get(m, set()); ss |= sti.get(m, set())
            ids.append(mid); a.append(len(sm)); b.append(len(ss))
        out.append(pl.DataFrame({"MVT_ID_mvt": ids, "adsb_cong_moving": a, "adsb_cong_still": b}))
    return pl.concat(out)


STANDS = "data/processed/adsb_stands.parquet"
_STANDS: dict | None = None


def _stand_xy() -> dict:
    global _STANDS
    if _STANDS is None:
        st = pl.read_parquet(STANDS).filter(pl.col("n_fix") >= 3)
        _STANDS = {(a, s): (la, lo) for a, s, la, lo in st.select("airport", "STAND_mvt", "stand_lat", "stand_lon").iter_rows()}
    return _STANDS


def _stand_feats(m: pl.DataFrame) -> pl.DataFrame:
    """adsb_d0_stand: distance (m) of the first ground point to the departure's own stand;
    adsb_leave_stand: take-off minus the first point after the last stationary point within 25 m of
    the stand (start of pushback); adsb_near_stand_t: take-off minus the last point within 60 m."""
    xy = _stand_xy()
    d0, leave, near = [], [], []
    for ap, stand, tm, ts, la, lo, gs in m.select("airport", "STAND_mvt", "t_mvt", "seg_t", "seg_lat", "seg_lon",
                                                   "seg_gs").iter_rows():
        p = xy.get((ap, stand))
        if p is None:
            d0.append(None); leave.append(None); near.append(None)
            continue
        la, lo, ts, gs = np.asarray(la), np.asarray(lo), np.asarray(ts), np.asarray(gs)
        d = np.hypot((la - p[0]) * 111000.0, (lo - p[1]) * 111000.0 * np.cos(np.radians(p[0])))
        d0.append(float(d[0]))
        w = np.where((d < 25) & (gs <= 1.0))[0]
        leave.append(float(tm - ts[min(w[-1] + 1, len(ts) - 1)]) if len(w) else None)
        w60 = np.where(d < 60)[0]
        near.append(float(tm - ts[w60[-1]]) if len(w60) else None)
    return m.with_columns(pl.Series("adsb_d0_stand", d0, dtype=pl.Float64),
                          pl.Series("adsb_leave_stand", leave, dtype=pl.Float64),
                          pl.Series("adsb_near_stand_t", near, dtype=pl.Float64))


def _match(deps: pl.DataFrame, seg: pl.DataFrame) -> pl.DataFrame:
    c = deps.join(seg, on="airport").filter((pl.col("t_mvt") - pl.col("t_takeoff")).abs() <= MATCH_TOL)
    c = c.with_columns(
        (pl.col("CALLSIGN_flt").is_not_null() & (pl.col("CALLSIGN_flt") == pl.col("adsb_cs"))).alias("adsb_cs_ok"),
        (pl.col("AIRCRAFT_TYPE_mvt") == pl.col("adsb_type")).fill_null(False).alias("adsb_type_ok"),
        (pl.col("t_mvt") - pl.col("t_takeoff")).alias("adsb_dt"))
    c = c.with_columns((pl.col("adsb_dt").abs()
                        + 300.0 * (pl.col("CALLSIGN_flt").is_not_null() & ~pl.col("adsb_cs_ok"))
                        + 60.0 * ~pl.col("adsb_type_ok")).alias("score")).sort(["score", "MVT_ID_mvt", "icao", "t_takeoff"])  # ties: deterministic
    used_d, used_k, keep = set(), set(), []
    for k, (mid, ic, tt) in enumerate(c.select("MVT_ID_mvt", "icao", "t_takeoff").iter_rows()):
        if mid in used_d or (ic, tt) in used_k:
            continue
        used_d.add(mid); used_k.add((ic, tt)); keep.append(k)
    m = _stand_feats(c[keep])
    return m.select("MVT_ID_mvt", "adsb_d0_stand", "adsb_leave_stand", "adsb_near_stand_t", "adsb_dt", "adsb_cs_ok", "adsb_type_ok", "adsb_still0", "adsb_n_stops",
                    "adsb_stop_time", "adsb_lat0", "adsb_lon0", "adsb_gs0",
                    pl.col("path_len").alias("adsb_path_len"), pl.col("dist_to_lift").alias("adsb_dist_to_lift"),
                    pl.col("mean_gs").alias("adsb_mean_gs"),
                    (pl.col("t_mvt") - pl.col("fast_t")).alias("adsb_since_fast"),
                    (pl.col("t_mvt") - pl.col("last_stop_end_t")).alias("adsb_since_last_stop"),
                    (pl.col("t_mvt") - pl.col("first_ground")).alias("adsb_seen"),
                    (pl.col("t_mvt") - pl.col("move0_t")).alias("adsb_move0"))


def day_features(day: dt.date, movements: pl.DataFrame) -> pl.DataFrame | None:
    f = f"{PTS}/{day}.parquet"
    if not os.path.exists(f):
        return None
    parts = [pl.read_parquet(f)]
    prev = f"{PTS}/{day - dt.timedelta(days=1)}.parquet"
    if os.path.exists(prev):  # sessions that started before midnight
        parts.append(pl.read_parquet(prev).filter(pl.col("t") > dt.datetime(day.year, day.month, day.day,
                                                  tzinfo=dt.timezone.utc).timestamp() - 7200))
    pts = pl.concat(parts)
    deps = movements.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("MVT_TIME_UTC_mvt").dt.date() == day)).select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("ADEP_mvt").alias("airport"),
        pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("t_mvt"), "CALLSIGN_flt", "AIRCRAFT_TYPE_mvt", "STAND_mvt")
    seg = _segments(pts)
    m = _match(deps, seg) if seg.height else None
    out = deps.select("MVT_ID_mvt").join(_congestion(pts, deps), on="MVT_ID_mvt", how="left")
    if m is not None:
        out = out.join(m, on="MVT_ID_mvt", how="left")
    return out.with_columns(pl.lit(day).alias("adsb_day"))


def main() -> None:
    raw = sorted(glob.glob("data/raw/training_2025-0[17]-*.parquet")) + ["data/raw/ranking.parquet"]
    mv = pl.concat([load_movements(p) for p in raw], how="diagonal_relaxed")
    days = sorted(dt.date.fromisoformat(os.path.basename(p)[:10]) for p in glob.glob(f"{PTS}/*.parquet"))
    frames = []
    for d in days:
        fr = day_features(d, mv)
        if fr is not None and fr.height:
            frames.append(fr)
            print(d, fr.height, f"matched {fr['adsb_seen'].is_not_null().mean():.2f}" if "adsb_seen" in fr.columns else "", flush=True)
    pl.concat(frames, how="diagonal_relaxed").write_parquet(OUT)
    print("->", OUT)


if __name__ == "__main__":
    main()
