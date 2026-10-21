"""Turn extracted adsb.lol airport points into per-take-off surface features.

For every aircraft and airport: detect take-offs (last ground point followed by an airborne point
within 120 s), then walk back through the ground segment before the take-off and record when the
aircraft started to move. Several definitions are kept as features, the model chooses:

- `adsb_first_ground`: first ground point of the session (no gap > 20 min), lower bound on taxi;
- `adsb_move_k`: end of the last stationary spell (gs <= 1 kt, < 40 m drift) lasting >= k minutes
  (k = 2, 5, 10), i.e. the start of the final movement (pushback or taxi);
- `adsb_n_pts`, `adsb_gap_max`: quality of the ground track.

Times are epoch seconds. Matching to the movement records is by airport and take-off time.
"""

from __future__ import annotations

import numpy as np
import polars as pl

SESSION_GAP = 180.0
LIFTOFF_GAP = 120.0
STILL_KT = 1.0
STILL_M = 40.0
SPELLS = (120.0, 300.0, 600.0)


def _dist_m(lat0, lon0, lat, lon):
    return np.hypot((lat - lat0) * 111_000.0, (lon - lon0) * 111_000.0 * np.cos(np.radians(lat0)))


def takeoffs(points: pl.DataFrame) -> pl.DataFrame:
    out = []
    for (icao, ap), g in points.sort("t").group_by(["icao", "airport"], maintain_order=True):
        t = g["t"].to_numpy(); gr = g["ground"].to_numpy(); gs = np.nan_to_num(g["gs"].to_numpy())
        lat = g["lat"].to_numpy(); lon = g["lon"].to_numpy(); cs = g["callsign"].to_list()
        idx = np.where(gr[:-1] & ~gr[1:] & (np.diff(t) < LIFTOFF_GAP))[0]
        for i in idx:
            t_to = t[i + 1]
            # session start: walk back until a gap > SESSION_GAP or a non-ground point
            j = i
            while j > 0 and gr[j - 1] and t[j] - t[j - 1] < SESSION_GAP:
                j -= 1
            seg = np.arange(j, i + 1)
            row = {"icao": icao, "airport": ap, "t_takeoff": t_to, "callsign": cs[i + 1] or cs[i],
                   "type": g["type"][int(i)], "adsb_first_ground": t[j], "adsb_n_pts": len(seg),
                   "adsb_gap_max": float(np.diff(t[seg]).max()) if len(seg) > 1 else np.nan,
                   "adsb_lat0": lat[j], "adsb_lon0": lon[j]}
            # stationary spells inside the segment
            still = gs[seg] <= STILL_KT
            for k in SPELLS:
                row[f"adsb_move_{int(k // 60)}"] = np.nan
            s = None
            spells = []
            for q, ii in enumerate(seg):
                if still[q] and (s is None):
                    s = q
                if (not still[q] or q == len(seg) - 1) and s is not None:
                    e = q if still[q] else q - 1
                    # spell from seg[s] to seg[e]; drift check
                    if _dist_m(lat[seg[s]], lon[seg[s]], lat[seg[e]], lon[seg[e]]) < STILL_M:
                        spells.append((t[seg[s]], t[seg[e]], seg[e]))
                    s = None
            for k in SPELLS:
                long = [sp for sp in spells if sp[1] - sp[0] >= k]
                if long:
                    end = long[-1][2]
                    nxt = min(end + 1, i)
                    row[f"adsb_move_{int(k // 60)}"] = t[nxt]
            out.append(row)
    return pl.DataFrame(out)


def match(deps: pl.DataFrame, tk: pl.DataFrame, tol: float = 90.0) -> pl.DataFrame:
    """deps: MVT_ID_mvt, airport, t_mvt (epoch s), callsign_flt. Nearest take-off within tol;
    callsign agreement as tie-break."""
    d = deps.sort("t_mvt")
    k = tk.sort("t_takeoff").rename({"callsign": "adsb_callsign"})
    j = d.join_asof(k, left_on="t_mvt", right_on="t_takeoff", by="airport", strategy="nearest",
                    tolerance=tol)
    return j.filter(pl.col("t_takeoff").is_not_null())
