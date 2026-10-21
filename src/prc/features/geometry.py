"""Airport geometry features from OpenStreetMap (ODbL, (c) OpenStreetMap contributors).

For each departure: position of its stand, position of the departure threshold of its runway, and
the distances between them. These describe the physical taxi route and help where the model has
no history (new stands, rare stand x runway pairs).

Stand matching, in order (flag `geo_stand_match`): 3 exact normalised ref; 2 zero-normalised
(C08 == C8); 1 same stand number with suffix letters averaged (115 -> 115A/115B); 0.5 centroid of
the stand group sharing the leading letters or the first digit; missing otherwise.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import polars as pl

OSM_DIR = Path("data/external/osm")
EARTH = 6_371_000.0


def _norm(x: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(x).upper())


def _canon(x: str) -> str:
    """Letters + number without leading zeros + suffix: 'C08' -> 'C8', '0115B' -> '115B'."""
    m = re.match(r"^([A-Z]*)0*(\d+)([A-Z]*)$", _norm(x))
    return f"{m.group(1)}{m.group(2)}{m.group(3)}" if m else _norm(x)


def _base(x: str) -> str:
    m = re.match(r"^([A-Z]*)0*(\d+)", _norm(x))
    return f"{m.group(1)}{m.group(2)}" if m else _norm(x)


def _group(x: str) -> str:
    m = re.match(r"^([A-Z]+)", _norm(x))
    if m:
        return m.group(1)
    m = re.match(r"^0*(\d)", _norm(x))
    return f"#{m.group(1)}" if m else ""


def _centroid(e: dict) -> tuple[float, float] | None:
    if "lat" in e:
        return e["lat"], e["lon"]
    g = e.get("geometry")
    if g:
        return float(np.mean([p["lat"] for p in g])), float(np.mean([p["lon"] for p in g]))
    return None


def _bearing(a: tuple[float, float], b: tuple[float, float]) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    y = math.sin(lo2 - lo1) * math.cos(la2)
    x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(lo2 - lo1)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def load_airport(icao: str) -> tuple[dict, dict, tuple[float, float]]:
    """Returns (stand lookups, runway designator -> (threshold, far end), airport centre)."""
    el = json.loads((OSM_DIR / f"{icao}.json").read_text())["elements"]
    exact, canon, base, group = {}, {}, {}, {}
    pts = []
    for e in el:
        t = e.get("tags", {})
        if t.get("aeroway") not in ("parking_position", "stand") or not t.get("ref"):
            continue
        c = _centroid(e)
        if c is None:
            continue
        pts.append(c)
        for ref in re.split(r"[;,]", t["ref"]):
            exact.setdefault(_norm(ref), []).append(c)
            canon.setdefault(_canon(ref), []).append(c)
            base.setdefault(_base(ref), []).append(c)
            group.setdefault(_group(ref), []).append(c)
    mean = lambda d: {k: (float(np.mean([p[0] for p in v])), float(np.mean([p[1] for p in v]))) for k, v in d.items()}  # noqa: E731
    stands = {"exact": mean(exact), "canon": mean(canon), "base": mean(base), "group": mean(group)}
    runways = {}
    for e in el:
        t = e.get("tags", {})
        if t.get("aeroway") != "runway" or not t.get("ref") or not e.get("geometry"):
            continue
        g = e["geometry"]
        a, b = (g[0]["lat"], g[0]["lon"]), (g[-1]["lat"], g[-1]["lon"])
        for des in t["ref"].split("/"):
            m = re.match(r"^0*(\d{1,2})", des.strip())
            if not m:
                continue
            heading = int(m.group(1)) * 10
            # take-off roll starts at the end from which the runway points along its heading
            d_ab = abs((_bearing(a, b) - heading + 180) % 360 - 180)
            start, end = (a, b) if d_ab < 90 else (b, a)
            runways[des.strip().upper()] = (start, end)
    centre = (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))) if pts else (0.0, 0.0)
    return stands, runways, centre


def _xy(p: tuple[float, float], o: tuple[float, float]) -> tuple[float, float]:
    x = math.radians(p[1] - o[1]) * EARTH * math.cos(math.radians(o[0]))
    y = math.radians(p[0] - o[0]) * EARTH
    return x, y


def geometry_features(masked: pl.DataFrame) -> pl.DataFrame | None:
    if not OSM_DIR.exists():
        return None
    dep = masked.filter(pl.col("PHASE_mvt") == "DEP").select("MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt")
    out = []
    for (ap,), g in dep.group_by(["ADEP_mvt"]):
        if not (OSM_DIR / f"{ap}.json").exists():
            continue
        stands, runways, centre = load_airport(ap)
        rows = {k: [] for k in ["MVT_ID_mvt", "geo_stand_match", "geo_stand_x", "geo_stand_y", "geo_thr_x", "geo_thr_y",
                                "geo_dist_stand_thr", "geo_dist_stand_mid", "geo_rwy_len"]}
        cache: dict[tuple, tuple] = {}
        for mid, st, rw in g.select("MVT_ID_mvt", "STAND_mvt", "RUNWAY_mvt").iter_rows():
            key = (st, rw)
            if key not in cache:
                pos, level = None, None
                if st is not None:
                    for lvl, table, fn in [(3, "exact", _norm), (2, "canon", _canon), (1, "base", _base), (0.5, "group", _group)]:
                        k = fn(st)
                        if k and k in stands[table]:
                            pos, level = stands[table][k], lvl
                            break
                thr = runways.get(str(rw).upper()) if rw is not None else None
                sx = sy = tx = ty = d_thr = d_mid = rlen = None
                if pos is not None:
                    sx, sy = _xy(pos, centre)
                if thr is not None:
                    tx, ty = _xy(thr[0], centre)
                    ex, ey = _xy(thr[1], centre)
                    rlen = math.hypot(ex - tx, ey - ty)
                    if pos is not None:
                        d_thr = math.hypot(sx - tx, sy - ty)
                        d_mid = math.hypot(sx - (tx + ex) / 2, sy - (ty + ey) / 2)
                cache[key] = (level, sx, sy, tx, ty, d_thr, d_mid, rlen)
            vals = cache[key]
            rows["MVT_ID_mvt"].append(mid)
            for k, v in zip(list(rows)[1:], vals):
                rows[k].append(v)
        out.append(pl.DataFrame(rows, schema={"MVT_ID_mvt": pl.Int64, **{k: pl.Float32 for k in list(rows)[1:]}}))
    return pl.concat(out) if out else None
