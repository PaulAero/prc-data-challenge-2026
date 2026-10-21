"""Extract the low-altitude points near the 10 challenge airports from one adsb.lol daily archive.

Input: the split tar of one `globe_history` release (`<day>.tar.aa`, `.ab`, ...), streamed without
unpacking to disk. Output: one parquet of points (icao, reg, type, callsign, t, lat, lon, ground,
alt_ft, gs, track, airport) for aircraft seen within RADIUS_KM of an airport below MAX_ALT_FT.

adsb.lol data: github.com/adsblol/globe_history_2025 / _2026, ODbL-1.0 (repo), CC0 (archive).

Usage: uv run python scripts/adsb/extract_day.py <dir> <day-prefix> <out.parquet>
"""

from __future__ import annotations

import glob
import gzip
import subprocess
import sys
import tarfile

import numpy as np
import orjson
import polars as pl

AIRPORTS = {  # aerodrome reference points
    "EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861), "EGLL": (51.4700, -0.4543),
    "EHAM": (52.3086, 4.7639), "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626),
    "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389), "LSZH": (47.4647, 8.5492),
    "LTFM": (41.2753, 28.7519),
}
RADIUS_KM = 8.0
MAX_ALT_FT = 3000
_NAMES = list(AIRPORTS)
_LAT = np.array([v[0] for v in AIRPORTS.values()])
_LON = np.array([v[1] for v in AIRPORTS.values()])
_DLAT = np.full(len(_LAT), RADIUS_KM / 111.0)
_DLON = RADIUS_KM / (111.0 * np.cos(np.radians(_LAT)))


def _near(lat: float, lon: float) -> int:
    for k in range(len(_NAMES)):
        if abs(lat - _LAT[k]) < _DLAT[k] and abs(lon - _LON[k]) < _DLON[k]:
            return k
    return -1


def parse(raw: bytes, rows: dict[str, list]) -> None:
    d = orjson.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
    t0 = d.get("timestamp", 0.0)
    icao, reg, typ = d.get("icao"), d.get("r"), d.get("t")
    cs = None
    for p in d.get("trace", ()):
        det = p[8]
        if det and det.get("flight"):
            cs = det["flight"].strip()
        lat, lon = p[1], p[2]
        if lat is None or lon is None:
            continue
        alt = p[3]
        ground = alt == "ground"
        if not ground and (alt is None or alt > MAX_ALT_FT):
            continue
        k = _near(lat, lon)
        if k < 0:
            continue
        rows["icao"].append(icao); rows["reg"].append(reg); rows["type"].append(typ)
        rows["callsign"].append(cs); rows["t"].append(t0 + p[0]); rows["lat"].append(lat)
        rows["lon"].append(lon); rows["ground"].append(ground)
        rows["alt_ft"].append(None if ground else float(alt)); rows["gs"].append(p[4])
        rows["track"].append(p[5]); rows["airport"].append(_NAMES[k])


def main(src_dir: str, prefix: str, out: str) -> None:
    parts = sorted(glob.glob(f"{src_dir}/{prefix}.tar.a?"))
    cat = subprocess.Popen(["cat", *parts], stdout=subprocess.PIPE)
    rows: dict[str, list] = {k: [] for k in ("icao", "reg", "type", "callsign", "t", "lat", "lon",
                                             "ground", "alt_ft", "gs", "track", "airport")}
    n = 0
    with tarfile.open(fileobj=cat.stdout, mode="r|") as tf:
        for m in tf:
            if not m.isfile() or "/trace_full_" not in m.name:
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            try:
                parse(f.read(), rows)
            except Exception:  # corrupt member: skip
                continue
            n += 1
    cat.wait()
    df = pl.DataFrame(rows, schema={"icao": pl.String, "reg": pl.String, "type": pl.String,
                                    "callsign": pl.String, "t": pl.Float64, "lat": pl.Float64,
                                    "lon": pl.Float64, "ground": pl.Boolean, "alt_ft": pl.Float64,
                                    "gs": pl.Float64, "track": pl.Float64, "airport": pl.String})
    df.sort("icao", "t").write_parquet(out)
    print(f"{prefix}: {n} traces, {df.height} points, {df['icao'].n_unique()} aircraft -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
