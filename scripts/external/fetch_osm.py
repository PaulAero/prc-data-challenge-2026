"""Fetch airport geometry from OpenStreetMap (Overpass API) for the ten airports.

Usage: uv run python scripts/external/fetch_osm.py
Writes data/external/osm/<ICAO>.json (raw Overpass JSON) and MANIFEST.json.

Licence: OpenStreetMap data, (c) OpenStreetMap contributors, ODbL 1.0
(https://www.openstreetmap.org/copyright). Attribution required; derived databases that are
distributed publicly must be offered under ODbL.
Features: aeroway = parking_position, runway, taxiway, holding_position, apron, deicing_pad (if any).
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path("data/external/osm")
AIRPORTS = {  # aerodrome reference points; bounding box +-HALF degrees around them
    "EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861), "EGLL": (51.4700, -0.4543),
    "EHAM": (52.3105, 4.7683), "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626),
    "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389), "LSZH": (47.4647, 8.5492),
    "LTFM": (41.2753, 28.7519),
}
HALF = {"lat": 0.045, "lon": 0.07}
URLS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
QUERY = """[out:json][timeout:180];
(
  nwr["aeroway"~"^(parking_position|runway|taxiway|holding_position|apron|deicing_pad|stand)$"]({s},{w},{n},{e});
);
out geom tags;"""


def post(q: str) -> bytes:
    last = None
    for attempt in range(6):
        url = URLS[attempt % len(URLS)]
        try:
            req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": q}).encode(),
                                         headers={"User-Agent": "prc-data-challenge-research/0.1"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception as exc:  # noqa: BLE001  (server busy / timeout: back off and retry)
            last = exc
            print("  retry", attempt + 1, url, exc, flush=True)
            time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"Overpass failed: {last}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"source": "OpenStreetMap via Overpass API", "licence": "ODbL 1.0, (c) OpenStreetMap contributors",
                "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": {}}
    for icao, (lat, lon) in AIRPORTS.items():
        path = OUT / f"{icao}.json"
        q = QUERY.format(s=lat - HALF["lat"], w=lon - HALF["lon"], n=lat + HALF["lat"], e=lon + HALF["lon"])
        raw = path.read_bytes() if path.exists() and path.stat().st_size > 1000 else post(q)
        path.write_bytes(raw)
        elems = json.loads(raw)["elements"]
        counts: dict[str, int] = {}
        for e in elems:
            k = e.get("tags", {}).get("aeroway", "?")
            counts[k] = counts.get(k, 0) + 1
        manifest["files"][path.name] = {"query": q, "sha256": hashlib.sha256(raw).hexdigest(), "counts": counts}
        print(icao, counts, flush=True)
        time.sleep(5)
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
