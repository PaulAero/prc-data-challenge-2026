"""Fetch hourly weather at the ten airports from the Open-Meteo historical archive (ERA5-based).

Usage: uv run python scripts/external/fetch_weather.py
Writes data/external/weather/openmeteo_<ICAO>.parquet and MANIFEST.json (source URL, licence, sha256).

Data licence: CC BY 4.0, "Weather data by Open-Meteo.com" (https://open-meteo.com/), built on
Copernicus ERA5 / ERA5-Land reanalysis. One request per airport (10 in total), used for an
open-source competition entry.
Coordinates: aerodrome reference points (approximate, 4 decimals).
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import polars as pl

OUT = Path("data/external/weather")
AIRPORTS = {
    "EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861), "EGLL": (51.4700, -0.4543),
    "EHAM": (52.3105, 4.7683), "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626),
    "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389), "LSZH": (47.4647, 8.5492),
    "LTFM": (41.2753, 28.7519),
}
HOURLY = ["temperature_2m", "dew_point_2m", "relative_humidity_2m", "precipitation", "rain", "snowfall",
          "snow_depth", "weather_code", "cloud_cover_low", "wind_speed_10m", "wind_direction_10m",
          "wind_gusts_10m", "surface_pressure"]
START, END = "2024-12-25", "2026-08-02"
URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch(icao: str, lat: float, lon: float) -> tuple[pl.DataFrame, str]:
    q = dict(latitude=lat, longitude=lon, start_date=START, end_date=END, hourly=",".join(HOURLY),
             timezone="UTC", wind_speed_unit="kmh")
    url = f"{URL}?{urllib.parse.urlencode(q)}"
    with urllib.request.urlopen(url, timeout=120) as r:
        data = json.load(r)
    h = data["hourly"]
    frame = pl.DataFrame(h).with_columns(
        pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M", time_zone="UTC"), pl.lit(icao).alias("airport"))
    return frame, url


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"source": "Open-Meteo historical weather API (ERA5 / ERA5-Land reanalysis)",
                "licence": "CC BY 4.0, attribution: Weather data by Open-Meteo.com",
                "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": {}}
    for icao, (lat, lon) in AIRPORTS.items():
        frame, url = fetch(icao, lat, lon)
        path = OUT / f"openmeteo_{icao}.parquet"
        frame.write_parquet(path)
        manifest["files"][path.name] = {"url": url, "rows": frame.height,
                                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(icao, frame.height, frame["time"].min(), frame["time"].max(), flush=True)
        time.sleep(2)
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
