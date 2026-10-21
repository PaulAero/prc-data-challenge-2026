"""Fetch hourly weather at the ten airports from the Copernicus Climate Data Store (official ERA5).

Usage: uv run --group weather python scripts/external/fetch_weather_cds.py
Needs ~/.cdsapirc (CDS API key) and the dataset licences accepted on the CDS website.
Writes data/external/weather_cds/era5_<ICAO>.parquet and MANIFEST.json.

Sources (licence CC BY 4.0, "Contains modified Copernicus Climate Change Service information"):
- reanalysis-era5-single-levels-timeseries: t2m, d2m, tp, u10, v10, fg10, tcc, cbh, sp
- reanalysis-era5-land-timeseries: sde (snow depth), snowc (snow cover)
"""

from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import cdsapi
import polars as pl

OUT = Path("data/external/weather_cds")
AIRPORTS = {
    "EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861), "EGLL": (51.4700, -0.4543),
    "EHAM": (52.3105, 4.7683), "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626),
    "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389), "LSZH": (47.4647, 8.5492),
    "LTFM": (41.2753, 28.7519),
}
DATES = ["2024-12-25/2026-08-02"]
REQUESTS = {
    "era5": ("reanalysis-era5-single-levels-timeseries",
             ["2m_temperature", "2m_dewpoint_temperature", "total_precipitation", "10m_u_component_of_wind",
              "10m_v_component_of_wind", "10m_wind_gust_since_previous_post_processing", "total_cloud_cover",
              "cloud_base_height", "surface_pressure"]),
    "land": ("reanalysis-era5-land-timeseries", ["snow_depth", "snow_cover"]),
}


def read_zip(raw: bytes) -> pl.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        frames = [pl.read_csv(z.read(n)).drop("latitude", "longitude") for n in z.namelist() if n.endswith(".csv")]
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, on="valid_time", how="full", coalesce=True)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c = cdsapi.Client(quiet=True)
    manifest = {"licence": "CC BY 4.0; Contains modified Copernicus Climate Change Service information",
                "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": {}}
    tmp = OUT / "_tmp.zip"
    for icao, (lat, lon) in AIRPORTS.items():
        parts = []
        for key, (dataset, variables) in REQUESTS.items():
            req = {"variable": variables, "location": {"longitude": lon, "latitude": lat}, "date": DATES,
                   "data_format": "csv"}
            c.retrieve(dataset, req, str(tmp))
            parts.append(read_zip(tmp.read_bytes()))
            manifest["files"].setdefault(f"era5_{icao}.parquet", {})[key] = {"dataset": dataset, "request": req}
        frame = parts[0].join(parts[1], on="valid_time", how="left").with_columns(
            pl.col("valid_time").str.to_datetime("%Y-%m-%d %H:%M:%S", time_zone="UTC"), pl.lit(icao).alias("airport"))
        path = OUT / f"era5_{icao}.parquet"
        frame.write_parquet(path)
        manifest["files"][path.name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["files"][path.name]["rows"] = frame.height
        print(icao, frame.height, frame["valid_time"].min(), frame["valid_time"].max(), flush=True)
    tmp.unlink(missing_ok=True)
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
