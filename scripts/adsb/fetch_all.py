"""Download adsb.lol daily archives and extract airport points, resumable.

Writes only under ROOT (raw/ and points/). Days: January and July of 2025 and 2026, interleaved so
every period gets coverage early. A day is skipped when its points parquet exists.

Usage (as a capped unit): uv run python scripts/adsb/fetch_all.py
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.environ.get("ADSB_ROOT", "data/adsb")
RAW, PTS = f"{ROOT}/raw", f"{ROOT}/points"
N_DL, N_EX = 1, 6


def rate() -> str:
    """The line tops out near 12 MB/s and is shared: half by day, most of it at night (2026-10-05:
    two 8 MB/s streams saturated it and cut the office off)."""
    h = dt.datetime.now().hour
    return "6M"  # 11 MB/s at night also cut the line (2026-10-05 23:45)


def days() -> list[dt.date]:
    """2025 sample (every third day) first, for validation; then the 2026 ranking days; then the
    rest of 2025."""
    j25 = [dt.date(2025, m, d + 1) for d in range(31) for m in (1, 7)]
    first = [d for d in j25 if (d.day - 1) % 3 == 1]
    r26 = [dt.date(2026, m, d + 1) for d in range(31) for m in (1, 7)]
    return first + r26 + [d for d in j25 if d not in first]


def assets(day: dt.date) -> list[tuple[str, str]]:
    tag = f"v{day:%Y.%m.%d}-planes-readsb-prod-0"
    for attempt in range(20):  # network errors: retry; a missing release answers 404
        r = subprocess.run(["gh", "api", f"repos/adsblol/globe_history_{day.year}/releases/tags/{tag}"],
                           capture_output=True, text=True)
        if r.returncode == 0 or "404" in r.stdout + r.stderr:
            break
        time.sleep(30)
    if r.returncode != 0:
        return []
    return [(a["name"], a["browser_download_url"]) for a in json.loads(r.stdout)["assets"]]


def fetch(day: dt.date) -> dt.date | None:
    pre = f"v{day:%Y.%m.%d}"
    if os.path.exists(f"{PTS}/{day}.parquet") or os.path.exists(f"{RAW}/{pre}.ok"):
        return day
    a = assets(day)
    if not a:
        print(f"{day}: no prod release", flush=True)
        return None
    for name, url in a:
        part = name.rsplit(".", 1)[-1]
        part = "aa" if part == "tar" else part  # single-file release
        dst = f"{RAW}/{pre}.tar.{part}"
        for attempt in range(20):
            if subprocess.run(["curl", "-sSL", "--fail", "--limit-rate", rate(), "-o", dst, url]).returncode == 0:
                break
            time.sleep(30)
        else:
            print(f"{day}: download failed", flush=True)
            return None
    open(f"{RAW}/{pre}.ok", "w").close()
    print(f"{day}: downloaded", flush=True)
    return day


def extract(day: dt.date) -> None:
    pre = f"v{day:%Y.%m.%d}"
    out = f"{PTS}/{day}.parquet"
    if os.path.exists(out):
        return
    r = subprocess.run([sys.executable, "scripts/adsb/extract_day.py", RAW, pre, out + ".tmp"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        os.replace(out + ".tmp", out)
        print(r.stdout.strip(), flush=True)
    else:
        print(f"{day}: extract failed {r.stderr[-300:]}", flush=True)


def main() -> None:
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(PTS, exist_ok=True)
    with ThreadPoolExecutor(N_DL) as dl, ThreadPoolExecutor(N_EX) as ex:
        futs = [ex.submit(extract, d) for d in dl.map(fetch, days()) if d is not None]
        for f in futs:
            f.result()
    print("all done", flush=True)


if __name__ == "__main__":
    main()
