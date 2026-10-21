"""Upload one submission file to the team bucket.

Usage: uv run python scripts/upload_submission.py data/submissions/quick-quicksand_v1.parquet

Reads OSN_ACCESS_KEY / OSN_SECRET_KEY from the environment or the gitignored .env; failing that,
from the gitignored credentials*.json file the MinIO console downloads when an access key is
created ("Access Keys", fields accessKey / secretKey). Re-checks the file against the template before
uploading, uploads it at the bucket root under its own name, then lists the bucket.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import polars as pl
from minio import Minio

ENDPOINT = "s3.opensky-network.org"
BUCKET = "prc-2026-quick-quicksand"
NAME = re.compile(r"^quick-quicksand_(v\d+|final)\.parquet$")


def load_env() -> None:
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def load_credentials_json() -> None:
    for f in sorted(Path(".").glob("credentials*.json")):
        d = json.loads(f.read_text())
        if "accessKey" in d and "secretKey" in d:
            os.environ.setdefault("OSN_ACCESS_KEY", d["accessKey"])
            os.environ.setdefault("OSN_SECRET_KEY", d["secretKey"])
            print(f"keys read from {f.name}")
            return


def check(path: Path) -> None:
    if not NAME.match(path.name):
        raise SystemExit(f"bad name {path.name}: must be quick-quicksand_vN.parquet or quick-quicksand_final.parquet")
    t = pl.read_parquet("data/raw/submitting.parquet")
    s = pl.read_parquet(path)
    ok = (s.schema == t.schema and s.height == t.height and s["MVT_ID_mvt"].equals(t["MVT_ID_mvt"])
          and s["TAXITIME_SEC_mvt"].null_count() == 0)
    if not ok:
        raise SystemExit("file does not match the template (schema, ids, order or nulls)")
    print(f"check ok: {path.name}, {s.height} rows")


def main(file: str) -> None:
    path = Path(file)
    check(path)
    load_env()
    if not os.environ.get("OSN_ACCESS_KEY"):
        load_credentials_json()
    key, secret = os.environ.get("OSN_ACCESS_KEY"), os.environ.get("OSN_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("no keys: set OSN_ACCESS_KEY / OSN_SECRET_KEY in .env or add the MinIO credentials json")
    client = Minio(ENDPOINT, access_key=key, secret_key=secret, secure=True)
    existing = [o.object_name for o in client.list_objects(BUCKET)]
    if path.name in existing:
        raise SystemExit(f"{path.name} already in the bucket; use the next version number")
    client.fput_object(BUCKET, path.name, str(path), content_type="application/octet-stream")
    print(f"uploaded {path.name} to {BUCKET}")
    for o in client.list_objects(BUCKET):
        print(" ", o.object_name, o.size)


if __name__ == "__main__":
    main(sys.argv[1])
