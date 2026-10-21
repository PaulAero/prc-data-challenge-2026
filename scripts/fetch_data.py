"""Download the official challenge files from the OpenSky S3 bucket into data/raw/.

Credentials come from the gitignored credentials.json issued by the platform, or from the
environment / a gitignored .env at the repo root:
    OSN_ACCESS_KEY, OSN_SECRET_KEY
They are generated on the OSN account once the team is approved. Never commit them.

Usage:
    uv run python scripts/fetch_data.py --list          # list buckets and objects only
    uv run python scripts/fetch_data.py                 # download everything missing
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

from minio import Minio

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ENDPOINT = "s3.opensky-network.org"
DATA_BUCKET = "competition-data"  # name used in 2024/2025; confirm with --list


def load_env() -> None:
    creds = ROOT / "credentials.json"  # as issued by the platform: {"clientId", "clientSecret"}
    if creds.exists():
        c = json.loads(creds.read_text())
        os.environ.setdefault("OSN_ACCESS_KEY", c["clientId"])
        os.environ.setdefault("OSN_SECRET_KEY", c["clientSecret"])
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def client() -> Minio:
    load_env()
    return Minio(
        ENDPOINT,
        access_key=os.environ["OSN_ACCESS_KEY"],
        secret_key=os.environ["OSN_SECRET_KEY"],
        secure=True,
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--bucket", default=DATA_BUCKET)
    args = ap.parse_args()

    mc = client()
    if args.list:
        for b in mc.list_buckets():
            print("bucket:", b.name)
        for o in mc.list_objects(args.bucket, recursive=True):
            print(f"  {o.object_name}  {o.size / 1e6:.1f} MB  {o.last_modified}")
        return

    RAW.mkdir(parents=True, exist_ok=True)
    manifest = []
    for o in mc.list_objects(args.bucket, recursive=True):
        dest = RAW / o.object_name
        if not dest.exists() or dest.stat().st_size != o.size:
            print("downloading", o.object_name)
            mc.fget_object(args.bucket, o.object_name, str(dest))
        manifest.append(f"{sha256(dest)}  {o.object_name}  {o.size}  {o.last_modified}")
    (RAW / "MANIFEST.sha256").write_text("\n".join(manifest) + "\n")
    print(f"{len(manifest)} files, manifest written to data/raw/MANIFEST.sha256")


if __name__ == "__main__":
    main()
