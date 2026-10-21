"""Build the compact cache behind the data explorer (data/processed/explorer/).

Reads data/raw/ only (never writes there). Run once after the raw data changes:

    uv run python scripts/explore/build_explorer_cache.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from prc.explore.common import CACHE_DIR, RAW_DIR, configure_threads  # noqa: E402

configure_threads(12)

import polars as pl  # noqa: E402

from prc.explore import analysis, prep  # noqa: E402
from prc.explore.common import PROXIES  # noqa: E402


def main() -> None:
    t0 = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_paths = sorted(RAW_DIR.glob("training_2025-*.parquet"))
    ranking_path = RAW_DIR / "ranking.parquet"
    if len(train_paths) != 12 or not ranking_path.exists():
        raise SystemExit(f"expected 12 training files and ranking.parquet in {RAW_DIR}")

    raw_cols = list(pl.read_parquet_schema(train_paths[0]))
    movements = prep.build_movements(train_paths, ranking_path).collect()
    mv_path = CACHE_DIR / "movements.parquet"
    movements.write_parquet(mv_path, row_group_size=65_536, statistics=True)
    print(f"movements: {movements.shape} -> {mv_path} ({time.time() - t0:.0f}s)")

    lf = movements.lazy()
    outputs = {
        "inventory": prep.column_inventory(lf),
        "null_rates": prep.null_rates(lf, raw_cols),
        "rounding": prep.rounding_table(lf),
        "quality": prep.quality_counts(lf),
        "daily": prep.daily_stats(lf),
        "hourly": prep.hourly_counts(lf),
        "runway_hourly": prep.runway_hourly(lf),
    }
    dep_train = movements.filter((pl.col("dataset") == "train") & (pl.col("PHASE_mvt") == "DEP"))
    dep_all = movements.filter(pl.col("PHASE_mvt") == "DEP")
    outputs["target_quantiles"] = pl.concat(
        [
            analysis.target_quantiles(dep_train, "TAXITIME_SEC_mvt", ["airport"]),
            analysis.target_quantiles(
                dep_train.with_columns(pl.lit("ALL").alias("airport")), "TAXITIME_SEC_mvt", ["airport"]
            ),
        ]
    )
    outputs["tail"] = pl.concat(
        [
            analysis.tail_dominance(dep_train, "TAXITIME_SEC_mvt", ["airport"]),
            analysis.tail_dominance(
                dep_train.with_columns(pl.lit("ALL").alias("airport")), "TAXITIME_SEC_mvt", ["airport"]
            ),
        ]
    )
    # Proxy summary on train (target present) and availability on both datasets.
    outputs["proxy_summary"] = analysis.proxy_summary(
        dep_train, "TAXITIME_SEC_mvt", list(PROXIES), ["airport"]
    )
    outputs["proxy_availability"] = (
        dep_all.group_by("dataset", "airport", "month")
        .agg(pl.len().alias("n"), *[pl.col(p).is_not_null().mean().alias(p) for p in PROXIES])
        .sort("dataset", "airport", "month")
    )
    for name, df in outputs.items():
        df.write_parquet(CACHE_DIR / f"{name}.parquet")
        print(f"{name}: {df.shape}")

    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {
            p.name: {"bytes": p.stat().st_size, "mtime": p.stat().st_mtime}
            for p in [*train_paths, ranking_path]
        },
        "tables": sorted([*outputs, "movements"]),
        "seconds": round(time.time() - t0, 1),
    }
    (CACHE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
