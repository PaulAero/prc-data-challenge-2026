#!/usr/bin/env python3
"""Create regular airport-state time series for local forecasting experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from prc.timeseries.airport_state import build_airport_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("training", "ranking"),
        default="training",
        help="Official local dataset to aggregate (default: training)",
    )
    parser.add_argument("--interval", default="15m", help="Polars duration, e.g. 5m or 15m")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "training":
        inputs = sorted(args.raw_dir.glob("training_*.parquet"))
    else:
        inputs = [args.raw_dir / "ranking.parquet"]

    output = args.output or Path(
        f"data/processed/airport_state_{args.interval}_{args.dataset}.parquet"
    )
    state = build_airport_state(inputs, interval=args.interval)
    output.parent.mkdir(parents=True, exist_ok=True)
    state.write_parquet(output)

    target_observed = state.get_column("taxi_out_mean_sec").is_not_null().sum()
    print(f"wrote {state.height:,} rows to {output}")
    print(f"series: {state.get_column('series_id').n_unique()}")
    print(f"target-observed bins: {target_observed:,}")
    print(state.group_by("series_id").agg(
        pl.col("timestamp").min().alias("start"),
        pl.col("timestamp").max().alias("end"),
        pl.col("departure_count").sum().alias("departures"),
    ).sort("series_id"))


if __name__ == "__main__":
    main()
