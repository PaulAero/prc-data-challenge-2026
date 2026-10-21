"""On-the-fly analysis functions used by the explorer views.

All functions are pure polars on (small or filtered) DataFrames and are tested
in ``tests/test_explore.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

QUANTILES: tuple[float, ...] = (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999)


def target_quantiles(df: pl.DataFrame, value: str, by: Sequence[str]) -> pl.DataFrame:
    """n, mean, sd, quantiles, min and max of ``value`` per group."""

    v = pl.col(value)
    aggs = [
        v.count().alias("n"),
        v.mean().alias("mean"),
        v.std().alias("sd"),
        v.min().alias("min"),
        *[v.quantile(q, interpolation="linear").alias(f"p{q * 100:g}") for q in QUANTILES],
        v.max().alias("max"),
    ]
    return df.group_by(list(by)).agg(aggs).sort(list(by))


def tail_dominance(
    df: pl.DataFrame, value: str, by: Sequence[str], fracs: Sequence[float] = (0.01, 0.05)
) -> pl.DataFrame:
    """Share of the total squared deviation from the group mean carried by the
    top ``frac`` of flights (ranked by squared deviation).

    The group-mean predictor has RMSE = sqrt(mean squared deviation), so this is
    the share of that predictor's MSE explained by the worst ``frac`` flights.
    ``rmse_without_top_x`` is the RMSE after dropping them (mean re-estimated
    is not done on purpose: it shows how much of the error the tail alone is).
    """

    d = df.filter(pl.col(value).is_not_null()).with_columns(
        ((pl.col(value) - pl.col(value).mean().over(by)) ** 2).alias("_sq")
    )
    d = d.with_columns(
        pl.col("_sq").rank(method="ordinal", descending=True).over(by).alias("_rank"),
        pl.len().over(by).alias("_n"),
    )
    aggs = [
        pl.len().alias("n"),
        pl.col("_sq").mean().sqrt().alias("rmse_group_mean"),
    ]
    for f in fracs:
        top = pl.col("_rank") <= (pl.col("_n") * f).ceil()
        tag = f"{f * 100:g}pct"
        aggs += [
            (pl.col("_sq").filter(top).sum() / pl.col("_sq").sum()).alias(f"share_sq_top_{tag}"),
            pl.col("_sq").filter(~top).mean().sqrt().alias(f"rmse_without_top_{tag}"),
        ]
    return d.group_by(list(by)).agg(aggs).sort(list(by))


def proxy_summary(
    df: pl.DataFrame, target: str, proxies: Sequence[str], by: Sequence[str]
) -> pl.DataFrame:
    """How well each proxy predicts the target, per group (long format).

    - availability: share of rows with a non-null proxy
    - rmse_raw: RMSE of the proxy used as is (rows where both are present)
    - bias: mean of (target - proxy)
    - rmse_debiased: RMSE after removing the group bias (= sd of residual)
    - mae_raw, median_residual, corr
    - within_60s: share with |target - proxy| <= 60 s
    """

    frames = []
    for p in proxies:
        r = pl.col(target) - pl.col(p)
        both = pl.col(target).is_not_null() & pl.col(p).is_not_null()
        agg = df.group_by(list(by)).agg(
            pl.len().alias("n"),
            pl.col(p).is_not_null().mean().alias("availability"),
            both.sum().alias("n_both"),
            (r.filter(both) ** 2).mean().sqrt().alias("rmse_raw"),
            r.filter(both).mean().alias("bias"),
            r.filter(both).std(ddof=0).alias("rmse_debiased"),
            r.filter(both).abs().mean().alias("mae_raw"),
            r.filter(both).median().alias("median_residual"),
            pl.corr(pl.col(target).filter(both), pl.col(p).filter(both)).alias("corr"),
            (r.filter(both).abs() <= 60).mean().alias("within_60s"),
        )
        frames.append(agg.with_columns(pl.lit(p).alias("proxy")))
    return pl.concat(frames).sort([*by, "proxy"])


def histogram(
    df: pl.DataFrame, value: str, by: Sequence[str], lo: float, hi: float, width: float
) -> pl.DataFrame:
    """Binned counts of ``value`` in [lo, hi) with fixed ``width``.

    Values outside the range are counted in ``below`` / ``above`` rows
    (bin_left = lo - width and hi) so nothing is silently dropped.
    """

    v = pl.col(value)
    d = df.filter(v.is_not_null()).with_columns(
        pl.when(v < lo)
        .then(pl.lit(lo - width))
        .when(v >= hi)
        .then(pl.lit(hi))
        .otherwise(((v - lo) / width).floor() * width + lo)
        .cast(pl.Float64)
        .alias("bin_left")
    )
    return (
        d.group_by([*by, "bin_left"])
        .agg(pl.len().alias("count"))
        .with_columns(
            pl.when(pl.col("bin_left") < lo)
            .then(pl.lit("below"))
            .when(pl.col("bin_left") >= hi)
            .then(pl.lit("above"))
            .otherwise(pl.lit("in_range"))
            .alias("zone")
        )
        .sort([*by, "bin_left"])
    )


def density_2d(
    df: pl.DataFrame, x: str, y: str, lo: float, hi: float, width: float
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """2D histogram counts for x, y clipped to [lo, hi]. Returns
    (counts[y_bin, x_bin], bin_edges, n_in_range, n_total_pairs)."""

    d = df.select(x, y).drop_nulls()
    edges = np.arange(lo, hi + width, width, dtype=float)
    xs = d[x].to_numpy().astype(float)
    ys = d[y].to_numpy().astype(float)
    counts, _, _ = np.histogram2d(ys, xs, bins=[edges, edges])
    return counts, edges, int(counts.sum()), d.height


def category_summary(
    df: pl.DataFrame, category: str, value: str, min_n: int = 1, by: Sequence[str] = ()
) -> pl.DataFrame:
    """Counts and box-plot quantiles of ``value`` per category, sorted by median."""

    v = pl.col(value)
    return (
        df.filter(v.is_not_null())
        .group_by([*by, category])
        .agg(
            pl.len().alias("n"),
            v.mean().alias("mean"),
            v.quantile(0.05).alias("p5"),
            v.quantile(0.25).alias("p25"),
            v.median().alias("p50"),
            v.quantile(0.75).alias("p75"),
            v.quantile(0.9).alias("p90"),
            v.quantile(0.95).alias("p95"),
        )
        .filter(pl.col("n") >= min_n)
        .sort([*by, "p50"])
    )


def unseen_share(reference: pl.DataFrame, candidate: pl.DataFrame, column: str) -> dict[str, float]:
    """Share of ``candidate`` rows (and distinct values) whose ``column`` value
    never appears in ``reference``. Nulls count as a value of their own."""

    ref_values = reference.select(pl.col(column).unique())
    cand = candidate.select(column)
    seen = cand.join(ref_values, on=column, how="semi", nulls_equal=True)
    n_rows = cand.height
    distinct = cand.unique()
    distinct_seen = distinct.join(ref_values, on=column, how="semi", nulls_equal=True)
    return {
        "rows": n_rows,
        "rows_unseen_share": (n_rows - seen.height) / n_rows if n_rows else float("nan"),
        "distinct": distinct.height,
        "distinct_unseen": distinct.height - distinct_seen.height,
    }


def active_count(
    df: pl.DataFrame, start: str, end: str, grid: Sequence[datetime] | pl.Series
) -> pl.DataFrame:
    """Number of intervals [start, end) active at each grid time.

    Used for the count of aircraft currently taxiing out (off-block <= t <
    take-off). Rows with a null bound or end < start are ignored.
    """

    d = df.select(start, end).drop_nulls().filter(pl.col(end) >= pl.col(start))
    grid_s = pl.Series("t", list(grid)) if not isinstance(grid, pl.Series) else grid.alias("t")
    if d.height == 0:
        return pl.DataFrame({"t": grid_s, "active": np.zeros(len(grid_s), dtype=np.int64)})
    s = np.sort(d[start].dt.epoch("us").to_numpy())
    e = np.sort(d[end].dt.epoch("us").to_numpy())
    t = grid_s.dt.epoch("us").to_numpy()
    started = np.searchsorted(s, t, side="right")
    ended = np.searchsorted(e, t, side="right")
    return pl.DataFrame({"t": grid_s, "active": (started - ended).astype(np.int64)})


def runway_configs(
    rwy_hourly: pl.DataFrame, min_count: int = 2, by: Sequence[str] = ("airport", "local_date", "hour_local")
) -> pl.DataFrame:
    """Runway configuration per hour: the sorted set of runways used at least
    ``min_count`` times for DEP and for ARR, joined as 'DEP a,b | ARR c,d'.

    Input: rows of (by..., PHASE_mvt, RUNWAY_mvt, n) as in ``prep.runway_hourly``.
    """

    used = rwy_hourly.filter(pl.col("n") >= min_count)
    sets = (
        used.group_by([*by, "PHASE_mvt"])
        .agg(pl.col("RUNWAY_mvt").sort().str.join(","))
        .pivot(on="PHASE_mvt", index=list(by), values="RUNWAY_mvt")
    )
    for ph in ("DEP", "ARR"):
        if ph not in sets.columns:
            sets = sets.with_columns(pl.lit(None, dtype=pl.String).alias(ph))
    return (
        sets.rename({"DEP": "dep_runways", "ARR": "arr_runways"})
        .with_columns(
            pl.format(
                "DEP {} | ARR {}",
                pl.col("dep_runways").fill_null("-"),
                pl.col("arr_runways").fill_null("-"),
            ).alias("config")
        )
        .sort(list(by))
    )


def config_changes(configs: pl.DataFrame, by: Sequence[str] = ("airport", "local_date")) -> pl.DataFrame:
    """Number of hour-to-hour configuration changes per group (consecutive
    observed hours only; hours with no qualifying movement are skipped)."""

    return (
        configs.sort([*by, "hour_local"])
        .with_columns(
            (pl.col("config") != pl.col("config").shift(1).over(list(by))).fill_null(False).alias("_chg")
        )
        .group_by(list(by))
        .agg(pl.col("_chg").sum().alias("n_changes"), pl.col("config").n_unique().alias("n_configs"))
        .sort(list(by))
    )


def residual_summary(df: pl.DataFrame, target: str, pred: str, by: Sequence[str]) -> pl.DataFrame:
    """n, RMSE, bias (mean target - pred), MAE and sum of squared error per group."""

    r = pl.col(target) - pl.col(pred)
    return (
        df.filter(pl.col(target).is_not_null() & pl.col(pred).is_not_null())
        .group_by(list(by))
        .agg(
            pl.len().alias("n"),
            (r**2).mean().sqrt().alias("rmse"),
            r.mean().alias("bias"),
            r.abs().mean().alias("mae"),
            (r**2).sum().alias("sse"),
        )
        .sort(list(by))
    )


def append_note(path: Path, text: str, context: str = "", now: datetime | None = None) -> str:
    """Append a timestamped note to a markdown file and return the entry."""

    stamp = (now or datetime.now()).strftime("%Y-%m-%d %H:%M")
    header = f"## {stamp}" + (f" ({context})" if context else "")
    entry = f"{header}\n\n{text.strip()}\n\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", encoding="utf-8") as fh:
        if new_file:
            fh.write("# Explorer notes\n\nObservations made in the data explorer, newest last.\n\n")
        fh.write(entry)
    return entry
