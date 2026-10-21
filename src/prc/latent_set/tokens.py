"""Event tokens and airport time windows for the joint window model.

Every movement becomes a token at its airport (departures at ADEP, arrivals at ADES). Departure
tokens are built only from a table passed through `prc.data.mask_departures` plus the per-departure
hand features (`prc.features.flight.build_features`, built from the same masked table). Arrivals
keep everything they have in the ranking file, including in-block time and stand.

A window covers 2 h of one airport: a central 30-min slice plus 45 min of context on each side.
Windows step by 30 min, so every departure is central in exactly one window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from prc.features.flight import _assert_masked

CENTRAL = 1800  # seconds
CONTEXT = 2700  # seconds on each side of the central slice
CAT_FIELDS = ["airport", "runway", "stand", "aircraft_type", "operator", "wake", "segment"]
MIN_CAT_COUNT = 20

# Departure numeric inputs taken from the hand-feature table (all masked-safe).
DEP_SECONDS = ["aobt_proxy_sec", "mvt_vs_schedule_sec", "mvt_vs_lobt_sec", "mvt_vs_iobt_sec",
               "mvt_vs_eobt_sec", "aobt_vs_eobt_sec", "rw_gap_prev", "rw_gap_next", "rw_gap_prev5_med",
               "st_gap_last_arr", "st_gap_2nd_arr", "st_gap_prevdep", "st_next_arr_after"]
DEP_COUNTS = ["dep_all_prev300", "dep_all_next300", "arr_all_prev300", "dep_all_prev600", "dep_all_next600",
              "arr_all_prev600", "dep_all_prev1200", "dep_all_next1200", "arr_all_prev1200", "dep_all_prev1800",
              "dep_all_next1800", "arr_all_prev1800", "push_prev600", "push_prev1200", "push_prev1800",
              "dep_between_aobt_to", "push_between_aobt_to", "sched_dep_pm30", "rw_prev600", "rw_prev1800",
              "rw_between_aobt_to", "st_n_arr_since_prevdep"]
ARR_SECONDS = ["arr_taxi_in", "arr_vs_sched", "arr_vs_arvt3", "arr_vs_arvt1"]
SHARED = ["tod_sin", "tod_cos", "week_sin", "week_cos", "no_nm"]
# Hourly weather at the airport (prc.features.flight.load_weather), joined to every token at the hour
# of its reference time. Fixed physical scalings, no standardisation, so that values outside the 2025
# range (the January 2026 EHAM snow) stay proportional instead of being clipped at +-8 sd.
WX_TRANSFORMS = {
    "wx_temp": ("lin", 10.0), "wx_dewdep": ("lin", 10.0), "wx_tmin_6h": ("lin", 10.0),
    "wx_precip": ("log", 1.0), "wx_precip_3h": ("log", 1.0),
    "wx_snowfall": ("log", 1.0), "wx_snow_3h": ("log", 1.0), "wx_snow_12h": ("log", 1.0),
    "wx_snow_depth": ("log", 0.01),  # metres -> log1p(cm)
    "wx_code": ("lin", 100.0), "wx_cloud_low": ("lin", 100.0), "wx_wind": ("lin", 20.0),
    "wx_gust": ("lin", 40.0), "wx_wdir_sin": ("lin", 1.0), "wx_wdir_cos": ("lin", 1.0),
    "wx_icing_3h": ("lin", 3.0), "wx_frozen_6h": ("lin", 6.0),
}


def _slog(x: np.ndarray, scale: float) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x) / scale)


def _pair_id(col: pl.Series) -> np.ndarray:
    h = (col.hash(seed=0) // 2 + 1).cast(pl.Int64)
    return pl.select(pl.when(col.is_null()).then(0).otherwise(h)).to_series().to_numpy().astype(np.int64)


@dataclass
class Encoding:
    """Vocabularies and numeric scaling, fitted on one event table and reused unchanged (ranking)."""

    vocab: dict[str, dict[str, int]] = field(default_factory=dict)
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    num_names: list[str] = field(default_factory=list)
    weather: bool = False
    airline: bool = False  # EXP-045: carrier from the flight number as an extra categorical field

    def cat_fields(self) -> list[str]:
        return CAT_FIELDS + (["airline"] if self.airline else [])


@dataclass
class Events:
    """Event arrays sorted by (airport, t_ref). Index i is one token."""

    mvt_id: np.ndarray     # int64
    is_dep: np.ndarray     # bool
    airport: np.ndarray    # int (airport code index, same as cats[:, 0])
    t_ref: np.ndarray      # int64 epoch s: take-off (DEP) or landing (ARR)
    t_stand: np.ndarray    # int64 epoch s: take-off (DEP) or in-block (ARR)
    cats: np.ndarray       # int32 [n, len(CAT_FIELDS)], 0 = unknown / missing
    num: np.ndarray        # float32 [n, n_num]
    day: np.ndarray        # int32 days since epoch of t_ref (UTC)
    rw_pair: np.ndarray    # int64 hash of airport|runway, 0 = missing (full vocabulary, for pair relations)
    st_pair: np.ndarray    # int64 hash of airport|stand, 0 = missing

    @property
    def n(self) -> int:
        return len(self.mvt_id)


def _raw_event_frame(masked: pl.DataFrame, feats: pl.DataFrame) -> pl.DataFrame:
    """One row per movement with the raw token fields. Departure off-block is never selected."""
    _assert_masked(masked)
    ep = lambda c: pl.col(c).dt.epoch("s")  # noqa: E731
    secs = lambda a, b: (pl.col(a) - pl.col(b)).dt.total_seconds().cast(pl.Float64)  # noqa: E731
    common = [
        pl.col("MVT_ID_mvt"),
        pl.col("RUNWAY_mvt").alias("runway"),
        pl.col("STAND_mvt").alias("stand"),
        pl.col("AIRCRAFT_TYPE_mvt").alias("aircraft_type"),
        pl.col("AIRCRAFT_OPERATOR_flt").alias("operator"),
        pl.col("WK_TBL_CAT_flt").alias("wake"),
        pl.col("MARKET_SEGMENT_flt").alias("segment"),
        # carrier from the flight number (NM's operator is anonymised and empty without an NM match)
        pl.coalesce(pl.col("FLIGHT_mvt").str.extract(r"^([A-Z]{3})", 1),
                    pl.col("FLIGHT_mvt").str.extract(r"^([A-Z0-9]{2})[A-Z]?\d", 1)).alias("airline"),
        ep("MVT_TIME_UTC_mvt").alias("t_ref"),
        pl.col("FLIGHT_ID_mvt").is_null().cast(pl.Float64).alias("no_nm"),
        ((pl.col("MVT_TIME_UTC_mvt").dt.hour() * 60 + pl.col("MVT_TIME_UTC_mvt").dt.minute()) / 1440 * 2 * np.pi)
        .alias("_tod"),
        (pl.col("MVT_TIME_UTC_mvt").dt.weekday() / 7 * 2 * np.pi).alias("_wd"),
    ]
    dep = masked.filter(pl.col("PHASE_mvt") == "DEP").select(
        *common, pl.col("ADEP_mvt").alias("airport"), ep("MVT_TIME_UTC_mvt").alias("t_stand"),
        pl.lit(True).alias("is_dep"))
    hand = feats.select("MVT_ID_mvt", *DEP_SECONDS, *DEP_COUNTS)
    dep = dep.join(hand, on="MVT_ID_mvt", how="left")
    arr = masked.filter(pl.col("PHASE_mvt") == "ARR").select(
        *common, pl.col("ADES_mvt").alias("airport"),
        pl.coalesce(ep("BLOCK_TIME_UTC_mvt"), ep("MVT_TIME_UTC_mvt") + 600).alias("t_stand"),
        pl.lit(False).alias("is_dep"),
        secs("BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt").alias("arr_taxi_in"),
        secs("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("arr_vs_sched"),
        secs("MVT_TIME_UTC_mvt", "ARVT_3_flt").alias("arr_vs_arvt3"),
        secs("MVT_TIME_UTC_mvt", "ARVT_1_flt").alias("arr_vs_arvt1"),
    )
    ev = pl.concat([dep, arr], how="diagonal_relaxed")
    ev = ev.with_columns(pl.col("_tod").sin().alias("tod_sin"), pl.col("_tod").cos().alias("tod_cos"),
                         pl.col("_wd").sin().alias("week_sin"), pl.col("_wd").cos().alias("week_cos"))
    # Prefix runway and stand with the airport: the same name at two airports is two places.
    ev = ev.with_columns(
        pl.when(pl.col(c).is_not_null()).then(pl.col("airport") + "|" + pl.col(c)).alias(c) for c in ("runway", "stand"))
    return ev.filter(pl.col("airport").is_not_null() & pl.col("t_ref").is_not_null()).sort(["airport", "t_ref"])


def build_events(masked: pl.DataFrame, feats: pl.DataFrame, enc: Encoding | None = None,
                 weather: pl.DataFrame | None = None, airline: bool = False) -> tuple[Events, Encoding]:
    """Token arrays for every movement of a masked table. Fits `enc` if not given.

    weather: hourly table from `prc.features.flight.load_weather` (airport, wx_hour, wx_*). When the
    encoding was fitted with weather, it is required."""
    ev = _raw_event_frame(masked, feats)
    fit = enc is None
    enc = enc or Encoding(weather=weather is not None, airline=airline)
    if enc.weather and weather is None:
        raise ValueError("this encoding expects weather inputs")
    fields = enc.cat_fields()
    cats = np.zeros((ev.height, len(fields)), dtype=np.int32)
    for k, c in enumerate(fields):
        col = ev[c]
        if fit:
            vc = col.drop_nulls().value_counts()
            keep = vc.filter(pl.col("count") >= (1 if c == "airport" else MIN_CAT_COUNT))[c].sort().to_list()
            enc.vocab[c] = {v: i + 1 for i, v in enumerate(keep)}
        m = enc.vocab[c]
        cats[:, k] = col.replace_strict(m, default=0, return_dtype=pl.Int32).fill_null(0).to_numpy()

    blocks, names = [], []
    for c in DEP_SECONDS + ARR_SECONDS:
        blocks.append(_slog(ev[c].cast(pl.Float64).to_numpy(), 60.0)); names.append(c)
    for c in DEP_COUNTS:
        blocks.append(np.log1p(np.clip(ev[c].cast(pl.Float64).to_numpy(), 0, None))); names.append(c)
    raw = np.stack(blocks, 1)
    miss = np.isnan(raw)
    if fit:
        enc.mean = np.nanmean(raw, 0)
        enc.std = np.nanstd(raw, 0) + 1e-6
        enc.num_names = names + SHARED + [f"miss_{n}" for n in names]
    z = np.where(miss, 0.0, (raw - enc.mean) / enc.std)
    shared = np.stack([ev[c].cast(pl.Float64).to_numpy() for c in SHARED], 1)
    parts = [z, shared, miss.astype(np.float64)]
    if enc.weather:
        wx = ev.select("airport", pl.from_epoch("t_ref", "s").dt.replace_time_zone("UTC").dt.truncate("1h")
                       .alias("wx_hour")).with_row_index("_i")
        wx = wx.join(weather.with_columns(pl.col("wx_hour").cast(wx.schema["wx_hour"])), on=["airport", "wx_hour"],
                     how="left").sort("_i")
        assert wx.height == ev.height, "duplicate (airport, hour) rows in the weather table"
        cols = []
        for c, (kind, scale) in WX_TRANSFORMS.items():
            x = wx[c].cast(pl.Float64).to_numpy()
            cols.append(np.log1p(np.clip(x, 0, None) / scale) if kind == "log" else x / scale)
        wxa = np.stack(cols, 1)
        wmiss = np.isnan(wxa).any(1, keepdims=True)
        parts += [np.nan_to_num(wxa), wmiss.astype(np.float64)]
        if fit:
            enc.num_names += list(WX_TRANSFORMS) + ["miss_wx"]
    num = np.concatenate(parts, 1).astype(np.float32)

    t_ref = ev["t_ref"].to_numpy().astype(np.int64)
    events = Events(
        mvt_id=ev["MVT_ID_mvt"].to_numpy().astype(np.int64),
        is_dep=ev["is_dep"].to_numpy().astype(bool),
        airport=cats[:, 0].copy(),
        t_ref=t_ref,
        t_stand=ev["t_stand"].to_numpy().astype(np.int64),
        cats=cats,
        num=num,
        day=(t_ref // 86400).astype(np.int32),
        rw_pair=_pair_id(ev["runway"]),
        st_pair=_pair_id(ev["stand"]),
    )
    return events, enc


@dataclass
class Windows:
    lo: np.ndarray        # first token index (into Events)
    hi: np.ndarray        # one past last token index
    c0: np.ndarray        # central slice start, epoch s
    airport: np.ndarray

    @property
    def n(self) -> int:
        return len(self.lo)


def subset_events(ev: Events, keep: np.ndarray) -> Events:
    """Events restricted to a boolean mask (order preserved)."""
    return Events(**{f: getattr(ev, f)[keep] for f in Events.__dataclass_fields__})


def concat_events(a: Events, b: Events) -> Events:
    """Union of two event tables (e.g. 2025 train and 2026 ranking), re-sorted by (airport, t_ref),
    so windows run across the file boundary exactly as inside a file."""
    cat = {f: np.concatenate([getattr(a, f), getattr(b, f)]) for f in Events.__dataclass_fields__}
    order = np.lexsort((cat["t_ref"], cat["airport"]))
    return Events(**{f: v[order] for f, v in cat.items()})


def build_windows(ev: Events, max_tokens: int = 320, context: int = CONTEXT) -> Windows:
    """All windows with at least one central departure. Oversized windows are trimmed symmetrically
    around the central slice (the central slice itself is never trimmed)."""
    los, his, c0s, aps = [], [], [], []
    for ap in np.unique(ev.airport):
        idx = np.where(ev.airport == ap)[0]
        a0, a1 = idx[0], idx[-1] + 1
        t = ev.t_ref[a0:a1]
        dep_t = t[ev.is_dep[a0:a1]]
        ks = np.unique(dep_t // CENTRAL)
        c0 = ks * CENTRAL
        lo = np.searchsorted(t, c0 - context, "left")
        hi = np.searchsorted(t, c0 + CENTRAL + context, "left")
        clo = np.searchsorted(t, c0, "left")
        chi = np.searchsorted(t, c0 + CENTRAL, "left")
        over = (hi - lo) > max_tokens
        if over.any():
            spare = np.maximum(max_tokens - (chi - clo), 0)
            left = np.minimum(clo - lo, spare // 2)
            right = np.minimum(hi - chi, spare - left)
            left = np.minimum(clo - lo, spare - right)
            lo = np.where(over, clo - left, lo)
            hi = np.where(over, chi + right, hi)
        los.append(lo + a0); his.append(hi + a0); c0s.append(c0); aps.append(np.full(len(c0), ap))
    return Windows(np.concatenate(los), np.concatenate(his), np.concatenate(c0s), np.concatenate(aps))


def central_mask(ev: Events, win: Windows, w: int, idx: np.ndarray) -> np.ndarray:
    t = ev.t_ref[idx]
    return (t >= win.c0[w]) & (t < win.c0[w] + CENTRAL) & ev.is_dep[idx]
