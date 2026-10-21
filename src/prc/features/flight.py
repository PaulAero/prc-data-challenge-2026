"""Per-departure features built from a masked movement table (see `prc.data.mask_departures`).

Blocks:
- base: flight attributes, calendar, official NM / schedule time differences.
- seq: runway take-off sequence and airport traffic around the take-off (Idris-style queue).
- stand: stand occupancy, from arrivals' in-block times at the same stand.
- arr: arrivals' taxi-in times around the take-off (not blanked in ranking), an observed measure of
  surface congestion.
- wx: hourly weather at the airport (Open-Meteo / ERA5, CC BY 4.0, data/external/weather), with
  trailing aggregates aimed at de-icing and winter disruption.
"""

from __future__ import annotations

from math import pi

import numpy as np
import polars as pl

from prc.features.geometry import geometry_features

CATEGORICAL = ["airport", "destination", "runway", "stand", "aircraft_type", "market_segment",
               "wake_category", "operator", "flight_type", "flight_rule", "airline"]
# `airline`: leading letters of the flight number. AIRCRAFT_OPERATOR_flt is anonymised and empty for
# flights without an NM match, where the schedule-fill rate depends strongly on the carrier (EXP-028).
SCHED_REL = ["aobt_vs_sched_sec", "eobt_vs_sched_sec", "lobt_vs_sched_sec", "abs_aobt_vs_sched_sec"]
BASE_NUMERIC = ["aobt_proxy_sec", "mvt_vs_schedule_sec", "mvt_vs_lobt_sec", "mvt_vs_iobt_sec",
                "mvt_vs_eobt_sec", "aobt_vs_eobt_sec", "tod_sin", "tod_cos", "week_sin", "week_cos",
                "month", "no_nm", "local_hour"]

LOCAL_TZ = {"EDDF": "Europe/Berlin", "EDDM": "Europe/Berlin", "EGLL": "Europe/London",
            "EHAM": "Europe/Amsterdam", "LEBL": "Europe/Madrid", "LEMD": "Europe/Madrid",
            "LFPG": "Europe/Paris", "LIRF": "Europe/Rome", "LSZH": "Europe/Zurich",
            "LTFM": "Europe/Istanbul"}


def _assert_masked(movements: pl.DataFrame) -> None:
    dep = movements.filter(pl.col("PHASE_mvt") == "DEP")
    if dep["BLOCK_TIME_UTC_mvt"].null_count() != dep.height or dep["TAXITIME_SEC_mvt"].null_count() != dep.height:
        raise ValueError("feature builders only accept movements passed through mask_departures()")


def _secs(a: str, b: str, alias: str) -> pl.Expr:
    return (pl.col(a) - pl.col(b)).dt.total_seconds().cast(pl.Float32).alias(alias)


def base_features(movements: pl.DataFrame) -> pl.DataFrame:
    dep = movements.filter(pl.col("PHASE_mvt") == "DEP")
    local_hour = pl.lit(None, dtype=pl.Int8)
    for ap, tz in LOCAL_TZ.items():
        local_hour = pl.when(pl.col("ADEP_mvt") == ap).then(
            pl.col("MVT_TIME_UTC_mvt").dt.convert_time_zone(tz).dt.hour().cast(pl.Int8)).otherwise(local_hour)
    frame = dep.select(
        "MVT_ID_mvt",
        pl.col("MVT_TIME_UTC_mvt").alias("event_time"),
        pl.col("ADEP_mvt").alias("airport"),
        pl.col("ADES_mvt").alias("destination"),
        pl.col("RUNWAY_mvt").alias("runway"),
        pl.col("STAND_mvt").alias("stand"),
        pl.col("AIRCRAFT_TYPE_mvt").alias("aircraft_type"),
        pl.col("MARKET_SEGMENT_flt").alias("market_segment"),
        pl.col("WK_TBL_CAT_flt").alias("wake_category"),
        pl.col("AIRCRAFT_OPERATOR_flt").alias("operator"),
        pl.col("FLIGHT_TYPE_flt").alias("flight_type"),
        pl.col("FLIGHT_RULE_flt").alias("flight_rule"),
        # ICAO 3-letter prefix (AFR123, also garbled WMTT9066 -> WMT), else IATA 2-character prefix,
        # which may contain a digit (4Y514, U2123)
        pl.coalesce(pl.col("FLIGHT_mvt").str.extract(r"^([A-Z]{3})", 1),
                    pl.col("FLIGHT_mvt").str.extract(r"^([A-Z0-9]{2})[A-Z]?\d", 1)).alias("airline"),
        # record-quality flags: garbled or missing flight number, missing aircraft type (EXP-028)
        (~pl.col("FLIGHT_mvt").str.contains(r"^([A-Z]{3}\d{1,4}[A-Z]{0,2}|[A-Z0-9]{2}\d{1,4}[A-Z]?)$").fill_null(False))
        .cast(pl.Float32).alias("flight_no_odd"),
        pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Float32).alias("actype_missing"),
        _secs("MVT_TIME_UTC_mvt", "AOBT_3_flt", "aobt_proxy_sec"),
        _secs("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "mvt_vs_schedule_sec"),
        _secs("MVT_TIME_UTC_mvt", "LOBT_flt", "mvt_vs_lobt_sec"),
        _secs("MVT_TIME_UTC_mvt", "IOBT_flt", "mvt_vs_iobt_sec"),
        _secs("MVT_TIME_UTC_mvt", "EOBT_1_flt", "mvt_vs_eobt_sec"),
        _secs("AOBT_3_flt", "EOBT_1_flt", "aobt_vs_eobt_sec"),
        # NM times relative to the airport schedule: schedule-filled off-blocks concentrate where the
        # NM off-block is close to the schedule (EXP-015)
        _secs("AOBT_3_flt", "SCHED_TIME_UTC_mvt", "aobt_vs_sched_sec"),
        _secs("EOBT_1_flt", "SCHED_TIME_UTC_mvt", "eobt_vs_sched_sec"),
        _secs("LOBT_flt", "SCHED_TIME_UTC_mvt", "lobt_vs_sched_sec"),
        (pl.col("AOBT_3_flt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs().cast(pl.Float32)
        .alias("abs_aobt_vs_sched_sec"),
        (pl.col("MVT_TIME_UTC_mvt").dt.hour() * 60 + pl.col("MVT_TIME_UTC_mvt").dt.minute()).alias("_mod"),
        pl.col("MVT_TIME_UTC_mvt").dt.weekday().alias("_wd"),
        pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Float32).alias("month"),
        pl.col("FLIGHT_ID_mvt").is_null().cast(pl.Float32).alias("no_nm"),
        local_hour.cast(pl.Float32).alias("local_hour"),
    )
    return frame.with_columns(
        (2 * pi * pl.col("_mod") / 1440).sin().cast(pl.Float32).alias("tod_sin"),
        (2 * pi * pl.col("_mod") / 1440).cos().cast(pl.Float32).alias("tod_cos"),
        (2 * pi * pl.col("_wd") / 7).sin().cast(pl.Float32).alias("week_sin"),
        (2 * pi * pl.col("_wd") / 7).cos().cast(pl.Float32).alias("week_cos"),
        pl.col(CATEGORICAL).fill_null("<NA>"),
    ).drop("_mod", "_wd")


def _cnt(sorted_times: np.ndarray, lo, hi) -> np.ndarray:
    return np.searchsorted(sorted_times, hi, "left") - np.searchsorted(sorted_times, lo, "left")


def _epoch(col: str) -> pl.Expr:
    return pl.col(col).dt.epoch("s")


def sequence_features(movements: pl.DataFrame) -> pl.DataFrame:
    dep = movements.filter(pl.col("PHASE_mvt") == "DEP").with_columns(
        _epoch("MVT_TIME_UTC_mvt").alias("t"), _epoch("AOBT_3_flt").alias("a"), _epoch("SCHED_TIME_UTC_mvt").alias("s"))
    arr = movements.filter(pl.col("PHASE_mvt") == "ARR").with_columns(_epoch("MVT_TIME_UTC_mvt").alias("t"))
    out = []
    for (ap,), g in dep.group_by(["ADEP_mvt"]):
        g = g.sort("t")
        t = g["t"].to_numpy()
        a = g["a"].to_numpy().astype(float)
        s = g["s"].to_numpy()
        at = np.sort(arr.filter(pl.col("ADES_mvt") == ap)["t"].to_numpy())
        asort = np.sort(a[~np.isnan(a)])
        f: dict[str, np.ndarray] = {"MVT_ID_mvt": g["MVT_ID_mvt"].to_numpy()}
        for w in (300, 600, 1200, 1800):
            f[f"dep_all_prev{w}"] = _cnt(t, t - w, t)
            f[f"dep_all_next{w}"] = _cnt(t, t + 1, t + w + 1)
            f[f"arr_all_prev{w}"] = _cnt(at, t - w, t)
        for w in (600, 1200, 1800):
            f[f"push_prev{w}"] = _cnt(asort, t - w, t)
        an = np.where(np.isnan(a), t - 900, a)
        f["dep_between_aobt_to"] = np.where(np.isnan(a), np.nan, _cnt(t, an, t))
        f["push_between_aobt_to"] = np.where(np.isnan(a), np.nan, _cnt(asort, an, t))
        f["sched_dep_pm30"] = _cnt(np.sort(s), t - 1800, t + 1800)
        rw = g["RUNWAY_mvt"].fill_null("NA").to_numpy()
        for k in ["rw_prev600", "rw_prev1800", "rw_gap_prev", "rw_gap_next", "rw_gap_prev5_med", "rw_between_aobt_to"]:
            f[k] = np.full(len(t), np.nan)
        for r in np.unique(rw):
            idx = np.where(rw == r)[0]
            tr = t[idx]
            f["rw_prev600"][idx] = _cnt(tr, tr - 600, tr)
            f["rw_prev1800"][idx] = _cnt(tr, tr - 1800, tr)
            gp = np.diff(tr, prepend=np.nan)
            f["rw_gap_prev"][idx] = gp
            f["rw_gap_next"][idx] = np.append(np.diff(tr), np.nan)
            f["rw_gap_prev5_med"][idx] = pl.Series(gp).rolling_median(5, min_samples=1).to_numpy()
            f["rw_between_aobt_to"][idx] = np.where(np.isnan(a[idx]), np.nan, _cnt(tr, an[idx], tr))
        out.append(pl.DataFrame(f))
    seq = pl.concat(out, how="vertical_relaxed")
    return seq.with_columns([pl.col(c).cast(pl.Float32) for c in seq.columns if c != "MVT_ID_mvt"])


def stand_features(movements: pl.DataFrame) -> pl.DataFrame:
    """Gaps between my take-off and the latest arrivals' in-block at my stand.

    If an arrival blocks in at my stand after I pushed back and before I took off, take-off minus
    that in-block is a hard lower bound on my taxi-out.
    """
    dep = movements.filter(pl.col("PHASE_mvt") == "DEP").with_columns(_epoch("MVT_TIME_UTC_mvt").alias("t"))
    arr = movements.filter(pl.col("PHASE_mvt") == "ARR").with_columns(_epoch("BLOCK_TIME_UTC_mvt").alias("u"))
    names = ["st_gap_last_arr", "st_gap_2nd_arr", "st_n_arr_since_prevdep", "st_gap_prevdep", "st_next_arr_after"]
    out = []
    for (ap,), g in dep.group_by(["ADEP_mvt"]):
        A = arr.filter(pl.col("ADES_mvt") == ap).drop_nulls(["STAND_mvt", "u"])
        amap = {s: np.sort(x["u"].to_numpy()) for (s,), x in A.group_by(["STAND_mvt"])}
        f = {k: np.full(g.height, np.nan) for k in names}
        f["MVT_ID_mvt"] = g["MVT_ID_mvt"].to_numpy()
        gs = g.with_row_index("i").drop_nulls("STAND_mvt")
        for (s,), x in gs.group_by(["STAND_mvt"]):
            idx = x["i"].to_numpy()
            t = x["t"].to_numpy()
            o = np.argsort(t)
            idx, t = idx[o], t[o]
            u = amap.get(s)
            prevdep = np.concatenate([[np.nan], t[:-1]]).astype(float)
            f["st_gap_prevdep"][idx] = t - prevdep
            if u is None or len(u) == 0:
                continue
            k = np.searchsorted(u, t, "left")
            f["st_gap_last_arr"][idx] = np.where(k >= 1, t - u[np.maximum(k - 1, 0)], np.nan)
            f["st_gap_2nd_arr"][idx] = np.where(k >= 2, t - u[np.maximum(k - 2, 0)], np.nan)
            f["st_next_arr_after"][idx] = np.where(k < len(u), u[np.minimum(k, len(u) - 1)] - t, np.nan)
            kp = np.searchsorted(u, np.nan_to_num(prevdep, nan=-1), "right")
            f["st_n_arr_since_prevdep"][idx] = np.where(np.isnan(prevdep), np.nan, k - kp)
        out.append(pl.DataFrame(f))
    st = pl.concat(out, how="vertical_relaxed")
    return st.with_columns([pl.col(c).cast(pl.Float32) for c in names])


def arrival_taxi_features(movements: pl.DataFrame) -> pl.DataFrame:
    """Taxi-in of arrivals near my take-off: raw mean, and excess over each arrival's stand x runway median."""
    dep = movements.filter(pl.col("PHASE_mvt") == "DEP").with_columns(_epoch("MVT_TIME_UTC_mvt").alias("t"))
    arr = movements.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("TAXITIME_SEC_mvt").is_between(0, 7200)).with_columns(
        _epoch("MVT_TIME_UTC_mvt").alias("t"), _epoch("BLOCK_TIME_UTC_mvt").alias("u"),
        pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("ti"))
    # excess taxi-in over the typical value for that stand x runway (computed within the same table;
    # arrivals' taxi-in is an input on ranking too, so this is not label leakage)
    arr = arr.with_columns((pl.col("ti") - pl.col("ti").median().over("ADES_mvt", "STAND_mvt", "RUNWAY_mvt")).alias("xs"))
    names = []
    out = []
    for (ap,), g in dep.group_by(["ADEP_mvt"]):
        a = arr.filter(pl.col("ADES_mvt") == ap).sort("t")
        at, ti, xs = a["t"].to_numpy(), a["ti"].to_numpy(), a["xs"].to_numpy()
        cti, cxs = np.concatenate([[0.0], np.cumsum(ti)]), np.concatenate([[0.0], np.cumsum(xs)])
        t = g["t"].to_numpy()
        f: dict[str, np.ndarray] = {"MVT_ID_mvt": g["MVT_ID_mvt"].to_numpy()}
        for w in (900, 1800, 3600):
            lo, hi = np.searchsorted(at, t - w, "left"), np.searchsorted(at, t + 300, "right")
            n = hi - lo
            f[f"arr_ti_mean_{w}"] = np.where(n > 0, (cti[hi] - cti[lo]) / np.maximum(n, 1), np.nan)
            f[f"arr_ti_excess_{w}"] = np.where(n > 0, (cxs[hi] - cxs[lo]) / np.maximum(n, 1), np.nan)
        # arrivals taxiing in (landed, not yet in-block) at my take-off
        us = np.sort(a["u"].to_numpy())
        f["arr_taxiing_at_to"] = np.searchsorted(at, t, "right") - np.searchsorted(us, t, "right")
        names = [k for k in f if k != "MVT_ID_mvt"]
        out.append(pl.DataFrame(f))
    res = pl.concat(out, how="vertical_relaxed")
    return res.with_columns([pl.col(c).cast(pl.Float32) for c in names])


WEATHER_DIR = "data/external/weather"


def _openmeteo_hourly(path: str) -> pl.DataFrame | None:
    import glob as _glob

    files = sorted(_glob.glob(f"{path}/openmeteo_*.parquet"))
    if not files:
        return None
    return pl.concat([pl.read_parquet(f) for f in files]).sort("airport", "time")


def _cds_hourly(path: str = "data/external/weather_cds") -> pl.DataFrame | None:
    """Official ERA5 / ERA5-Land point time series mapped to the Open-Meteo column names.

    ERA5 has no snowfall or weather code in these datasets: snowfall is precipitation (mm water)
    falling at 2 m temperature <= 1 C, and the weather code is a coarse WMO-like class (0 dry,
    61 rain, 71 snow)."""
    import glob as _glob

    files = sorted(_glob.glob(f"{path}/era5_*.parquet"))
    if not files:
        return None
    w = pl.concat([pl.read_parquet(f) for f in files]).sort("airport", "valid_time")
    t = pl.col("t2m") - 273.15
    d = pl.col("d2m") - 273.15
    precip = (pl.col("tp") * 1000).clip(0, None)
    rh = 100 * ((17.625 * d) / (243.04 + d)).exp() / ((17.625 * t) / (243.04 + t)).exp()
    snow = pl.when(t <= 1).then(precip).otherwise(0.0)
    return w.select(
        "airport", pl.col("valid_time").alias("time"),
        t.alias("temperature_2m"), d.alias("dew_point_2m"), rh.clip(0, 100).alias("relative_humidity_2m"),
        precip.alias("precipitation"), pl.when(t > 1).then(precip).otherwise(0.0).alias("rain"),
        snow.alias("snowfall"), pl.col("sde").fill_null(0.0).alias("snow_depth"),
        pl.when(precip < 0.1).then(0).when(t <= 1).then(71).otherwise(61).alias("weather_code"),
        (pl.col("tcc") * 100).alias("cloud_cover_low"),
        ((pl.col("u10") ** 2 + pl.col("v10") ** 2).sqrt() * 3.6).alias("wind_speed_10m"),
        ((180 + pl.arctan2(pl.col("u10"), pl.col("v10")) * 180 / pi) % 360).alias("wind_direction_10m"),
        (pl.col("fg10") * 3.6).alias("wind_gusts_10m"), (pl.col("sp") / 100).alias("surface_pressure"),
    )


def load_weather(path: str = WEATHER_DIR) -> pl.DataFrame | None:
    """Hourly weather features. Source: env PRC_WEATHER = "openmeteo" (default) or "cds" (official
    Copernicus ERA5 / ERA5-Land, planned for the final model)."""
    import os

    src = os.environ.get("PRC_WEATHER", "openmeteo")
    w = _cds_hourly() if src == "cds" else _openmeteo_hourly(path)
    if w is None:
        return None
    icing = ((pl.col("temperature_2m") <= 3) & ((pl.col("precipitation") > 0) | (pl.col("relative_humidity_2m") >= 95)))
    frozen = (pl.col("snowfall") > 0) | pl.col("weather_code").is_in([56, 57, 66, 67, 71, 73, 75, 77, 85, 86])
    w = w.with_columns(icing.cast(pl.Float32).alias("_icing"), frozen.cast(pl.Float32).alias("_frozen"))
    roll = lambda c, n, f: getattr(pl.col(c), f"rolling_{f}")(n, min_samples=1).over("airport")  # noqa: E731
    return w.select(
        "airport", pl.col("time").alias("wx_hour"),
        pl.col("temperature_2m").cast(pl.Float32).alias("wx_temp"),
        (pl.col("temperature_2m") - pl.col("dew_point_2m")).cast(pl.Float32).alias("wx_dewdep"),
        pl.col("precipitation").cast(pl.Float32).alias("wx_precip"),
        pl.col("snowfall").cast(pl.Float32).alias("wx_snowfall"),
        pl.col("snow_depth").cast(pl.Float32).alias("wx_snow_depth"),
        pl.col("weather_code").cast(pl.Float32).alias("wx_code"),
        pl.col("cloud_cover_low").cast(pl.Float32).alias("wx_cloud_low"),
        pl.col("wind_speed_10m").cast(pl.Float32).alias("wx_wind"),
        pl.col("wind_gusts_10m").cast(pl.Float32).alias("wx_gust"),
        (pl.col("wind_direction_10m") * pi / 180).sin().cast(pl.Float32).alias("wx_wdir_sin"),
        (pl.col("wind_direction_10m") * pi / 180).cos().cast(pl.Float32).alias("wx_wdir_cos"),
        roll("snowfall", 3, "sum").cast(pl.Float32).alias("wx_snow_3h"),
        roll("snowfall", 12, "sum").cast(pl.Float32).alias("wx_snow_12h"),
        roll("precipitation", 3, "sum").cast(pl.Float32).alias("wx_precip_3h"),
        roll("temperature_2m", 6, "min").cast(pl.Float32).alias("wx_tmin_6h"),
        roll("_icing", 3, "sum").cast(pl.Float32).alias("wx_icing_3h"),
        roll("_frozen", 6, "sum").cast(pl.Float32).alias("wx_frozen_6h"),
    )


def weather_features(masked: pl.DataFrame, weather: pl.DataFrame) -> pl.DataFrame:
    dep = masked.filter(pl.col("PHASE_mvt") == "DEP").select(
        "MVT_ID_mvt", pl.col("ADEP_mvt").alias("airport"), pl.col("MVT_TIME_UTC_mvt").dt.truncate("1h").alias("wx_hour"))
    return dep.join(weather, on=["airport", "wx_hour"], how="left").drop("airport", "wx_hour")


def proxy_default_features(base: pl.DataFrame) -> pl.DataFrame:
    """AOBT_3 relative to the airport-day default (EXP-030).

    At some airports NM's AOBT_3 is mostly take-off minus a default taxi time whose value changes
    between periods (LTFM: 1,200 / 1,440 s in 2025, 1,500 s in January 2026) while actual taxi-out
    does not. The day's modal value is computable in the ranking file; the deviation from it carries
    the flight-specific information.
    """
    px10 = (pl.col("aobt_proxy_sec") / 10).round(0)
    day = pl.col("event_time").dt.date()
    f = base.select("MVT_ID_mvt", "airport", "aobt_proxy_sec", day.alias("_d"), px10.alias("_px10"))
    counts = f.drop_nulls("_px10").group_by("airport", "_d", "_px10").len()
    total = counts.group_by("airport", "_d").agg(pl.col("len").sum().alias("_n"))
    # most frequent value per airport-day, ties broken by the smallest value (deterministic)
    modes = (counts.sort(["airport", "_d", "len", "_px10"], descending=[False, False, True, False])
             .group_by("airport", "_d", maintain_order=True).first()
             .join(total, on=["airport", "_d"])
             .select("airport", "_d", (pl.col("_px10") * 10).cast(pl.Float32).alias("px_day_default"),
                     (pl.col("len") / pl.col("_n")).cast(pl.Float32).alias("px_day_default_share")))
    return f.join(modes, on=["airport", "_d"], how="left").select(
        "MVT_ID_mvt", "px_day_default", "px_day_default_share",
        (pl.col("aobt_proxy_sec") - pl.col("px_day_default")).alias("px_dev"),
        ((pl.col("aobt_proxy_sec") - pl.col("px_day_default")).abs() <= 10).cast(pl.Float32).alias("px_is_default"))


def recovered_nm_features(masked: pl.DataFrame) -> pl.DataFrame:
    """NM record of a no-NM departure recovered from its arrival row at the destination, when the
    destination is one of the ten airports (EXP-035): same flight number, landing 0-14 h after the
    take-off; the earliest such arrival. Label-free. Returns take-off minus recovered LOBT."""
    dep = masked.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("FLIGHT_ID_mvt").is_null() & pl.col("FLIGHT_mvt").is_not_null()).select(
        "MVT_ID_mvt", "FLIGHT_mvt", "ADES_mvt", pl.col("MVT_TIME_UTC_mvt").alias("t"))
    arr = masked.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("LOBT_flt").is_not_null()).select(
        "FLIGHT_mvt", pl.col("ADES_mvt"), pl.col("MVT_TIME_UTC_mvt").alias("land"), pl.col("LOBT_flt").alias("rec_lobt"))
    j = dep.join(arr, on=["FLIGHT_mvt", "ADES_mvt"], how="inner").filter(
        (pl.col("land") > pl.col("t")) & (pl.col("land") <= pl.col("t") + pl.duration(hours=14)))
    j = j.sort("MVT_ID_mvt", "land").group_by("MVT_ID_mvt", maintain_order=True).first()
    return j.select("MVT_ID_mvt", (pl.col("t") - pl.col("rec_lobt")).dt.total_seconds().cast(pl.Float32).alias("rec_to_minus_lobt"))


def build_features(masked: pl.DataFrame) -> pl.DataFrame:
    """All feature blocks for every departure of a masked movement table."""
    _assert_masked(masked)
    base = base_features(masked)
    return (
        base.join(proxy_default_features(base), on="MVT_ID_mvt", how="left")
        .join(sequence_features(masked), on="MVT_ID_mvt", how="left")
        .join(stand_features(masked), on="MVT_ID_mvt", how="left")
        .join(arrival_taxi_features(masked), on="MVT_ID_mvt", how="left")
        .pipe(lambda f: f if (w := load_weather()) is None
              else f.join(weather_features(masked, w), on="MVT_ID_mvt", how="left"))
        .pipe(lambda f: f if (g := geometry_features(masked)) is None
              else f.join(g, on="MVT_ID_mvt", how="left"))
        .join(recovered_nm_features(masked), on="MVT_ID_mvt", how="left")
        .sort("MVT_ID_mvt")  # fixed row order: LightGBM row subsampling depends on it
    )


def block_columns(frame: pl.DataFrame) -> dict[str, list[str]]:
    seq = [c for c in frame.columns if c.startswith(("dep_all_", "arr_all_", "push_", "dep_between", "sched_dep", "rw_"))]
    stand = [c for c in frame.columns if c.startswith("st_")]
    arr = [c for c in frame.columns if c.startswith(("arr_ti_", "arr_taxiing"))]
    base = [c for c in CATEGORICAL if c != "airline"] + BASE_NUMERIC
    return {"base": base, "airline": ["airline", "flight_no_odd", "actype_missing"], "seq": seq, "stand": stand, "arr": arr, "schedrel": SCHED_REL,
            "wx": [c for c in frame.columns if c.startswith("wx_")],
            "geo": [c for c in frame.columns if c.startswith("geo_")],
            "pxdev": ["px_day_default", "px_day_default_share", "px_dev", "px_is_default"]}
