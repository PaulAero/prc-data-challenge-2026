from datetime import date, datetime, timezone

import polars as pl
import pytest

from prc.explore import analysis, prep
from prc.explore.common import CACHE_DIR, QUALITY_FLAGS, REPO_ROOT

UTC = timezone.utc


def _raw() -> pl.DataFrame:
    t = lambda h, m, s=0: datetime(2025, 7, 1, h, m, s, tzinfo=UTC)  # noqa: E731
    return pl.DataFrame(
        {
            "MVT_ID_mvt": [1.0, 2.0, 3.0, 4.0],
            "FLIGHT_ID_mvt": [10.0, None, 30.0, 30.0],
            "FLIGHT_mvt": ["AB1", "AB2", "AB3", "AB3"],
            "PHASE_mvt": ["DEP", "DEP", "ARR", "DEP"],
            "ADEP_mvt": ["LTFM", "EGLL", "EHAM", "LTFM"],
            "ADES_mvt": ["EGLL", "EHAM", "LTFM", "EGLL"],
            "MVT_TIME_UTC_mvt": [t(10, 20), t(10, 0), t(11, 0), t(12, 0)],
            "BLOCK_TIME_UTC_mvt": [t(10, 5), t(10, 5), t(11, 10), t(11, 30)],
            "SCHED_TIME_UTC_mvt": [t(10, 0), t(10, 0), t(11, 0), t(11, 30)],
            "AOBT_3_flt": [t(10, 4), None, None, t(11, 0)],
            "EOBT_1_flt": [t(10, 0), None, None, t(11, 30)],
            "LOBT_flt": [t(10, 0), None, None, t(11, 30)],
            "IOBT_flt": [t(10, 0), None, None, t(11, 30)],
            "TAXITIME_SEC_mvt": [900, -300, 600, 1800],
            "ADEP_flt": ["LTFM", None, "EHAM", "LTFM"],
            "ADES_flt": ["EGLL", None, "LTFM", "EGLL"],
            "ADES_FILED_flt": ["EGLL", None, "LTFM", "LIRF"],
            "AIRCRAFT_TYPE_mvt": ["A320", "A320", "B738", "A321"],
            "AIRCRAFT_TYPE_flt": ["A320", None, "B737", "A321"],
            "RUNWAY_mvt": ["35R", "NA", "18C", "35R"],
            "STAND_mvt": ["A1", "UNKNOWN", "B2", "C3"],
        }
    )


def _movements() -> pl.DataFrame:
    lf = prep.derive_movements(_raw().lazy(), "train")
    return prep.add_quality_flags(lf).collect()


def test_derive_movements_airport_local_time_and_proxies() -> None:
    d = _movements()
    assert d["airport"].to_list() == ["LTFM", "EGLL", "LTFM", "LTFM"]
    # Istanbul is UTC+3 all year, London is UTC+1 in July.
    assert d["hour_local"].to_list() == [13, 11, 14, 15]
    assert d["local_date"][0] == date(2025, 7, 1)
    assert d["month"][0] == "2025-07"
    assert d["px_aobt3"][0] == 16 * 60  # take-off 10:20 minus AOBT 10:04
    assert d["px_sched"][0] == 20 * 60
    assert d["px_aobt3"][2] is None  # arrivals get no departure proxies
    assert d["blk_minus_aobt3"][3] == 30 * 60
    assert d["blk_minus_sched"][1] == 5 * 60


def test_quality_flags() -> None:
    d = _movements()
    assert set(QUALITY_FLAGS).issubset(d.columns)
    assert d["q_taxi_le0"].to_list() == [False, True, False, False]
    assert d["q_block_order"].to_list() == [False, True, False, False]
    assert d["q_no_nm_match"].to_list() == [False, True, False, False]
    assert d["q_dup_nm_flight"].to_list() == [False, False, False, False]  # different phases
    assert d["q_diverted"].to_list() == [False, False, False, True]
    assert d["q_type_mismatch"].to_list() == [False, False, True, False]
    assert d["q_runway_malformed"].to_list() == [False, True, False, False]
    assert d["q_stand_malformed"].to_list() == [False, True, False, False]
    assert d["q_aobt_far"].to_list() == [False, False, False, True]


def test_null_rates_and_rounding() -> None:
    lf = _movements().lazy()
    nr = prep.null_rates(lf, ["AOBT_3_flt", "RUNWAY_mvt"])
    row = nr.filter((pl.col("airport") == "EGLL") & (pl.col("column") == "AOBT_3_flt"))
    assert row["null_rate"][0] == 1.0
    na = nr.filter((pl.col("airport") == "EGLL") & (pl.col("column") == "RUNWAY_mvt"))
    assert na["na_string_rate"][0] == 1.0 and na["null_rate"][0] == 0.0
    rt = prep.rounding_table(lf, ["MVT_TIME_UTC_mvt"])
    assert set(rt.columns) >= {"sec0", "sec_mod5", "sec_mid", "n"}
    assert rt["sec0"].min() == 1.0 and rt["sec_mid"].max() == 0.0


def test_tail_dominance() -> None:
    df = pl.DataFrame({"g": ["a"] * 100, "y": [0.0] * 99 + [100.0]})
    t = analysis.tail_dominance(df, "y", ["g"], fracs=(0.01,))
    # mean = 1; the outlier's squared deviation is 99^2 out of 99^2 + 99 * 1
    assert t["share_sq_top_1pct"][0] == pytest.approx(99**2 / (99**2 + 99))
    assert t["rmse_without_top_1pct"][0] == pytest.approx(1.0)


def test_proxy_summary() -> None:
    df = pl.DataFrame({"g": ["a"] * 4, "y": [100.0, 200.0, 300.0, 400.0], "p": [90.0, 190.0, None, 390.0]})
    s = analysis.proxy_summary(df, "y", ["p"], ["g"])
    assert s["availability"][0] == 0.75
    assert s["rmse_raw"][0] == pytest.approx(10.0)
    assert s["bias"][0] == pytest.approx(10.0)
    assert s["rmse_debiased"][0] == pytest.approx(0.0)
    assert s["within_60s"][0] == 1.0


def test_histogram_keeps_overflow() -> None:
    df = pl.DataFrame({"g": ["a"] * 5, "v": [-5.0, 0.0, 15.0, 29.9, 31.0]})
    h = analysis.histogram(df, "v", ["g"], 0, 30, 10)
    assert h["count"].sum() == 5
    assert h.filter(pl.col("zone") == "below")["count"].sum() == 1
    assert h.filter(pl.col("zone") == "above")["count"].sum() == 1
    assert h.filter(pl.col("zone") == "in_range")["bin_left"].to_list() == [0.0, 10.0, 20.0]


def test_active_count() -> None:
    t = lambda m: datetime(2025, 1, 1, 10, m)  # noqa: E731
    df = pl.DataFrame({"s": [t(0), t(5), t(10)], "e": [t(10), t(15), t(12)]})
    grid = pl.datetime_range(t(0), t(15), "5m", eager=True)
    a = analysis.active_count(df, "s", "e", grid)
    # at 10:00 -> 1, 10:05 -> 2, 10:10 -> 2 (first ended, third started), 10:15 -> 0
    assert a["active"].to_list() == [1, 2, 2, 0]


def test_runway_configs_and_changes() -> None:
    rw = pl.DataFrame({
        "airport": ["X"] * 6, "local_date": [date(2025, 1, 1)] * 6, "hour_local": [6, 6, 6, 7, 7, 8],
        "PHASE_mvt": ["DEP", "DEP", "ARR", "DEP", "ARR", "DEP"],
        "RUNWAY_mvt": ["25C", "18", "25L", "25C", "25L", "07C"], "n": [5, 1, 4, 3, 3, 6],
    })
    cfg = analysis.runway_configs(rw, min_count=2)
    assert cfg["config"].to_list() == ["DEP 25C | ARR 25L", "DEP 25C | ARR 25L", "DEP 07C | ARR -"]
    ch = analysis.config_changes(cfg)
    assert ch["n_changes"][0] == 1 and ch["n_configs"][0] == 2


def test_unseen_share_and_residuals() -> None:
    ref = pl.DataFrame({"c": ["a", "b", None]})
    cand = pl.DataFrame({"c": ["a", "c", "c", None]})
    u = analysis.unseen_share(ref, cand, "c")
    assert u["rows_unseen_share"] == 0.5 and u["distinct_unseen"] == 1
    df = pl.DataFrame({"g": ["a", "a"], "y": [10.0, 20.0], "p": [10.0, 10.0]})
    r = analysis.residual_summary(df, "y", "p", ["g"])
    assert r["rmse"][0] == pytest.approx((50.0) ** 0.5) and r["bias"][0] == 5.0 and r["sse"][0] == 100.0


def test_append_note(tmp_path) -> None:
    path = tmp_path / "notes.md"
    analysis.append_note(path, "first", "ctx", now=datetime(2026, 9, 26, 10, 0))
    analysis.append_note(path, "second", now=datetime(2026, 9, 26, 11, 0))
    text = path.read_text()
    assert text.startswith("# Explorer notes")
    assert "## 2026-09-26 10:00 (ctx)\n\nfirst" in text and "## 2026-09-26 11:00\n\nsecond" in text


@pytest.mark.skipif(not (CACHE_DIR / "manifest.json").exists(), reason="explorer cache not built")
def test_app_smoke_all_views() -> None:
    """Every view renders without an exception on the real cache (needs the viz group)."""

    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(REPO_ROOT / "apps" / "explorer" / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception
    for view in at.sidebar.radio(key="view").options:
        at.sidebar.radio(key="view").set_value(view).run()
        assert not at.exception, (view, [e.value for e in at.exception])
