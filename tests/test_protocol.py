import numpy as np
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from prc.data import departure_labels, mask_departures
from prc.features.flight import build_features
from prc.validation.protocol import FOLDS, score

T0 = datetime(2025, 1, 31, 20, tzinfo=timezone.utc)


def _movements(offset_block: int = 600) -> pl.DataFrame:
    rows = []
    for i in range(6):
        t = T0 + timedelta(minutes=10 * i)
        rows.append(dict(MVT_ID_mvt=i, FLIGHT_ID_mvt=100 + i, PHASE_mvt="DEP", ADEP_mvt="EDDF", ADES_mvt="LFPG",
                         MVT_TIME_UTC_mvt=t, BLOCK_TIME_UTC_mvt=t - timedelta(seconds=offset_block + i),
                         SCHED_TIME_UTC_mvt=t - timedelta(minutes=20), TAXITIME_SEC_mvt=offset_block + i,
                         RUNWAY_mvt="18", STAND_mvt="A1" if i % 2 else "B2", AIRCRAFT_TYPE_mvt="A320",
                         AOBT_3_flt=t - timedelta(minutes=12), LOBT_flt=t - timedelta(minutes=20),
                         IOBT_flt=t - timedelta(minutes=20), EOBT_1_flt=t - timedelta(minutes=20)))
        rows.append(dict(MVT_ID_mvt=50 + i, FLIGHT_ID_mvt=200 + i, PHASE_mvt="ARR", ADEP_mvt="LFPG", ADES_mvt="EDDF",
                         MVT_TIME_UTC_mvt=t - timedelta(minutes=5), BLOCK_TIME_UTC_mvt=t - timedelta(minutes=1),
                         SCHED_TIME_UTC_mvt=t, TAXITIME_SEC_mvt=240, RUNWAY_mvt="25C", STAND_mvt="A1",
                         AIRCRAFT_TYPE_mvt="A320", AOBT_3_flt=None, LOBT_flt=None, IOBT_flt=None, EOBT_1_flt=None))
    f = pl.DataFrame(rows)
    for c in ["MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt", "FLIGHT_TYPE_flt", "FLIGHT_RULE_flt"]:
        f = f.with_columns(pl.lit("x").alias(c))
    return f.with_columns(pl.lit("AFR123").alias("FLIGHT_mvt"))


def test_features_refuse_unmasked_input():
    with pytest.raises(ValueError):
        build_features(_movements())


def test_features_do_not_depend_on_departure_block_times():
    a = build_features(mask_departures(_movements(600))).sort("MVT_ID_mvt")
    b = build_features(mask_departures(_movements(1500))).sort("MVT_ID_mvt")
    assert a.equals(b)
    assert a.height == 6


def test_labels_and_fill():
    lab = departure_labels(_movements(1200)).sort("MVT_ID_mvt")
    assert lab["target"].to_list()[0] == 1200
    assert lab["fill"].to_list()[0] == 1  # block == schedule (20 min before take-off)


def test_stand_bound_uses_arrival_in_block():
    f = build_features(mask_departures(_movements())).sort("MVT_ID_mvt")
    # arrival at stand A1 blocks in 1 minute before each take-off
    assert f.filter(pl.col("stand") == "A1")["st_gap_last_arr"].to_list() == [60.0, 60.0, 60.0]


def test_folds_have_buffer_and_are_disjoint():
    times = pl.DataFrame({"event_time": [datetime(2025, 1, 15, tzinfo=timezone.utc),
                                         datetime(2025, 2, 2, tzinfo=timezone.utc),
                                         datetime(2025, 2, 10, tzinfo=timezone.utc)]})
    jan = FOLDS["jan"]
    assert times.select(jan.holdout())["event_time"].to_list() == [True, False, False]
    assert times.select(jan.train())["event_time"].to_list() == [False, False, True]


def test_score_clean_and_all():
    f = pl.DataFrame({"target": [100.0, 200.0, 9000.0], "pred": [110.0, 190.0, 1000.0],
                      "no_nm": [0.0, 0.0, 0.0], "airport": ["A", "A", "B"]})
    s = score(f)
    assert s["clean"] == pytest.approx(10.0)
    assert s["all"] == pytest.approx(np.sqrt((100 + 100 + 8000**2) / 3))


def test_proxy_default_features_and_offset():
    from datetime import datetime, timezone

    from prc.features.flight import proxy_default_features
    from prc.models.gbdt import ProxyOffset

    t = datetime(2026, 1, 5, 12, tzinfo=timezone.utc)
    base = pl.DataFrame({"MVT_ID_mvt": [1, 2, 3, 4], "airport": ["LTFM"] * 3 + ["EHAM"],
                         "event_time": [t] * 4, "aobt_proxy_sec": [1500.0, 1500.0, 1800.0, 2000.0]})
    f = base.join(proxy_default_features(base), on="MVT_ID_mvt")
    assert f["px_day_default"].to_list()[:3] == [1500.0, 1500.0, 1500.0]
    assert f["px_dev"].to_list()[:3] == [0.0, 0.0, 300.0]
    train = f.with_columns(pl.Series("target", [950.0, 970.0, 1300.0, 1900.0]),
                           pl.Series("px_day_default_share", [0.9, 0.9, 0.9, 0.05]))
    off = ProxyOffset("aobt_proxy_dev").fit(train)
    assert off.default_airports == {"LTFM"}
    got = off(train)
    # LTFM: median target (970) + deviation from the day's default; EHAM: raw proxy
    assert list(got) == [970.0, 970.0, 1270.0, 2000.0]


def test_nm_window():
    from prc.models.gbdt import NMWindow

    f = pl.DataFrame({"airport": ["A", "A", "A"], "no_nm": [0.0, 0.0, 1.0],
                      "mvt_vs_lobt_sec": [1000.0, 90000.0, None], "mvt_vs_schedule_sec": [800.0, 95000.0, 99999.0],
                      "target": [900.0, 88000.0, 500.0]})
    w = NMWindow().fit(f.head(1))  # offset A = 1000 - 900 = 100
    # schedule within 60 min of LOBT -> fill possible; 5,000 s away -> impossible; unmatched -> possible
    assert list(w.fill_possible(f.with_columns(pl.Series("mvt_vs_schedule_sec", [800.0, 95000.0, 9.0])))) == [True, False, True]
    out = w.apply(f, np.array([900.0, 11718.0, 12345.0]))
    assert out[0] == 900.0 and out[2] == 12345.0     # inside window / unmatched untouched
    assert out[1] == 90000.0 - 3600.0                # outside -> clipped to the nearest window edge
