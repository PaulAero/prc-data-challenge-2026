from datetime import datetime, timezone

import polars as pl

from prc.timeseries.airport_state import _contiguous_date_ranges, build_airport_state


UTC = timezone.utc


def test_contiguous_date_ranges_splits_ranking_gap() -> None:
    dates = [
        datetime(2026, 1, 1).date(),
        datetime(2026, 1, 2).date(),
        datetime(2026, 7, 1).date(),
    ]
    assert _contiguous_date_ranges(dates) == [
        (dates[0], dates[1]),
        (dates[2], dates[2]),
    ]


def test_build_airport_state_keeps_target_separate(tmp_path) -> None:
    source = tmp_path / "movements.parquet"
    pl.DataFrame(
        {
            "ADEP_mvt": ["LFPG", "EHAM", "LFPG"],
            "ADES_mvt": ["EHAM", "LFPG", "EGLL"],
            "PHASE_mvt": ["DEP", "ARR", "DEP"],
            "MVT_TIME_UTC_mvt": [
                datetime(2025, 1, 1, 8, 1, tzinfo=UTC),
                datetime(2025, 1, 1, 8, 4, tzinfo=UTC),
                datetime(2025, 1, 1, 8, 16, tzinfo=UTC),
            ],
            "SCHED_TIME_UTC_mvt": [
                datetime(2025, 1, 1, 8, 2, tzinfo=UTC),
                datetime(2025, 1, 1, 8, 0, tzinfo=UTC),
                datetime(2025, 1, 1, 8, 17, tzinfo=UTC),
            ],
            "TAXITIME_SEC_mvt": [600, 420, None],
        },
        schema_overrides={"TAXITIME_SEC_mvt": pl.Int32},
    ).write_parquet(source)

    state = build_airport_state([source], interval="15m")
    lfpg = state.filter(
        (pl.col("airport") == "LFPG")
        & (pl.col("timestamp") == datetime(2025, 1, 1, 8, 0, tzinfo=UTC))
    ).row(0, named=True)

    assert lfpg["departure_count"] == 1
    assert lfpg["arrival_count"] == 1
    assert lfpg["scheduled_departure_count"] == 1
    assert lfpg["taxi_out_mean_sec"] == 600

    missing_target = state.filter(
        (pl.col("airport") == "LFPG")
        & (pl.col("timestamp") == datetime(2025, 1, 1, 8, 15, tzinfo=UTC))
    ).row(0, named=True)
    assert missing_target["departure_count"] == 1
    assert missing_target["taxi_out_mean_sec"] is None
