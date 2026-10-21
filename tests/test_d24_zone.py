"""D24 zone rule (EXP-039): which departures can carry the 24-hour date artefact."""
from datetime import datetime, timezone

import numpy as np
import polars as pl

from prc.models.gbdt import D24Zone


def _frame(rows):
    return pl.DataFrame(
        {
            "event_time": [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t, *_ in rows],
            "mvt_vs_schedule_sec": [float(ms) for _, ms, _, _ in rows],
            "airport": [ap for *_, ap, _ in rows],
            "no_nm": [float(n) for *_, n in rows],
        }
    )


def test_zone_and_night():
    rows = [
        # LIRF (UTC+2 in July): scheduled 20:30 local on 1 July, off-block about 12:30 local on 2 July
        ("2026-07-02T10:45:00", 16 * 3600 + 15 * 60, "LIRF", 1),
        # night crossing: scheduled 23:30 local, off-block about 00:20 local next day
        ("2026-07-01T22:35:00", 65 * 60, "LIRF", 1),
        # same day: no date change
        ("2026-07-01T12:00:00", 3600, "LIRF", 1),
        # matched flight and another airport: never in the zone
        ("2026-07-02T10:45:00", 16 * 3600 + 15 * 60, "LIRF", 0),
        ("2026-07-02T10:45:00", 16 * 3600 + 15 * 60, "LFPG", 1),
    ]
    z, night = D24Zone().zones(_frame(rows), np.full(len(rows), 900.0))
    assert z.tolist() == [True, False, False, False, False]
    assert night.tolist() == [False, True, False, False, False]
