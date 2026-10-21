"""Constants and paths shared by the explorer cache builder and the app."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw"
CACHE_DIR = REPO_ROOT / "data" / "processed" / "explorer"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
NOTES_PATH = REPO_ROOT / "docs" / "explorer_notes.md"

AIRPORTS: tuple[str, ...] = (
    "EDDF",
    "EDDM",
    "EGLL",
    "EHAM",
    "LEBL",
    "LEMD",
    "LFPG",
    "LIRF",
    "LSZH",
    "LTFM",
)

AIRPORT_TZ: dict[str, str] = {
    "EDDF": "Europe/Berlin",
    "EDDM": "Europe/Berlin",
    "EGLL": "Europe/London",
    "EHAM": "Europe/Amsterdam",
    "LEBL": "Europe/Madrid",
    "LEMD": "Europe/Madrid",
    "LFPG": "Europe/Paris",
    "LIRF": "Europe/Rome",
    "LSZH": "Europe/Zurich",
    "LTFM": "Europe/Istanbul",
}

DATASETS: tuple[str, ...] = ("train", "ranking")

# Proxies of the off-block time that exist in the ranking file for departures.
# Each proxy value is MVT_TIME_UTC_mvt (take-off) minus the reference time, in
# seconds, so it is directly comparable to the target TAXITIME_SEC_mvt.
PROXIES: dict[str, str] = {
    "px_aobt3": "AOBT_3_flt",
    "px_eobt1": "EOBT_1_flt",
    "px_lobt": "LOBT_flt",
    "px_iobt": "IOBT_flt",
    "px_sched": "SCHED_TIME_UTC_mvt",
}

PROXY_LABELS: dict[str, str] = {
    "px_aobt3": "take-off minus AOBT_3_flt (NM actual off-block)",
    "px_eobt1": "take-off minus EOBT_1_flt (NM filed estimate)",
    "px_lobt": "take-off minus LOBT_flt (NM last known off-block)",
    "px_iobt": "take-off minus IOBT_flt (NM initial off-block)",
    "px_sched": "take-off minus SCHED_TIME_UTC_mvt (airport schedule)",
}

TIME_COLUMNS: tuple[str, ...] = (
    "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "LOBT_flt",
    "IOBT_flt",
    "EOBT_1_flt",
    "ARVT_1_flt",
    "AOBT_3_flt",
    "ARVT_3_flt",
)

QUALITY_FLAGS: dict[str, str] = {
    "q_taxi_le0": "taxi time <= 0 s",
    "q_taxi_gt1h": "taxi time > 3600 s",
    "q_taxi_gt2h": "taxi time > 7200 s",
    "q_block_order": "block time on wrong side of movement time (DEP: off-block after take-off; ARR: in-block before landing)",
    "q_dup_mvt_id": "MVT_ID_mvt appears more than once",
    "q_dup_flight_day": "same airport, phase, FLIGHT_mvt and local date more than once",
    "q_dup_nm_flight": "same FLIGHT_ID_mvt matched to more than one movement of the same phase and airport",
    "q_no_nm_match": "FLIGHT_ID_mvt is null (no NM flight matched)",
    "q_airport_mismatch": "airport code differs between movement and NM (DEP: ADEP, ARR: ADES)",
    "q_diverted": "ADES_flt differs from ADES_FILED_flt (diversion)",
    "q_type_mismatch": "AIRCRAFT_TYPE_mvt differs from AIRCRAFT_TYPE_flt",
    "q_runway_malformed": "RUNWAY_mvt does not look like NN[LCR] (e.g. 'NA', 'H')",
    "q_stand_malformed": "STAND_mvt null, 'UNKNOWN', or without any digit",
    "q_aobt_far": "DEP: |BLOCK_TIME - AOBT_3| > 900 s",
    "q_aobt_after_takeoff": "DEP: AOBT_3_flt later than take-off",
}


def configure_threads(n: int = 12) -> None:
    """Cap polars / BLAS threads. Must run before polars is imported."""

    for var in (
        "POLARS_MAX_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        os.environ.setdefault(var, str(n))
    os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
