"""PRC Data Challenge 2026: local data explorer (Streamlit).

Launch:  uv run --group viz streamlit run apps/explorer/app.py
Needs the cache: uv run python scripts/explore/build_explorer_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for p in (str(_ROOT / "src"), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from prc.explore.common import AIRPORTS, configure_threads  # noqa: E402

configure_threads(12)

import streamlit as st  # noqa: E402

import ui  # noqa: E402
import views  # noqa: E402

st.set_page_config(page_title="PRC taxi-out explorer", layout="wide")

VIEWS = {
    "1. Inventory and data quality": views.quality.render,
    "2. What the model predicts": views.target.render,
    "3. Train vs ranking shift": views.shift.render,
    "4. Relationships": views.relations.render,
    "5. Temporal patterns": views.temporal.render,
    "6. Airport-day replay": views.replay.render,
    "7. Model error explorer": views.errors.render,
    "8. Runway configurations": views.runways.render,
    "9. Hypotheses notepad": views.notes.render,
}


def sidebar() -> ui.Filters:
    sb = st.sidebar
    sb.title("Taxi-out explorer")
    sb.radio("View", list(VIEWS), key="view")
    sb.divider()
    sb.caption("Global filters (each view states which ones it uses)")
    datasets = sb.multiselect("Dataset", ["train", "ranking"], default=["train"], key="f_datasets")
    airports = sb.multiselect("Airports", list(AIRPORTS), default=list(AIRPORTS), key="f_airports")
    phases = sb.multiselect("Phase", ["DEP", "ARR"], default=["DEP"], key="f_phases")
    months = sb.multiselect("Months (UTC, empty = all)", ui.month_options(), default=[], key="f_months")
    use_dates = sb.checkbox("Restrict local date range", value=False, key="f_use_dates")
    date_range = None
    if use_dates:
        daily = ui.table("daily")
        lo, hi = daily["local_date"].min(), daily["local_date"].max()
        picked = sb.date_input("Local dates", value=(lo, hi), min_value=lo, max_value=hi, key="f_dates")
        if isinstance(picked, tuple) and len(picked) == 2:
            date_range = (picked[0], picked[1])
    man = ui.manifest()
    sb.caption(f"Cache built {man['built_at_utc']} UTC. Local only, no telemetry.")
    return ui.Filters(
        datasets=tuple(datasets) or ("train",),
        airports=tuple(airports) or tuple(AIRPORTS),
        phases=tuple(phases) or ("DEP",),
        months=tuple(months),
        date_range=date_range,
    )


def main() -> None:
    if not ui.cache_ready():
        st.error(
            "Explorer cache missing. Build it first:\n\n"
            "`uv run python scripts/explore/build_explorer_cache.py`"
        )
        st.stop()
    filters = sidebar()
    VIEWS[st.session_state["view"]](filters)


main()
