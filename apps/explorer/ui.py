"""Cached loaders and shared UI helpers for the explorer views."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from prc.explore.common import AIRPORTS, CACHE_DIR, EXPERIMENTS_DIR

MOVEMENTS = CACHE_DIR / "movements.parquet"

# Reference categorical palette (dataviz skill, light mode), fixed order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER_COLOR = "#9a9893"
SEQ_SCALE = [[0, "#f3f7fd"], [0.25, "#9ec5f4"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1, "#0d366b"]]
DIV_SCALE = [[0, "#2a78d6"], [0.5, "#f0efec"], [1, "#e34948"]]
DATASET_COLOR = {"train": PALETTE[0], "ranking": PALETTE[1]}
TARGET = "TAXITIME_SEC_mvt"


def cache_ready() -> bool:
    return MOVEMENTS.exists() and (CACHE_DIR / "manifest.json").exists()


@st.cache_data(show_spinner=False)
def manifest() -> dict:
    return json.loads((CACHE_DIR / "manifest.json").read_text())


@st.cache_data(show_spinner=False)
def table(name: str) -> pl.DataFrame:
    """A precomputed aggregate from data/processed/explorer/<name>.parquet."""

    return pl.read_parquet(CACHE_DIR / f"{name}.parquet")


def scan() -> pl.LazyFrame:
    return pl.scan_parquet(MOVEMENTS)


# cache_resource, not cache_data: these frames reach millions of rows and
# cache_data would pickle a copy on every hit. Polars frames are never mutated
# in place by the views, so sharing one object is safe.
@st.cache_resource(show_spinner="Loading movements...", max_entries=24)
def movements(
    datasets: tuple[str, ...],
    airports: tuple[str, ...],
    phases: tuple[str, ...],
    months: tuple[str, ...] = (),
    date_range: tuple[date, date] | None = None,
    columns: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Filtered flight-level rows, loaded lazily from the movements cache."""

    lf = scan().filter(
        pl.col("dataset").is_in(list(datasets))
        & pl.col("airport").is_in(list(airports))
        & pl.col("PHASE_mvt").is_in(list(phases))
    )
    if months:
        lf = lf.filter(pl.col("month").is_in(list(months)))
    if date_range is not None:
        lf = lf.filter(pl.col("local_date").is_between(date_range[0], date_range[1]))
    if columns is not None:
        lf = lf.select(list(columns))
    return lf.collect()


@st.cache_data(show_spinner=False)
def month_options() -> list[str]:
    return table("hourly").get_column("month").unique().sort().to_list()


@st.cache_data(show_spinner=False)
def experiments() -> dict[str, str]:
    """Experiments whose predictions.parquet has the per-flight columns."""

    need = {"MVT_ID_mvt", "target", "baseline_prediction", "latent_prediction"}
    out: dict[str, str] = {}
    for p in sorted(EXPERIMENTS_DIR.glob("exp-*/predictions.parquet")):
        if need.issubset(pl.read_parquet_schema(p)):
            out[p.parent.name] = str(p)
    return out


@dataclass
class Filters:
    datasets: tuple[str, ...]
    airports: tuple[str, ...]
    phases: tuple[str, ...]
    months: tuple[str, ...]
    date_range: tuple[date, date] | None
    notes: list[str] = field(default_factory=list)

    def describe(self, used: tuple[str, ...] = ("datasets", "airports", "phases", "months", "dates")) -> str:
        parts = []
        if "datasets" in used:
            parts.append("dataset=" + ",".join(self.datasets))
        if "airports" in used:
            parts.append(
                "airports=" + ("all 10" if len(self.airports) == len(AIRPORTS) else ",".join(self.airports))
            )
        if "phases" in used:
            parts.append("phase=" + ",".join(self.phases))
        if "months" in used:
            parts.append("months=" + (",".join(self.months) if self.months else "all"))
        if "dates" in used and self.date_range:
            parts.append(f"local dates {self.date_range[0]}..{self.date_range[1]}")
        return "; ".join(parts)


def caption_filters(f: Filters, used: tuple[str, ...], extra: str = "") -> None:
    st.caption("Filters applied: " + f.describe(used) + (f"; {extra}" if extra else ""))


def show(fig: go.Figure, height: int | None = None) -> None:
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch")


def color_map(values: list[str], max_colors: int = 8) -> dict[str, str]:
    """Fixed-order colors for values (sorted), folding the tail into grey."""

    out = {}
    for i, v in enumerate(sorted(v for v in values if v is not None)):
        out[v] = PALETTE[i] if i < max_colors else OTHER_COLOR
    return out


def fmt_n(n: int) -> str:
    return f"{n:,}".replace(",", " ")


REPLAY_VIEW = "6. Airport-day replay"


def jump_to_replay(airport: str, day: date) -> None:
    """Button callback: open the replay view on an airport-day."""

    st.session_state["view"] = REPLAY_VIEW
    st.session_state["rp_ap"] = airport
    st.session_state["rp_date"] = day
