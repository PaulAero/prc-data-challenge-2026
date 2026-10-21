"""View 6: replay one airport-day: departures as off-block -> take-off bars, arrivals, queues."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import plotly.graph_objects as go
import polars as pl
import streamlit as st
from plotly.subplots import make_subplots

import ui
from prc.explore import analysis
from prc.explore.common import AIRPORT_TZ, AIRPORTS

COLS = (
    "dataset", "PHASE_mvt", "MVT_ID_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt",
    "AIRCRAFT_OPERATOR_flt", "RUNWAY_mvt", "STAND_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt", ui.TARGET, "px_aobt3",
)
TIME_COLS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt"]
LOCAL = {c: c.replace("_UTC_mvt", "").replace("_flt", "").lower() + "_local" for c in TIME_COLS}


@st.cache_data(show_spinner=False)
def runway_colors(airport: str) -> dict[str, str]:
    """Fixed colors per runway for an airport, by overall usage rank (top 8), grey beyond."""

    rw = ui.table("runway_hourly").filter(pl.col("airport") == airport).group_by("RUNWAY_mvt").agg(
        pl.col("n").sum()).sort("n", descending=True)
    return {r: (ui.PALETTE[i] if i < len(ui.PALETTE) else ui.OTHER_COLOR) for i, r in enumerate(rw["RUNWAY_mvt"])}


def load_day(airport: str, day: date) -> pl.DataFrame:
    ds = "ranking" if day.year == 2026 else "train"
    d = ui.movements((ds,), (airport,), ("DEP", "ARR"), (), (day, day), COLS)
    tz = AIRPORT_TZ[airport]
    return d.with_columns(
        [pl.col(c).dt.convert_time_zone(tz).dt.replace_time_zone(None).alias(LOCAL[c]) for c in TIME_COLS]
    )


def _gantt(dep: pl.DataFrame, colors: dict[str, str], has_target: bool, sort_by: str, show_aobt: bool,
           show_eobt: bool, title: str) -> go.Figure:
    key = "block_time_local" if (has_target and sort_by == "off-block") else "mvt_time_local"
    dep = dep.sort(key, nulls_last=True).with_row_index("row")
    fig = go.Figure()
    for rwy in sorted(dep["RUNWAY_mvt"].unique().to_list(), key=str):
        s = dep.filter(pl.col("RUNWAY_mvt") == rwy)
        color = colors.get(rwy, ui.OTHER_COLOR)
        hover = [
            f"{fl} ({ty}) stand {st_} rwy {rwy}<br>off-block {b}<br>take-off {m}<br>taxi {t} s<br>AOBT_3 {a}"
            for fl, ty, st_, b, m, t, a in zip(
                s["FLIGHT_mvt"], s["AIRCRAFT_TYPE_mvt"], s["STAND_mvt"], s["block_time_local"],
                s["mvt_time_local"], s[ui.TARGET], s["aobt_3_local"])
        ]
        if has_target:
            xs, ys = [], []
            for b, m, r in zip(s["block_time_local"], s["mvt_time_local"], s["row"]):
                xs += [b, m, None]
                ys += [r, r, None]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=2),
                                     name=f"{rwy} taxi-out", legendgroup=rwy, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=s["mvt_time_local"], y=s["row"], mode="markers", name=f"{rwy} take-off",
                                 legendgroup=rwy, marker=dict(color=color, size=8, line=dict(color="white", width=1)),
                                 text=hover, hovertemplate="%{text}<extra></extra>"))
    if show_aobt:
        fig.add_trace(go.Scatter(x=dep["aobt_3_local"], y=dep["row"], mode="markers", name="AOBT_3 (NM)",
                                 marker=dict(symbol="x-thin", size=8, line=dict(color="#0b0b0b", width=1.5)),
                                 hovertemplate="AOBT_3 %{x}<extra></extra>"))
    if show_eobt:
        fig.add_trace(go.Scatter(x=dep["eobt_1_local"], y=dep["row"], mode="markers", name="EOBT_1 (NM)",
                                 marker=dict(symbol="line-ns", size=8, line=dict(color="#52514e", width=1.5)),
                                 hovertemplate="EOBT_1 %{x}<extra></extra>"))
    fig.update_layout(title=title, height=max(500, min(2400, 4 * dep.height + 200)),
                      xaxis_title="local time", yaxis=dict(title=f"departures sorted by {sort_by}", autorange="reversed",
                                                           showticklabels=False),
                      legend=dict(orientation="v", x=1.01, y=1, title="runway"))
    return fig


def render(f: ui.Filters) -> None:
    st.header("6. Airport-day replay")
    st.markdown(
        "One airport, one local day. Train days show each departure as a bar from off-block to take-off "
        "(the target). Ranking days only have take-off plus the NM AOBT_3 / EOBT_1 markers, because the "
        "off-block time is blanked. All times are airport local time."
    )
    c1, c2, c3 = st.columns([1, 1, 2])
    if "rp_ap" not in st.session_state:
        st.session_state["rp_ap"] = f.airports[0]
    if "rp_date" not in st.session_state:
        st.session_state["rp_date"] = date(2025, 7, 15)
    ap = c1.selectbox("Airport", list(AIRPORTS), key="rp_ap")
    day = c2.date_input("Local date", min_value=date(2025, 1, 1), max_value=date(2026, 8, 1), key="rp_date")
    window = c3.slider("Local hours shown", 0, 24, (5, 12), key="rp_win")
    d = load_day(ap, day)
    if d.height == 0:
        st.info("No movements for this airport and local date (ranking covers only January and July 2026).")
        return
    has_target = d["dataset"][0] == "train"
    t0 = datetime.combine(day, datetime.min.time()) + timedelta(hours=window[0])
    t1 = datetime.combine(day, datetime.min.time()) + timedelta(hours=window[1])
    colors = runway_colors(ap)

    dep_all = d.filter(pl.col("PHASE_mvt") == "DEP")
    arr_all = d.filter(pl.col("PHASE_mvt") == "ARR")
    dep = dep_all.filter(pl.col("mvt_time_local").is_between(t0, t1))
    arr = arr_all.filter(pl.col("mvt_time_local").is_between(t0, t1))
    st.caption(f"{d['dataset'][0]} | {ap} {day} | window {t0:%H:%M}-{t1:%H:%M} local | "
               f"{dep.height} of {dep_all.height} departures, {arr.height} of {arr_all.height} arrivals in window"
               + ("" if has_target else " | ranking: BLOCK_TIME and target are null for all departures"))

    c1, c2, c3 = st.columns(3)
    sort_by = c1.radio("Sort departures by", ["off-block", "take-off"] if has_target else ["take-off"],
                       horizontal=True, key="rp_sort")
    show_aobt = c2.checkbox("Show AOBT_3 markers", value=not has_target, key="rp_aobt")
    show_eobt = c3.checkbox("Show EOBT_1 markers", value=not has_target, key="rp_eobt")
    title = (f"{ap} {day}: departures, off-block to take-off (bars, coloured by runway), n={dep.height}"
             if has_target else f"{ap} {day}: departures, take-off markers by runway, n={dep.height}")
    ui.show(_gantt(dep, colors, has_target, sort_by, show_aobt, show_eobt, title))

    # Queue panel: taxiing-out count, take-offs and landings per 15 min per runway.
    rows = 3
    titles = ["aircraft taxiing out at t, 1-min grid: true (off-block <= t < take-off) and NM proxy "
              "(AOBT_3 <= t < take-off)", "take-offs per 15 min, by runway", "landings per 15 min, by runway"]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.07, subplot_titles=titles)
    grid = pl.datetime_range(t0, t1, "1m", eager=True)
    if has_target:
        act = analysis.active_count(dep_all, "block_time_local", "mvt_time_local", grid)
        fig.add_trace(go.Scatter(x=act["t"], y=act["active"], mode="lines", line=dict(color=ui.PALETTE[0], width=2),
                                 name="taxiing out (true)"), row=1, col=1)
    prox = analysis.active_count(dep_all, "aobt_3_local", "mvt_time_local", grid)
    fig.add_trace(go.Scatter(x=prox["t"], y=prox["active"], mode="lines",
                             line=dict(color=ui.PALETTE[1], width=2, dash="dot"),
                             name="AOBT_3 to take-off (NM proxy)"), row=1, col=1)
    fig.update_yaxes(title_text="aircraft", row=1, col=1)
    r = 2
    shown: set[str] = set()
    for frame, row_i, label in [(dep, r, "take-offs"), (arr, r + 1, "landings")]:
        b = frame.with_columns(pl.col("mvt_time_local").dt.truncate("15m").alias("slot")).group_by(
            "slot", "RUNWAY_mvt").len().sort("slot")
        for rwy in sorted(b["RUNWAY_mvt"].unique().to_list(), key=str):
            s = b.filter(pl.col("RUNWAY_mvt") == rwy)
            fig.add_trace(go.Bar(x=s["slot"] + timedelta(minutes=7.5), y=s["len"], name=f"{rwy}",
                                 legendgroup=rwy, marker_color=colors.get(rwy, ui.OTHER_COLOR),
                                 showlegend=rwy not in shown, width=15 * 60 * 1000 * 0.9,
                                 hovertemplate=f"{rwy} {label}: %{{y}}<extra></extra>"), row=row_i, col=1)
            shown.add(rwy)
        fig.update_yaxes(title_text=label, row=row_i, col=1)
    fig.update_layout(barmode="stack", height=220 * rows + 80, title=f"{ap} {day}: queue and runway activity")
    fig.update_xaxes(range=[t0, t1], title_text="local time", row=rows, col=1)
    ui.show(fig)

    # Arrivals by runway
    fig = go.Figure()
    for rwy in sorted(arr["RUNWAY_mvt"].unique().to_list(), key=str):
        s = arr.filter(pl.col("RUNWAY_mvt") == rwy)
        fig.add_trace(go.Scatter(x=s["mvt_time_local"], y=[rwy] * s.height, mode="markers", name=rwy,
                                 marker=dict(color=colors.get(rwy, ui.OTHER_COLOR), size=8, symbol="triangle-down"),
                                 text=s["FLIGHT_mvt"], hovertemplate="%{text} landed %{x}<extra></extra>"))
    fig.update_layout(title=f"{ap} {day}: landings per runway (n={arr.height})", xaxis_title="local time",
                      yaxis_title="runway", height=260, xaxis=dict(range=[t0, t1]))
    ui.show(fig)

    st.subheader("Flights in window")
    ph = st.radio("Phase", ["DEP", "ARR"], horizontal=True, key="rp_tab_ph")
    tab = (dep if ph == "DEP" else arr).sort("mvt_time_local").select(
        "FLIGHT_mvt", "CALLSIGN_flt", "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "RUNWAY_mvt", "STAND_mvt",
        "sched_time_local", "eobt_1_local", "aobt_3_local", "block_time_local", "mvt_time_local", ui.TARGET,
        "px_aobt3", "ADEP_mvt", "ADES_mvt", "MVT_ID_mvt")
    st.dataframe(tab, hide_index=True, width="stretch", height=400)
