"""View 5: daily taxi-out over 2025 with the ranking months, and one day's intraday curve."""

from __future__ import annotations

from datetime import date

import plotly.graph_objects as go
import polars as pl
import streamlit as st
from plotly.subplots import make_subplots

import ui


def render(f: ui.Filters) -> None:
    st.header("5. Temporal patterns")
    st.markdown("Daily taxi-out level over 2025 (train) and traffic in both years. Ranking months "
                "(Jan and Jul 2026) have no target: only their traffic is shown. Sidebar airports apply.")
    ap = st.selectbox("Airport", list(f.airports), key="tp_ap")
    daily = ui.table("daily").filter(pl.col("airport") == ap).sort("local_date")
    tr = daily.filter(pl.col("dataset") == "train")
    rk = daily.filter(pl.col("dataset") == "ranking")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4],
                        subplot_titles=[f"{ap}: daily taxi-out (s), train 2025, by local date of take-off",
                                        f"{ap}: departures per local day (train and ranking)"])
    fig.add_trace(go.Scatter(x=tr["local_date"], y=tr["taxi_mean"], name="daily mean", mode="lines",
                             line=dict(color=ui.PALETTE[0], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=tr["local_date"], y=tr["taxi_p90"], name="daily p90", mode="lines",
                             line=dict(color=ui.PALETTE[1], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=tr["local_date"], y=tr["n_dep"], name="departures, train", mode="lines",
                             line=dict(color=ui.PALETTE[2], width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=rk["local_date"], y=rk["n_dep"], name="departures, ranking (no target)",
                             mode="lines+markers", marker=dict(size=4), line=dict(color=ui.PALETTE[6], width=2)),
                  row=2, col=1)
    for start, end in [(date(2026, 1, 1), date(2026, 1, 31)), (date(2026, 7, 1), date(2026, 7, 31))]:
        fig.add_vrect(x0=start, x1=end, fillcolor="#9a9893", opacity=0.15, line_width=0,
                      annotation_text="ranking: no target", annotation_position="top left", row=1, col=1)
    fig.update_yaxes(title_text="seconds", row=1, col=1)
    fig.update_yaxes(title_text="departures", row=2, col=1)
    fig.update_layout(height=620, hovermode="x unified")
    ui.show(fig)
    st.caption(f"n = {ui.fmt_n(int(tr['n_target'].sum()))} train departures with a target over "
               f"{tr.height} days; {ui.fmt_n(int(rk['n_dep'].sum()))} ranking departures over {rk.height} days.")

    worst = tr.sort("taxi_mean", descending=True).head(15).select(
        "local_date", "n_dep", "taxi_mean", "taxi_median", "taxi_p90")
    with st.expander("Days with the highest daily mean taxi-out"):
        st.dataframe(worst, hide_index=True, width="stretch")

    st.subheader("Intraday curve for one day")
    lo, hi = daily["local_date"].min(), daily["local_date"].max()
    day = st.date_input("Local date", value=date(2025, 7, 15), min_value=lo, max_value=hi, key="tp_day")
    ds = "ranking" if day.year == 2026 else "train"
    one = ui.movements((ds,), (ap,), ("DEP", "ARR"), (), (day, day),
                       ("PHASE_mvt", "mvt_local", ui.TARGET, "px_aobt3"))
    if one.height == 0:
        st.info("No movements that day.")
        return
    b = one.with_columns(pl.col("mvt_local").dt.truncate("15m").alias("slot")).group_by("slot", "PHASE_mvt").agg(
        pl.len().alias("n"), pl.col(ui.TARGET).mean().alias("taxi_mean"),
        pl.col("px_aobt3").mean().alias("px_mean")).sort("slot")
    dep = b.filter(pl.col("PHASE_mvt") == "DEP")
    arr = b.filter(pl.col("PHASE_mvt") == "ARR")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=[f"{ap} {day}: mean per 15 min of take-off (s)",
                                        "movements per 15 min"])
    if ds == "train":
        fig.add_trace(go.Scatter(x=dep["slot"], y=dep["taxi_mean"], name="taxi-out mean (target)",
                                 line=dict(color=ui.PALETTE[0], width=2), mode="lines+markers"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dep["slot"], y=dep["px_mean"], name="take-off minus AOBT_3, mean",
                             line=dict(color=ui.PALETTE[1], width=2, dash="dot"), mode="lines+markers"), row=1, col=1)
    fig.add_trace(go.Bar(x=dep["slot"], y=dep["n"], name="departures", marker_color=ui.PALETTE[2]), row=2, col=1)
    fig.add_trace(go.Bar(x=arr["slot"], y=arr["n"], name="arrivals", marker_color=ui.PALETTE[6]), row=2, col=1)
    fig.update_yaxes(title_text="seconds", row=1, col=1)
    fig.update_yaxes(title_text="movements", row=2, col=1)
    fig.update_xaxes(title_text="local time", row=2, col=1)
    fig.update_layout(height=560, barmode="group")
    ui.show(fig)
    st.caption(f"{ds}: {int(dep['n'].sum())} departures, {int(arr['n'].sum())} arrivals that local day.")
    st.button("Open this airport-day in the replay view", on_click=ui.jump_to_replay, args=(ap, day), key="tp_jump")
