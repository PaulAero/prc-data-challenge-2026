"""View 3: training 2025 vs ranking 2026 distribution shift, per airport."""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
import streamlit as st

import ui
from prc.explore import analysis

PERIODS = ["2025-01", "2025-07", "2026-01", "2026-07"]
PERIOD_COLOR = dict(zip(PERIODS, [ui.PALETTE[0], ui.PALETTE[2], ui.PALETTE[1], ui.PALETTE[3]]))
NOVELTY_COLUMNS = ["STAND_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt", "ADES_mvt", "FLIGHT_mvt", "RUNWAY_mvt"]


def render(f: ui.Filters) -> None:
    st.header("3. Train vs ranking comparison (distribution shift)")
    st.markdown(
        "Same calendar months side by side: January and July 2025 (train) against January and July "
        "2026 (ranking). Uses the sidebar airport selection for the airport picker; dataset, phase "
        "and month filters are fixed by the comparison."
    )
    ap = st.selectbox("Airport", list(f.airports), key="s_ap")

    # Departures per day
    daily = ui.table("daily").filter(pl.col("airport") == ap).with_columns(
        pl.col("local_date").dt.strftime("%Y-%m").alias("period"), pl.col("local_date").dt.day().alias("dom")
    ).filter(pl.col("period").is_in(PERIODS))
    fig = go.Figure()
    for per in PERIODS:
        s = daily.filter(pl.col("period") == per).sort("dom")
        fig.add_trace(go.Scatter(x=s["dom"], y=s["n_dep"], name=f"{per} (mean {s['n_dep'].mean():.0f}/day)",
                                 mode="lines+markers", line=dict(color=PERIOD_COLOR[per], width=2),
                                 marker=dict(size=6)))
    fig.update_layout(title=f"{ap}: departures per local day", xaxis_title="day of month (local)",
                      yaxis_title="departures per day")
    ui.show(fig, height=380)

    # Hourly profile
    ph = st.radio("Phase for hourly profile and runway shares", ["DEP", "ARR"], horizontal=True, key="s_ph")
    hourly = ui.table("hourly").filter(
        (pl.col("airport") == ap) & (pl.col("PHASE_mvt") == ph) & pl.col("month").is_in(PERIODS)
    )
    fig = go.Figure()
    for per in PERIODS:
        s = hourly.filter(pl.col("month") == per).sort("hour_local")
        tot = s["n"].sum()
        fig.add_trace(go.Scatter(x=s["hour_local"], y=s["n"] / tot if tot else s["n"], mode="lines+markers",
                                 name=f"{per} (n={ui.fmt_n(tot)})", line=dict(color=PERIOD_COLOR[per], width=2)))
    fig.update_layout(title=f"{ap}: share of {ph} movements by local hour", xaxis_title="local hour",
                      yaxis_title="share of month's movements", yaxis_tickformat=".0%")
    ui.show(fig, height=380)

    # Runway usage shares
    rw = ui.table("runway_hourly").filter((pl.col("airport") == ap) & (pl.col("PHASE_mvt") == ph)).with_columns(
        pl.col("local_date").dt.strftime("%Y-%m").alias("period")
    ).filter(pl.col("period").is_in(PERIODS))
    shares = rw.group_by("period", "RUNWAY_mvt").agg(pl.col("n").sum()).with_columns(
        (pl.col("n") / pl.col("n").sum().over("period")).alias("share")
    )
    fig = go.Figure()
    for per in PERIODS:
        s = shares.filter(pl.col("period") == per).sort("RUNWAY_mvt")
        fig.add_trace(go.Bar(x=s["RUNWAY_mvt"], y=s["share"], name=per, marker_color=PERIOD_COLOR[per],
                             customdata=s["n"], hovertemplate="%{x}: %{y:.1%} (n=%{customdata})<extra>" + per + "</extra>"))
    fig.update_layout(title=f"{ap}: {ph} runway usage share per period", barmode="group",
                      yaxis_tickformat=".0%", xaxis_title="runway", yaxis_title="share of movements")
    ui.show(fig, height=380)

    # Novelty
    st.subheader("Categories never seen in 2025 training")
    d = ui.movements(("train", "ranking"), (ap,), ("DEP",), (), None, ("dataset", "month", *NOVELTY_COLUMNS))
    train = d.filter(pl.col("dataset") == "train")
    q13 = train.filter(pl.col("month") <= "2025-09")
    rows = []
    for col in NOVELTY_COLUMNS:
        for label, ref, cand in [
            ("ranking 2026-01 vs train 2025", train, d.filter(pl.col("month") == "2026-01")),
            ("ranking 2026-07 vs train 2025", train, d.filter(pl.col("month") == "2026-07")),
            ("validation analog: 2025 Q4 vs 2025 Q1-Q3", q13, train.filter(pl.col("month") >= "2025-10")),
        ]:
            u = analysis.unseen_share(ref, cand, col)
            rows.append({"column": col, "comparison": label, **u})
    nov = pl.DataFrame(rows)
    st.dataframe(nov.pivot(on="comparison", index="column", values="rows_unseen_share"), hide_index=True, width="stretch")
    st.caption(f"Share of {ap} departures whose value was never seen in the reference period. "
               "If the validation analog row is much lower than the ranking rows, the Q4 holdout "
               "underestimates the novelty the model will meet.")
    with st.expander("Distinct-value counts"):
        st.dataframe(nov, hide_index=True, width="stretch")

    # Proxy availability and proxy distribution per month
    st.subheader("Proxies per month")
    av = ui.table("proxy_availability").filter(pl.col("airport") == ap)
    st.dataframe(av, hide_index=True, width="stretch")
    st.caption("Availability share of each proxy among departures, per UTC month (train and ranking).")
    px = ui.movements(("train", "ranking"), (ap,), ("DEP",), (), None, ("month", "px_aobt3", ui.TARGET))
    qm = px.group_by("month").agg(
        pl.col("px_aobt3").median().alias("px_aobt3_p50"), pl.col("px_aobt3").quantile(0.9).alias("px_aobt3_p90"),
        pl.col(ui.TARGET).median().alias("target_p50"), pl.col(ui.TARGET).quantile(0.9).alias("target_p90"),
        pl.len().alias("n"),
    ).sort("month")
    fig = go.Figure()
    for col, color, dash in [("target_p50", ui.PALETTE[0], "solid"), ("target_p90", ui.PALETTE[0], "dot"),
                             ("px_aobt3_p50", ui.PALETTE[1], "solid"), ("px_aobt3_p90", ui.PALETTE[1], "dot")]:
        fig.add_trace(go.Scatter(x=qm["month"], y=qm[col], name=col, mode="lines+markers",
                                 line=dict(color=color, dash=dash, width=2), marker=dict(size=8)))
    fig.update_layout(title=f"{ap}: monthly median and p90 of target and of take-off minus AOBT_3 (s)",
                      xaxis_title="month (UTC)", yaxis_title="seconds", xaxis_type="category")
    ui.show(fig, height=400)
    st.caption("The target is blank in 2026; take-off minus AOBT_3 is the closest observable stand-in "
               "for how taxi-out levels moved between years.")
