"""View 4: target against stand, runway, aircraft, operator, time of day."""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
import streamlit as st

import ui
from prc.explore import analysis

CATEGORICALS = [
    "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt",
    "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt", "ADES_mvt", "hour_local", "weekday", "month", "airport",
]
COLS = tuple(dict.fromkeys([*CATEGORICALS, ui.TARGET]))
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def render(f: ui.Filters) -> None:
    st.header("4. Relationships")
    st.markdown("How the taxi-out target (s) moves with stand, runway, aircraft, operator and time. "
                "Train DEP only; sidebar airports, months and dates apply.")
    d = ui.movements(("train",), f.airports, ("DEP",), f.months, f.date_range, COLS)
    if d.height == 0:
        st.warning("No rows for these filters.")
        return
    stat_name = st.radio("Statistic for heatmaps", ["p50", "p90", "mean"], horizontal=True, key="r_stat")

    st.subheader("Stand x runway")
    c1, c2, c3 = st.columns(3)
    ap = c1.selectbox("Airport", sorted(d["airport"].unique().to_list()), key="r_ap")
    top_n = c2.slider("Top N stands by departures", 10, 120, 40, 5, key="r_topn")
    min_n = c3.slider("Min flights per cell", 1, 200, 20, 1, key="r_minn")
    a = d.filter(pl.col("airport") == ap)
    top = a.group_by("STAND_mvt").len().sort("len", descending=True).head(top_n)["STAND_mvt"]
    cell = analysis.category_summary(a.filter(pl.col("STAND_mvt").is_in(top.implode())), "RUNWAY_mvt",
                                     ui.TARGET, min_n, by=["STAND_mvt"])
    stat_col = {"p50": "p50", "p90": "p90", "mean": "mean"}[stat_name]
    wide = cell.pivot(on="RUNWAY_mvt", index="STAND_mvt", values=stat_col)
    nwide = cell.pivot(on="RUNWAY_mvt", index="STAND_mvt", values="n")
    order = pl.DataFrame({"STAND_mvt": top})
    wide = order.join(wide, on="STAND_mvt", how="left")
    nwide = order.join(nwide, on="STAND_mvt", how="left")
    rws = sorted(c for c in wide.columns if c != "STAND_mvt")
    fig = go.Figure(go.Heatmap(
        z=wide.select(rws).to_numpy(), x=rws, y=wide["STAND_mvt"].to_list(), colorscale=ui.SEQ_SCALE,
        customdata=nwide.select(rws).to_numpy(), colorbar=dict(title="s"),
        hovertemplate="stand %{y}<br>runway %{x}<br>" + stat_name + " %{z:.0f} s<br>n=%{customdata}<extra></extra>",
    ))
    fig.update_layout(title=f"{ap}: {stat_name} taxi-out (s) by stand x runway, top {top_n} stands, "
                            f"cells with n>={min_n}, n={ui.fmt_n(a.height)} flights",
                      yaxis=dict(autorange="reversed", type="category"), xaxis_title="runway", yaxis_title="stand",
                      height=max(400, 16 * top_n))
    ui.show(fig)

    st.subheader("Hour of day x weekday (local time)")
    hw = d.filter(pl.col("airport") == ap).group_by("weekday", "hour_local").agg(
        pl.col(ui.TARGET).median().alias("p50"), pl.col(ui.TARGET).quantile(0.9).alias("p90"),
        pl.col(ui.TARGET).mean().alias("mean"), pl.len().alias("n"))
    wide = hw.pivot(on="hour_local", index="weekday", values=stat_col).sort("weekday")
    nw = hw.pivot(on="hour_local", index="weekday", values="n").sort("weekday")
    hours = sorted(int(c) for c in wide.columns if c != "weekday")
    fig = go.Figure(go.Heatmap(
        z=wide.select([str(h) for h in hours]).to_numpy(), x=hours,
        y=[WEEKDAYS[i - 1] for i in wide["weekday"]], colorscale=ui.SEQ_SCALE,
        customdata=nw.select([str(h) for h in hours]).to_numpy(), colorbar=dict(title="s"),
        hovertemplate="%{y} %{x}h<br>" + stat_name + " %{z:.0f} s<br>n=%{customdata}<extra></extra>",
    ))
    fig.update_layout(title=f"{ap}: {stat_name} taxi-out (s) by local hour and weekday (hour of take-off)",
                      xaxis_title="local hour of take-off", yaxis=dict(autorange="reversed"), height=360)
    ui.show(fig)

    st.subheader("One categorical: counts and distribution")
    c1, c2, c3, c4 = st.columns(4)
    cat = c1.selectbox("Categorical", CATEGORICALS, index=2, key="r_cat")
    scope = c2.selectbox("Airports", ["selected airport only", "all selected airports"], key="r_scope")
    k = c3.slider("Show top K by count", 5, 80, 30, 5, key="r_k")
    mn = c4.slider("Min flights", 1, 1000, 50, 10, key="r_mn")
    src = a if scope.startswith("selected airport") else d
    cs = analysis.category_summary(src.with_columns(pl.col(cat).cast(pl.String)), cat, ui.TARGET, mn)
    cs = cs.sort("n", descending=True).head(k).sort("p50")
    labels = [str(v) for v in cs[cat].to_list()]
    fig = go.Figure(go.Box(
        x=labels, q1=cs["p25"], median=cs["p50"], q3=cs["p75"], lowerfence=cs["p5"], upperfence=cs["p95"],
        mean=cs["mean"], boxmean=True, marker_color=ui.PALETTE[0], name=cat,
    ))
    fig.update_layout(title=f"Taxi-out (s) by {cat}: box = p25/p50/p75, whiskers = p5/p95, dashed = mean; "
                            f"top {k} by count with n>={mn}, sorted by median ({scope}, n={ui.fmt_n(int(cs['n'].sum()))})",
                      xaxis=dict(type="category", title=cat), yaxis_title="taxi-out (s)", height=450)
    ui.show(fig)
    fig = go.Figure(go.Bar(x=labels, y=cs["n"], marker_color=ui.PALETTE[0]))
    fig.update_layout(title=f"Departures per {cat} (same order)", xaxis=dict(type="category"),
                      yaxis_title="departures", height=280)
    ui.show(fig)
    st.dataframe(cs, hide_index=True, width="stretch")
