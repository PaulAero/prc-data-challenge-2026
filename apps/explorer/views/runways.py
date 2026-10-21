"""View 8: runway configuration inferred per local hour, and taxi-out by configuration."""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
import streamlit as st

import ui
from prc.explore import analysis

BY = ("dataset", "local_date", "hour_local")


def render(f: ui.Filters) -> None:
    st.header("8. Runway configuration inference")
    st.markdown(
        "Configuration of an hour = set of runways with at least N departures, and set with at least N "
        "arrivals, in that local hour. Inferred from movements, so it is available in the ranking file too. "
        "Sidebar airports and datasets apply."
    )
    c1, c2 = st.columns(2)
    ap = c1.selectbox("Airport", list(f.airports), key="rc_ap")
    min_n = c2.slider("Min movements per runway and hour", 1, 10, 2, key="rc_min")
    rw = ui.table("runway_hourly").filter((pl.col("airport") == ap) & pl.col("dataset").is_in(f.datasets))
    cfg = analysis.runway_configs(rw, min_n, by=BY)
    if cfg.height == 0:
        st.info("No data.")
        return
    freq = cfg.group_by("dataset", "config").len().with_columns(
        (pl.col("len") / pl.col("len").sum().over("dataset")).alias("share_of_hours")).sort("len", descending=True)
    st.dataframe(freq.pivot(on="dataset", index="config", values="share_of_hours").head(25),
                 hide_index=True, width="stretch")
    st.caption(f"{ap}: share of observed hours in each configuration (top 25; {cfg.height} airport-hours).")

    top = freq.group_by("config").agg(pl.col("len").sum()).sort("len", descending=True)["config"].head(8).to_list()
    cmap = {c: ui.PALETTE[i] for i, c in enumerate(top)}
    days = cfg.with_columns(pl.when(pl.col("config").is_in(top)).then(pl.col("config")).otherwise(pl.lit("other"))
                            .alias("cfg8"))
    fig = go.Figure()
    for c in [*top, "other"]:
        s = days.filter(pl.col("cfg8") == c)
        fig.add_trace(go.Scatter(x=s["local_date"], y=s["hour_local"], mode="markers", name=c,
                                 marker=dict(symbol="square", size=5, color=cmap.get(c, ui.OTHER_COLOR)),
                                 hovertemplate="%{x} %{y}h<br>" + c + "<extra></extra>"))
    fig.update_layout(title=f"{ap}: configuration per local date and hour (top 8 configurations, rest grey)",
                      xaxis_title="local date", yaxis=dict(title="local hour", range=[-0.5, 23.5]), height=460,
                      legend=dict(orientation="h", y=-0.25))
    ui.show(fig)

    ch = analysis.config_changes(cfg, by=("dataset", "local_date"))
    fig = go.Figure()
    for i, ds in enumerate(sorted(ch["dataset"].unique().to_list(), reverse=True)):
        s = ch.filter(pl.col("dataset") == ds)
        fig.add_trace(go.Scatter(x=s["local_date"], y=s["n_changes"], mode="markers", name=ds,
                                 marker=dict(color=ui.DATASET_COLOR.get(ds, ui.PALETTE[i]), size=6)))
    fig.update_layout(title=f"{ap}: hour-to-hour configuration changes per local day (mean "
                            f"{ch['n_changes'].mean():.1f})", xaxis_title="local date", yaxis_title="changes per day",
                      height=320)
    ui.show(fig)

    st.subheader("Taxi-out by configuration (train)")
    dep = ui.movements(("train",), (ap,), ("DEP",), f.months, f.date_range, ("dataset", "local_date", "hour_local",
                                                                             ui.TARGET))
    j = dep.join(cfg, on=list(BY), how="left")
    cs = analysis.category_summary(j, "config", ui.TARGET, 200).sort("n", descending=True).head(20).sort("p50")
    fig = go.Figure(go.Box(x=cs["config"], q1=cs["p25"], median=cs["p50"], q3=cs["p75"], lowerfence=cs["p5"],
                           upperfence=cs["p95"], mean=cs["mean"], boxmean=True, marker_color=ui.PALETTE[0]))
    fig.update_layout(title=f"{ap}: taxi-out (s) by hourly configuration, top 20 by count with n>=200, sorted by "
                            f"median; box p25/p50/p75, whiskers p5/p95 (n={ui.fmt_n(int(cs['n'].sum()))})",
                      xaxis=dict(type="category"), yaxis_title="taxi-out (s)", height=520)
    ui.show(fig)
    st.dataframe(cs, hide_index=True, width="stretch")
    ui.caption_filters(f, ("months", "dates"), f"airport={ap}, train DEP")
