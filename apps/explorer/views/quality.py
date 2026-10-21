"""View 1: inventory, missingness, suspicious records, time granularity."""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
import streamlit as st

import ui
from prc.explore.common import QUALITY_FLAGS

DRILL_COLUMNS = [
    "dataset", "airport", "PHASE_mvt", "MVT_ID_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "local_date",
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt", "blk_minus_aobt3",
    "EOBT_1_flt", "SCHED_TIME_UTC_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt",
    "AIRCRAFT_TYPE_flt", "ADEP_mvt", "ADES_mvt", "ADEP_flt", "ADES_flt", "ADES_FILED_flt", "FLIGHT_ID_mvt",
]


def _heatmap(df: pl.DataFrame, row: str, col: str, value: str, title: str, fmt: str = ".1%") -> go.Figure:
    wide = df.pivot(on=col, index=row, values=value).sort(row)
    cols = [c for c in wide.columns if c != row]
    z = wide.select(cols).to_numpy()
    fig = go.Figure(
        go.Heatmap(
            z=z, x=cols, y=wide[row].to_list(), colorscale=ui.SEQ_SCALE, zmin=0,
            text=[[("" if v is None or v != v else format(v, fmt)) for v in r] for r in z],
            texttemplate="%{text}", hovertemplate=f"{row}=%{{y}}<br>{col}=%{{x}}<br>{value}=%{{z:{fmt}}}<extra></extra>",
            colorbar=dict(tickformat=fmt),
        )
    )
    fig.update_layout(title=title, yaxis=dict(autorange="reversed"))
    return fig


def render(f: ui.Filters) -> None:
    st.header("1. Inventory and data quality")
    st.markdown(
        "What is in each column, where it is missing, which records are suspicious, and which "
        "timestamps are minute-rounded. Counts come from the precomputed cache (all rows)."
    )
    nulls = ui.table("null_rates").filter(pl.col("airport").is_in(f.airports))

    st.subheader("Columns")
    inv = ui.table("inventory")
    overall = (
        nulls.group_by("dataset", "PHASE_mvt", "column")
        .agg(((pl.col("null_rate") * pl.col("n")).sum() / pl.col("n").sum()).alias("null_rate"))
        .with_columns(pl.format("null {} {}", pl.col("dataset"), pl.col("PHASE_mvt")).alias("k"))
        .pivot(on="k", index="column", values="null_rate")
    )
    st.dataframe(inv.join(overall, on="column", how="left"), hide_index=True, width="stretch")
    ui.caption_filters(f, ("airports",), "null rates are row-weighted over the selected airports")

    c1, c2 = st.columns(2)
    ds = c1.selectbox("Dataset for heatmaps", ["train", "ranking"], key="q_ds")
    ph = c2.selectbox("Phase for heatmaps", ["DEP", "ARR"], key="q_ph")
    sub = nulls.filter((pl.col("dataset") == ds) & (pl.col("PHASE_mvt") == ph))
    n_rows = int(sub.group_by("airport").agg(pl.col("n").first())["n"].sum())
    ui.show(
        _heatmap(sub, "column", "airport", "null_rate",
                 f"Null rate by column and airport ({ds}, {ph}, n={ui.fmt_n(n_rows)} rows)"),
        height=820,
    )
    na = sub.filter(pl.col("na_string_rate") > 0)
    if na.height:
        st.caption("Literal string 'NA' (not null) also appears:")
        st.dataframe(
            na.pivot(on="airport", index="column", values="na_string_rate"), hide_index=True, width="stretch"
        )

    st.subheader("NM join rate")
    join = nulls.filter(pl.col("column").is_in(["FLIGHT_ID_mvt", "ADEP_flt", "AOBT_3_flt"])).with_columns(
        (1 - pl.col("null_rate")).alias("present")
    )
    st.dataframe(
        join.with_columns(pl.format("{} {} {}", "dataset", "PHASE_mvt", "column").alias("k"))
        .pivot(on="airport", index="k", values="present")
        .sort("k"),
        hide_index=True, width="stretch",
    )
    st.caption(
        "Share of rows with the NM identifier (FLIGHT_ID_mvt), with joined NM columns (ADEP_flt), "
        "and with the NM actual off-block (AOBT_3_flt). Rows: dataset, phase, column; columns: airport."
    )

    st.subheader("Suspicious records")
    q = ui.table("quality").filter(
        pl.col("airport").is_in(f.airports) & pl.col("dataset").is_in(f.datasets) & pl.col("PHASE_mvt").is_in(f.phases)
    )
    summary = (
        q.group_by("flag")
        .agg(pl.col("n_flagged").sum(), pl.col("n_rows").sum())
        .with_columns((pl.col("n_flagged") / pl.col("n_rows")).alias("rate"))
        .with_columns(pl.col("flag").replace_strict(QUALITY_FLAGS, default="").alias("definition"))
    )
    by_ap = q.pivot(on="airport", index="flag", values="n_flagged", aggregate_function="sum")
    st.dataframe(
        summary.join(by_ap, on="flag").sort("n_flagged", descending=True), hide_index=True, width="stretch"
    )
    ui.caption_filters(f, ("datasets", "airports", "phases"))
    flag = st.selectbox("Drill down into flag", list(QUALITY_FLAGS), key="q_flag",
                        format_func=lambda k: f"{k}: {QUALITY_FLAGS[k]}")
    limit = st.slider("Max rows", 100, 5000, 500, 100, key="q_limit")
    rows = (
        ui.scan()
        .filter(
            pl.col(flag)
            & pl.col("airport").is_in(list(f.airports))
            & pl.col("dataset").is_in(list(f.datasets))
            & pl.col("PHASE_mvt").is_in(list(f.phases))
        )
        .select(DRILL_COLUMNS)
        .head(limit)
        .collect()
    )
    st.dataframe(rows, hide_index=True, width="stretch", height=350)
    st.caption(f"{rows.height} rows shown (first {limit} in airport/time order).")

    st.subheader("Time granularity")
    st.markdown(
        "Two diagnostics of the seconds field of each timestamp column. **sec_mid** = share of seconds in "
        "7..53: about **0.78** for genuine second resolution, **0** for a column that is minute-resolution "
        "with a small added jitter (seconds only in 54..59 and 0..6). **sec0** = share with seconds == 0: "
        "**0.017** for uniform seconds, **1.0** for plain minute rounding, about **0.083** for minute plus "
        "jitter."
    )
    rnd = ui.table("rounding").filter(
        (pl.col("dataset") == ds) & (pl.col("PHASE_mvt") == ph) & pl.col("airport").is_in(f.airports)
    )
    c1, c2 = st.columns(2)
    with c1:
        ui.show(_heatmap(rnd, "column", "airport", "sec_mid",
                         f"Share of timestamps with seconds in 7..53 ({ds}, {ph})", ".2f"), height=480)
    with c2:
        ui.show(_heatmap(rnd, "column", "airport", "sec0",
                         f"Share of timestamps with seconds == 0 ({ds}, {ph})", ".3f"), height=480)
    st.caption("Change the dataset / phase selectors above the null-rate heatmap to compare.")
