"""View 2: the target distribution and the near-target proxies in the ranking file."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from plotly.subplots import make_subplots

import ui
from prc.explore import analysis
from prc.explore.common import PROXIES, PROXY_LABELS

COLS = ("airport", "month", "local_date", ui.TARGET, *PROXIES, "blk_minus_aobt3", "blk_minus_sched", "q_no_nm_match")


def _load(f: ui.Filters) -> pl.DataFrame:
    return ui.movements(("train",), f.airports, ("DEP",), f.months, f.date_range, COLS)


def _small_multiples(hist: pl.DataFrame, airports: list[str], title: str, xlab: str, log_y: bool,
                     lo: float, hi: float, width: float) -> go.Figure:
    ncols = 1 if len(airports) == 1 else 2
    nrows = int(np.ceil(len(airports) / ncols))
    titles = []
    for ap in airports:
        h = hist.filter(pl.col("airport") == ap)
        n = int(h["count"].sum())
        below = int(h.filter(pl.col("zone") == "below")["count"].sum())
        above = int(h.filter(pl.col("zone") == "above")["count"].sum())
        titles.append(f"{ap} (n={ui.fmt_n(n)}, <{lo:g}: {below}, >={hi:g}: {above})")
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles, shared_xaxes=True,
                        vertical_spacing=0.06 if nrows > 2 else 0.12)
    for i, ap in enumerate(airports):
        h = hist.filter((pl.col("airport") == ap) & (pl.col("zone") == "in_range"))
        fig.add_trace(
            go.Bar(x=(h["bin_left"] + width / 2).to_list(), y=h["count"].to_list(), width=width,
                   marker_color=ui.PALETTE[0], name=ap, showlegend=False,
                   hovertemplate=f"{ap}<br>bin start %{{x}} s<br>count %{{y}}<extra></extra>"),
            row=i // ncols + 1, col=i % ncols + 1,
        )
    fig.update_yaxes(type="log" if log_y else "linear", title_text="flights")
    fig.update_xaxes(title_text=xlab, range=[lo, hi], row=nrows)
    fig.update_layout(title=title, bargap=0.05, height=max(320, 230 * nrows))
    return fig


def render(f: ui.Filters) -> None:
    st.header("2. What the model is asked to predict")
    st.markdown(
        "Target: `TAXITIME_SEC_mvt` of departures = take-off (`MVT_TIME_UTC_mvt`) minus off-block "
        "(`BLOCK_TIME_UTC_mvt`), in seconds. Only the 2025 training file has it. "
        "Sidebar dataset and phase filters are ignored here (train, DEP)."
    )
    d = _load(f)
    airports = sorted(d["airport"].unique().to_list())
    if d.height == 0:
        st.warning("No rows for these filters.")
        return
    ui.caption_filters(f, ("airports", "months", "dates"), f"train DEP, n={ui.fmt_n(d.height)}")

    st.subheader("Distribution")
    c1, c2, c3, c4 = st.columns(4)
    lo = c1.number_input("x min (s)", value=0, step=60, key="t_lo")
    hi = c2.number_input("x max (s)", value=3600, step=300, key="t_hi")
    width = c3.selectbox("bin width (s)", [10, 30, 60, 120], index=1, key="t_w")
    log_y = c4.checkbox("log y", value=True, key="t_log")
    hist = analysis.histogram(d, ui.TARGET, ["airport"], lo, hi, width)
    ui.show(_small_multiples(hist, airports, "Taxi-out time per departure (seconds), by airport",
                             "taxi-out time (s)", log_y, lo, hi, width))

    q = analysis.target_quantiles(d, ui.TARGET, ["airport"])
    q_all = analysis.target_quantiles(d.with_columns(pl.lit("ALL").alias("airport")), ui.TARGET, ["airport"])
    mult60 = d.group_by("airport").agg((pl.col(ui.TARGET) % 60 == 0).mean().alias("share_multiple_of_60s"))
    st.markdown("**Quantiles (seconds)**")
    st.dataframe(pl.concat([q, q_all]).join(mult60, on="airport", how="left"),
                 hide_index=True, width="stretch")
    st.caption("share_multiple_of_60s: 1/60 = 0.017 if the target had true second resolution. About 0.042 "
               "where both endpoints are minute-resolution plus a small jitter (see view 1, time granularity).")

    st.markdown("**Tail dominance of RMSE**")
    tail = pl.concat([
        analysis.tail_dominance(d, ui.TARGET, ["airport"]),
        analysis.tail_dominance(d.with_columns(pl.lit("ALL").alias("airport")), ui.TARGET, ["airport"]),
    ])
    st.dataframe(tail, hide_index=True, width="stretch")
    st.caption(
        "rmse_group_mean: RMSE (s) of predicting the airport mean (ALL: the global mean). "
        "share_sq_top_1pct: share of that squared error carried by the 1% of flights farthest from "
        "the mean. rmse_without_top_1pct: RMSE (s) of the same predictor on the other 99%. "
        "A high share means the metric is decided by a few extreme flights."
    )

    st.subheader("Near-target proxies available in the ranking file")
    st.markdown(
        "Each proxy is take-off minus an off-block-like time, so it is in the target's units. "
        "The ranking file keeps take-off time and these references for departures."
    )
    ps = analysis.proxy_summary(d, ui.TARGET, list(PROXIES), ["airport"])
    sd = q.select("airport", pl.col("sd").alias("target_sd"))
    st.dataframe(
        ps.join(sd, on="airport", how="left")
        .with_columns(pl.col("proxy").replace_strict(PROXY_LABELS).alias("meaning"))
        .select("airport", "proxy", "availability", "n_both", "rmse_raw", "bias", "rmse_debiased",
                "target_sd", "mae_raw", "median_residual", "corr", "within_60s", "meaning"),
        hide_index=True, width="stretch", height=420,
    )
    st.caption(
        "rmse_raw: RMSE (s) of using the proxy as the prediction. bias: mean(target - proxy) (s). "
        "rmse_debiased: RMSE (s) after subtracting the per-airport bias. target_sd: RMSE of the "
        "airport-mean predictor; a proxy is informative only if rmse_debiased is well below it. "
        "within_60s: share of flights where the proxy is within one minute of the target."
    )
    av = ui.table("proxy_availability").filter(pl.col("airport").is_in(f.airports))
    av_long = av.unpivot(index=["dataset", "airport", "month", "n"], on=list(PROXIES),
                         variable_name="proxy", value_name="availability")
    av_ds = (
        av_long.group_by("dataset", "proxy")
        .agg(((pl.col("availability") * pl.col("n")).sum() / pl.col("n").sum()).alias("availability"))
        .pivot(on="dataset", index="proxy", values="availability")
    )
    st.markdown("**Proxy availability, train 2025 vs ranking 2026 (DEP, selected airports)**")
    st.dataframe(av_ds, hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    proxy = c1.selectbox("Proxy", list(PROXIES), format_func=lambda p: f"{p}: {PROXY_LABELS[p]}", key="t_px")
    ap = c2.selectbox("Airport", airports, key="t_px_ap")
    sub = d.filter(pl.col("airport") == ap)
    counts, edges, n_in, n_pairs = analysis.density_2d(sub, proxy, ui.TARGET, 0, 3600, 30)
    centers = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure(go.Heatmap(
        z=np.log10(counts + 1), x=centers, y=centers, colorscale=ui.SEQ_SCALE,
        customdata=counts, colorbar=dict(title="log10(n+1)"),
        hovertemplate="proxy %{x:.0f} s<br>target %{y:.0f} s<br>n=%{customdata:.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(x=[0, 3600], y=[0, 3600], mode="lines", line=dict(color="#52514e", width=1, dash="dot"),
                             name="y = x", hoverinfo="skip"))
    fig.update_layout(
        title=f"{ap}: target vs {proxy} (s), 30 s bins, n={ui.fmt_n(n_in)} of {ui.fmt_n(n_pairs)} pairs in 0-3600 s",
        xaxis_title=f"{proxy} (s)", yaxis_title="taxi-out target (s)", height=560,
        xaxis=dict(range=[0, 3600]), yaxis=dict(range=[0, 3600]), width=None,
    )
    ui.show(fig)

    res = sub.select((pl.col(ui.TARGET) - pl.col(proxy)).alias("residual"), "airport").drop_nulls()
    hres = analysis.histogram(res, "residual", ["airport"], -1800, 1800, 30)
    ui.show(_small_multiples(hres, [ap], f"{ap}: target - {proxy} (s), n={ui.fmt_n(res.height)}",
                             "target - proxy (s)", log_y, -1800, 1800, 30), height=380)

    st.subheader("NM actual off-block (AOBT_3) vs airport off-block (BLOCK_TIME)")
    b = d.select("airport", "blk_minus_aobt3").drop_nulls()
    bsum = b.group_by("airport").agg(
        pl.len().alias("n"),
        pl.col("blk_minus_aobt3").median().alias("median_s"),
        (pl.col("blk_minus_aobt3").abs() <= 60).mean().alias("within_60s"),
        (pl.col("blk_minus_aobt3").abs() <= 300).mean().alias("within_300s"),
        pl.col("blk_minus_aobt3").quantile(0.05).alias("p5_s"),
        pl.col("blk_minus_aobt3").quantile(0.95).alias("p95_s"),
        (pl.col("blk_minus_aobt3") == 0).mean().alias("exactly_equal"),
    ).sort("airport")
    st.dataframe(bsum, hide_index=True, width="stretch")
    hb = analysis.histogram(b, "blk_minus_aobt3", ["airport"], -900, 900, 30)
    ui.show(_small_multiples(hb, airports, "BLOCK_TIME_UTC_mvt - AOBT_3_flt (s), by airport",
                             "airport off-block minus NM AOBT (s)", False, -900, 900, 30))
    st.caption(
        "Positive: the airport off-block is later than the NM actual off-block. AOBT_3 is "
        "minute-rounded almost everywhere (see view 1), so up to +/-60 s of spread is rounding alone."
    )

    st.subheader("Off-block time equal to the schedule (possible imputation)")
    st.markdown(
        "If an airport fills a missing off-block with the scheduled time, BLOCK - SCHED piles up at 0 "
        "(within the +/-6 s jitter). For those flights the target equals take-off minus schedule "
        "(`px_sched`), which is available in the ranking file. Compare the 0 cluster with the +/-1 min clusters."
    )
    bs = d.select("airport", "blk_minus_sched", "q_no_nm_match", ui.TARGET).drop_nulls("blk_minus_sched")
    near = lambda c: (pl.col("blk_minus_sched") - c).abs() <= 6  # noqa: E731
    st.dataframe(
        bs.group_by("airport").agg(
            pl.len().alias("n"),
            near(0).mean().alias("share_within_6s_of_sched"),
            ((near(-60).mean() + near(60).mean()) / 2).alias("share_at_plus_minus_1min"),
            near(0).filter(pl.col("q_no_nm_match")).mean().alias("share_within_6s_no_nm_match"),
            pl.col(ui.TARGET).filter(near(0)).quantile(0.99).alias("target_p99_when_block_eq_sched"),
            pl.col(ui.TARGET).filter(~near(0)).quantile(0.99).alias("target_p99_otherwise"),
        ).sort("airport"),
        hide_index=True, width="stretch",
    )
    ap2 = st.selectbox("Airport for BLOCK - SCHED histogram", airports, key="t_bs_ap")
    hb2 = analysis.histogram(bs.filter(pl.col("airport") == ap2), "blk_minus_sched", ["airport"], -600, 600, 1)
    ui.show(_small_multiples(hb2, [ap2], f"{ap2}: BLOCK_TIME - SCHED_TIME (s), 1 s bins", "off-block minus schedule (s)",
                             False, -600, 600, 1), height=360)
