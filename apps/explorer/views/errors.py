"""View 7: where the models are wrong (Q4 2025 holdout predictions)."""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
import streamlit as st

import ui
from prc.explore import analysis

JOIN_COLS = ("MVT_ID_mvt", "local_date", "hour_local", "weekday", "RUNWAY_mvt", "STAND_mvt",
             "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt", "FLIGHT_mvt", "px_aobt3")
DIMENSIONS = ["airport", "hour_local", "weekday", "local_date", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt",
              "AIRCRAFT_OPERATOR_flt", "target_bucket"]
BUCKETS = [0, 300, 600, 900, 1200, 1500, 1800, 2400, 3600, 7200]


@st.cache_resource(show_spinner="Joining predictions with movements...", max_entries=4)
def load_predictions(path: str) -> pl.DataFrame:
    p = pl.read_parquet(path)
    mv = ui.scan().filter((pl.col("dataset") == "train") & (pl.col("PHASE_mvt") == "DEP")).select(JOIN_COLS)
    out = p.lazy().join(mv, on="MVT_ID_mvt", how="left").collect()
    labels = [f"<{BUCKETS[1]}"] + [f"{a}-{b}" for a, b in zip(BUCKETS[1:-1], BUCKETS[2:])] + [f">={BUCKETS[-1]}"]
    return out.with_columns(
        pl.col("target").cut(BUCKETS[1:], labels=labels, left_closed=True).cast(pl.String).alias("target_bucket")
    )


def render(f: ui.Filters) -> None:
    st.header("7. Model error explorer")
    exps = ui.experiments()
    if not exps:
        st.info("No experiments/*/predictions.parquet with per-flight predictions found.")
        return
    c1, c2 = st.columns(2)
    exp = c1.selectbox("Experiment", list(exps), index=len(exps) - 1, key="e_exp")
    model = c2.radio("Model", ["latent_prediction", "baseline_prediction"], horizontal=True, key="e_model")
    d = load_predictions(exps[exp]).filter(pl.col("airport").is_in(f.airports))
    missing = d["local_date"].null_count()
    st.caption(f"{exp}: {ui.fmt_n(d.height)} holdout departures (sidebar airports apply). Residual = target - "
               f"prediction, in seconds. {missing} predictions did not join to the movements cache.")

    summ = analysis.residual_summary(d, "target", "baseline_prediction", ["airport"]).rename(
        {"rmse": "rmse_baseline", "bias": "bias_baseline"}).select("airport", "n", "rmse_baseline", "bias_baseline")
    summ = summ.join(analysis.residual_summary(d, "target", "latent_prediction", ["airport"]).select(
        "airport", pl.col("rmse").alias("rmse_latent"), pl.col("bias").alias("bias_latent")), on="airport")
    overall = pl.DataFrame({
        "airport": ["ALL"], "n": [d.height],
        "rmse_baseline": [float(((d["target"] - d["baseline_prediction"]) ** 2).mean() ** 0.5)],
        "bias_baseline": [float((d["target"] - d["baseline_prediction"]).mean())],
        "rmse_latent": [float(((d["target"] - d["latent_prediction"]) ** 2).mean() ** 0.5)],
        "bias_latent": [float((d["target"] - d["latent_prediction"]).mean())],
    })
    st.dataframe(pl.concat([summ, overall], how="vertical_relaxed").with_columns(
        (pl.col("rmse_baseline") - pl.col("rmse_latent")).alias("improvement")), hide_index=True, width="stretch")

    tail = d.with_columns(((pl.col("target") - pl.col(model)) ** 2).alias("se")).sort("se", descending=True)
    tot = tail["se"].sum()
    k1, k5 = int(0.01 * tail.height), int(0.05 * tail.height)
    st.markdown(
        f"**Tail share of squared error ({model})**: top 1% of flights = "
        f"{tail['se'].head(k1).sum() / tot:.1%}, top 5% = {tail['se'].head(k5).sum() / tot:.1%} of the total. "
        f"RMSE without the top 1%: {(tail['se'].tail(tail.height - k1).mean()) ** 0.5:.1f} s "
        f"(all: {(tot / tail.height) ** 0.5:.1f} s)."
    )

    st.subheader("Residual by dimension")
    c1, c2, c3 = st.columns(3)
    dim = c1.selectbox("Dimension", DIMENSIONS, index=1, key="e_dim")
    top_k = c2.slider("Max categories (by count)", 5, 100, 40, 5, key="e_topk")
    min_n = c3.slider("Min flights", 1, 1000, 30, 1, key="e_minn")
    rs = analysis.residual_summary(d.with_columns(pl.col(dim).cast(pl.String)), "target", model, [dim]).filter(
        pl.col("n") >= min_n)
    if dim in ("hour_local", "weekday", "local_date", "target_bucket"):
        rs = rs.sort("n", descending=True).head(top_k)
        if dim == "target_bucket":
            order = {lab: i for i, lab in enumerate(
                [f"<{BUCKETS[1]}"] + [f"{a}-{b}" for a, b in zip(BUCKETS[1:-1], BUCKETS[2:])] + [f">={BUCKETS[-1]}"])}
            rs = rs.with_columns(pl.col(dim).replace_strict(order, default=99).alias("_o")).sort("_o").drop("_o")
        elif dim in ("hour_local", "weekday"):
            rs = rs.with_columns(pl.col(dim).cast(pl.Int32).alias("_o")).sort("_o").drop("_o")
        else:
            rs = rs.sort(dim)
    else:
        rs = rs.sort("n", descending=True).head(top_k).sort("rmse", descending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=rs[dim], y=rs["rmse"], name="RMSE", marker_color=ui.PALETTE[0], customdata=rs["n"],
                         hovertemplate="%{x}<br>RMSE %{y:.0f} s<br>n=%{customdata}<extra></extra>"))
    fig.add_trace(go.Bar(x=rs[dim], y=rs["bias"], name="bias (mean target - prediction)", marker_color=ui.PALETTE[1],
                         hovertemplate="%{x}<br>bias %{y:.0f} s<extra></extra>"))
    fig.update_layout(title=f"{exp} {model}: RMSE and bias (s) by {dim}, n>={min_n}, "
                            f"{rs.height} categories, {ui.fmt_n(int(rs['n'].sum()))} flights",
                      barmode="group", xaxis=dict(type="category", title=dim), yaxis_title="seconds", height=440)
    ui.show(fig)
    fig = go.Figure(go.Bar(x=rs[dim], y=rs["sse"] / tot, marker_color=ui.PALETTE[2]))
    fig.update_layout(title=f"Share of total squared error by {dim}", xaxis=dict(type="category"),
                      yaxis=dict(tickformat=".1%", title="share of SSE"), height=260)
    ui.show(fig)

    st.subheader("Worst 200 flights")
    worst = tail.head(200).select(
        "airport", "local_date", "event_time", "FLIGHT_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt",
        "target", "baseline_prediction", "latent_prediction", "px_aobt3", "se", "MVT_ID_mvt")
    st.dataframe(worst, hide_index=True, width="stretch", height=360)
    st.caption("px_aobt3 = take-off minus NM AOBT_3 (s): when it is close to the target the error was predictable "
               "from data available at ranking time.")

    st.subheader("Worst airport-days (sum of squared error)")
    days = analysis.residual_summary(d, "target", model, ["airport", "local_date"]).sort("sse", descending=True)
    days = days.with_columns((pl.col("sse") / tot).alias("share_of_total_sse"))
    st.dataframe(days.head(50), hide_index=True, width="stretch", height=300)
    opts = [f"{a} {dd}" for a, dd in zip(days["airport"].head(50), days["local_date"].head(50))]
    if opts:
        pick = st.selectbox("Airport-day", opts, key="e_day")
        a, dd = pick.split(" ")
        row = days.filter((pl.col("airport") == a) & (pl.col("local_date").cast(pl.String) == dd)).row(0, named=True)
        st.button(f"Open {pick} in the replay view", on_click=ui.jump_to_replay, args=(a, row["local_date"]),
                  key="e_jump")

    st.subheader("Baseline vs latent per airport-day")
    b = analysis.residual_summary(d, "target", "baseline_prediction", ["airport", "local_date"]).select(
        "airport", "local_date", "n", pl.col("rmse").alias("rmse_baseline"))
    l_ = analysis.residual_summary(d, "target", "latent_prediction", ["airport", "local_date"]).select(
        "airport", "local_date", pl.col("rmse").alias("rmse_latent"))
    bl = b.join(l_, on=["airport", "local_date"])
    mx = float(max(bl["rmse_baseline"].quantile(0.99), bl["rmse_latent"].quantile(0.99)))
    fig = go.Figure(go.Scatter(
        x=bl["rmse_baseline"], y=bl["rmse_latent"], mode="markers",
        marker=dict(color=ui.PALETTE[0], size=8, opacity=0.6, line=dict(color="white", width=1)),
        text=[f"{a} {dd} n={n}" for a, dd, n in zip(bl["airport"], bl["local_date"], bl["n"])],
        hovertemplate="%{text}<br>baseline %{x:.0f} s<br>latent %{y:.0f} s<extra></extra>", name="airport-day"))
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color="#52514e", dash="dot", width=1),
                             name="y = x", hoverinfo="skip"))
    better = int((bl["rmse_latent"] < bl["rmse_baseline"]).sum())
    fig.update_layout(title=f"Per airport-day RMSE (s): latent below the diagonal is better "
                            f"({better} of {bl.height} airport-days)", xaxis_title="baseline RMSE (s)",
                      yaxis_title="latent RMSE (s)", height=520, xaxis=dict(range=[0, mx]), yaxis=dict(range=[0, mx]))
    ui.show(fig)
    st.caption("Axes clipped at the 99th percentile; hover for the airport-day.")
