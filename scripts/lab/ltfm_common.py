import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(30); pl.Config.set_fmt_str_lengths(40); pl.Config.set_tbl_width_chars(250)
N_BOARD = 344_841
s = lambda a, b: (pl.col(a) - pl.col(b)).dt.total_seconds()

def load():
    df = pl.read_parquet("experiments/lab-ltfm/ltfm.parquet")
    df = df.with_columns(
        m=pl.col("MVT_TIME_UTC_mvt").dt.month(), yr=pl.col("MVT_TIME_UTC_mvt").dt.year(),
        loc=pl.col("MVT_TIME_UTC_mvt").dt.convert_time_zone("Europe/Istanbul"),
        al=pl.col("FLIGHT_mvt").str.extract(r"^([A-Z0-9]{2,3}?)\d", 1),
        cs3=pl.col("CALLSIGN_flt").str.slice(0, 3),
        nm=pl.col("LOBT_flt").is_not_null(),
        d_sched=s("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"),
        to_sched=s("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"),
        to_lobt=s("MVT_TIME_UTC_mvt", "LOBT_flt"),
        px=s("MVT_TIME_UTC_mvt", "AOBT_3_flt"),
        sched_lobt=s("SCHED_TIME_UTC_mvt", "LOBT_flt"),
        y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
        stand_area=pl.col("STAND_mvt").str.extract(r"^([A-Z]+)", 1),
    ).with_columns(
        hr=pl.col("loc").dt.hour(), wd=pl.col("loc").dt.weekday(), day=pl.col("loc").dt.date(),
        res=pl.col("target") - pl.col("pred"),
        fill=(pl.col("d_sched").abs() <= 10),
    )
    return df

def oof_group_gain(d, keys, shrink=50.0, min_n=1, demean=True):
    """Fit mean residual per group on one fold, apply to the other; return SSE change summed over both."""
    out = {}
    tot_before = 0.0; tot_after = 0.0
    if demean:
        d = d.with_columns(res=pl.col("res") - pl.col("res").mean().over("fold"))
    for tr, te in [("jan", "jul"), ("jul", "jan")]:
        g = d.filter(pl.col("fold") == tr).group_by(keys).agg(sr=pl.col("res").sum(), n=pl.len())
        g = g.with_columns(corr=pl.col("sr") / (pl.col("n") + shrink))
        t = d.filter(pl.col("fold") == te).join(g.select(*keys, "corr"), on=keys, how="left").with_columns(pl.col("corr").fill_null(0.0))
        b = (t["res"] ** 2).sum(); a = ((t["res"] - t["corr"]) ** 2).sum()
        tot_before += b; tot_after += a
    return tot_before, tot_after
