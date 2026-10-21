import polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(30); pl.Config.set_fmt_str_lengths(40)
df = pl.read_parquet("experiments/lab-ltfm/ltfm.parquet")
d = df.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("src") == "train"))
s = lambda a, b: (pl.col(a) - pl.col(b)).dt.total_seconds()
d = d.with_columns(
    m=pl.col("MVT_TIME_UTC_mvt").dt.month(),
    blk_sec=pl.col("BLOCK_TIME_UTC_mvt").dt.second(), to_sec=pl.col("MVT_TIME_UTC_mvt").dt.second(),
    d_sched=s("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"), d_lobt=s("BLOCK_TIME_UTC_mvt", "LOBT_flt"),
    d_eobt=s("BLOCK_TIME_UTC_mvt", "EOBT_1_flt"), d_iobt=s("BLOCK_TIME_UTC_mvt", "IOBT_flt"),
    d_aobt=s("BLOCK_TIME_UTC_mvt", "AOBT_3_flt"), y=pl.col("TAXITIME_SEC_mvt"),
    px=s("MVT_TIME_UTC_mvt", "AOBT_3_flt"), nm=pl.col("LOBT_flt").is_not_null(),
)
print("n", d.height, "nm share", d["nm"].mean())
print("block seconds hist (top)"); print(d["blk_sec"].value_counts().sort("count", descending=True).head(15))
print("takeoff seconds hist (top)"); print(d["to_sec"].value_counts().sort("count", descending=True).head(10))
# seconds jitter: block_sec in [54..59] U [0..6]
d = d.with_columns(blk_minjit=((pl.col("blk_sec") >= 54) | (pl.col("blk_sec") <= 6)))
for c in ["d_sched", "d_lobt", "d_eobt", "d_iobt", "d_aobt"]:
    print(c, "|x|<=10:", d.select((pl.col(c).abs() <= 10).mean()).item(), " |x|<=60:", d.select((pl.col(c).abs() <= 60).mean()).item())
print("y quantiles", d["y"].quantile(0.01), d["y"].quantile(0.5), d["y"].quantile(0.99), d["y"].max())
# y mod 60 structure
print("y value counts top"); print(d["y"].value_counts().sort("count", descending=True).head(20))
by = d.group_by("m").agg(
    n=pl.len(), nm=pl.col("nm").mean(), minjit=pl.col("blk_minjit").mean(),
    fill_sched=(pl.col("d_sched").abs() <= 10).mean(), eq_lobt=(pl.col("d_lobt").abs() <= 10).mean(),
    eq_eobt=(pl.col("d_eobt").abs() <= 10).mean(), eq_aobt=(pl.col("d_aobt").abs() <= 10).mean(),
    eq_aobt60=(pl.col("d_aobt").abs() <= 60).mean(),
    ymed=pl.col("y").median(), pxmed=pl.col("px").median(),
).sort("m")
print(by)
# combined jitter: block minute jitter around block value? distribution of block - aobt in seconds
print("d_aobt hist (10 s bins, |x|<300)")
print(d.filter(pl.col("d_aobt").abs() < 300).with_columns(b=(pl.col("d_aobt") / 30).floor() * 30)["b"].value_counts().sort("b"))
