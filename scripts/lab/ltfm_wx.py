import polars as pl
pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250)
w = pl.read_parquet("data/external/weather/openmeteo_LTFM.parquet")
print(w.columns); print(w.head(2))
tc = [c for c in w.columns if "time" in c.lower()][0]
w = w.with_columns(pl.col(tc).cast(pl.Datetime("us", "UTC")) if w[tc].dtype != pl.Datetime else pl.col(tc))
w = w.with_columns(per=pl.col(tc).dt.year().cast(pl.Utf8) + "-" + pl.col(tc).dt.month().cast(pl.Utf8)).filter(pl.col(tc).dt.month().is_in([1, 7]))
num = [c for c in w.columns if w[c].dtype.is_numeric()]
print(w.group_by("per").agg([pl.col(c).mean() for c in num]).sort("per"))
