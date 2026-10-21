import polars as pl
d = pl.scan_parquet("experiments/lab-year/parts.parquet")
print(d.collect_schema())
x = d.filter(pl.col("airport") == "LTFM").collect() if "airport" in d.collect_schema().names() else None
if x is not None: print(x.shape); print(x.head(3)); print(x.select([c for c in x.columns if x[c].dtype == pl.String][:3]).head(2))
