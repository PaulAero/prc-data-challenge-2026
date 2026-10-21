"""Cache LTFM movements (2025 train + 2026 ranking) and the 8-model OOF average for jan/jul 2025."""
import glob, polars as pl
OUT = "experiments/lab-ltfm"
import os; os.makedirs(OUT, exist_ok=True)
frames = []
for f in sorted(glob.glob("data/raw/training_2025-*.parquet")) + ["data/raw/ranking.parquet"]:
    d = pl.scan_parquet(f).filter((pl.col("ADEP_mvt") == "LTFM") | (pl.col("ADES_mvt") == "LTFM")).collect()
    d = d.with_columns(pl.lit("rank" if "ranking" in f else "train").alias("src"))
    frames.append(d)
df = pl.concat(frames, how="vertical_relaxed").with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))
# airport of the movement
df = df.with_columns(pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")).alias("apt"))
df = df.filter(pl.col("apt") == "LTFM")
print(df.group_by("src", "PHASE_mvt").len())
preds = None
for v in ["v10", "v11", "v12", "v15"]:
    for m in ["lgb", "cb"]:
        p = pl.read_parquet(f"experiments/{v}-{m}/predictions.parquet").filter(pl.col("airport") == "LTFM")
        p = p.select("MVT_ID_mvt", "fold", "target", pl.col("pred").alias(f"p_{v}{m}"))
        preds = p if preds is None else preds.join(p.drop("fold", "target"), on="MVT_ID_mvt", how="inner")
pc = [c for c in preds.columns if c.startswith("p_")]
preds = preds.with_columns(pl.mean_horizontal(pc).alias("pred")).select("MVT_ID_mvt", "fold", "target", "pred")
print(preds.height)
df = df.join(preds, on="MVT_ID_mvt", how="left")
df.write_parquet(f"{OUT}/ltfm.parquet")
print(df.height, df.filter(pl.col("pred").is_not_null()).height)
