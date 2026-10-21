"""Cache LFPG movements (2025 train + 2026 ranking) and the 8-model OOF average."""
import glob, os, polars as pl
OUT = os.environ.get("LFPG_OUT", "experiments/lab-lfpg")
os.makedirs(OUT, exist_ok=True)
files = sorted(glob.glob("data/raw/training_2025-*.parquet"))
frames = []
for f in files + ["data/raw/ranking.parquet"]:
    d = pl.scan_parquet(f).filter((pl.col("ADEP_mvt") == "LFPG") | (pl.col("ADES_mvt") == "LFPG")).collect()
    d = d.with_columns(pl.lit("rank" if "ranking" in f else "train").alias("src"),
                       pl.col("MVT_ID_mvt").cast(pl.Int64))
    frames.append(d)
allm = pl.concat(frames, how="diagonal_relaxed")
# airport of the movement
allm = allm.with_columns(pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")).alias("apt"))
allm = allm.filter(pl.col("apt") == "LFPG")
allm.write_parquet(f"{OUT}/lfpg_all.parquet")
print(allm.group_by("src", "PHASE_mvt").len())
ps = []
for v in ["v10", "v11", "v12", "v15"]:
    for m in ["lgb", "cb"]:
        p = pl.read_parquet(f"experiments/{v}-{m}/predictions.parquet").select("MVT_ID_mvt", "fold", "airport", "target", "pred", "no_nm", "fill")
        ps.append(p.rename({"pred": f"p_{v}_{m}"}))
base = ps[0]
for p in ps[1:]:
    base = base.join(p.select("MVT_ID_mvt", p.columns[4]), on="MVT_ID_mvt", how="inner")
pc = [c for c in base.columns if c.startswith("p_")]
base = base.with_columns(pl.mean_horizontal(pc).alias("pred"))
print(base["fold"].value_counts())
base.write_parquet(f"{OUT}/oof_ens.parquet")
