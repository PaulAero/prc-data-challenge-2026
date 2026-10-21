"""Candidate physics features for the non-fill regression (EXP-037).

1. Out-of-fold target encodings of recurring identities (stand x runway, flight number, airline x
   runway x hour, aircraft type x runway, destination), fitted on non-fill training departures with
   the target clipped at 3,600 s, day-grouped 5-fold on training rows, full training rows for the
   holdout. Label-derived, so written per fold: <out>/te_{fold}.parquet.
2. Inbound rotation (label-free): latest arrival at the same stand, same airline prefix, blocking in
   before take-off - 10 min (20 h tolerance); stand-only fallback. Written once: <out>/rot.parquet.

Usage: uv run python scripts/lab/phys_feats.py <out_dir> [folds...]   (folds: jan jul q4 final)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "12")
import numpy as np
import polars as pl

from prc.data import load_ranking, load_training, mask_departures
from prc.validation.protocol import FOLDS

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/processed/lab_phys")
OUT.mkdir(parents=True, exist_ok=True)
FOLD_NAMES = sys.argv[2:] or ["jan", "jul", "final"]
FEAT = Path("data/processed/features")
K_SMOOTH = 20.0
YCLIP = 3600.0

GROUPS = {
    "te_stand_rwy": ["airport", "stand", "runway"],
    "te_stand": ["airport", "stand"],
    "te_flt": ["airport", "flight"],
    "te_flt_rwy": ["airport", "flight", "runway"],
    "te_al_rwy_hr": ["airport", "airline", "runway", "local_hour"],
    "te_type_rwy": ["airport", "aircraft_type", "runway"],
    "te_dest": ["airport", "destination"],
}


def base_frame() -> pl.DataFrame:
    tr = pl.read_parquet(FEAT / "train.parquet").join(pl.read_parquet(FEAT / "labels.parquet"), on="MVT_ID_mvt")
    rk = pl.read_parquet(FEAT / "ranking.parquet").with_columns(pl.lit(None, dtype=pl.Float64).alias("target"),
                                                                pl.lit(None, dtype=pl.Int8).alias("fill"))
    cols = ["MVT_ID_mvt", "event_time", "airport", "stand", "runway", "airline", "local_hour", "aircraft_type",
            "destination", "target", "fill"]
    f = pl.concat([tr.select(cols), rk.select(cols)], how="vertical_relaxed")
    mv = pl.concat([load_training(), load_ranking()], how="vertical_relaxed").filter(pl.col("PHASE_mvt") == "DEP")
    return f.join(mv.select("MVT_ID_mvt", pl.col("FLIGHT_mvt").fill_null("<NA>").alias("flight")), on="MVT_ID_mvt", how="left")


def encode(fit: pl.DataFrame, apply: pl.DataFrame) -> pl.DataFrame:
    fit = fit.filter((pl.col("fill") == 0) & pl.col("target").is_not_null()).with_columns(
        pl.col("target").clip(0, YCLIP).alias("_y"))
    prior = fit.group_by("airport", "runway").agg(pl.col("_y").mean().alias("_prior"))
    need = list(dict.fromkeys(["MVT_ID_mvt", "airport", "runway"] + [c for g in GROUPS.values() for c in g]))
    out = apply.select(need).join(
        prior, on=["airport", "runway"], how="left").with_columns(pl.col("_prior").fill_null(900.0))
    res = out.select("MVT_ID_mvt")
    for name, keys in GROUPS.items():
        st = fit.group_by(keys).agg(pl.col("_y").sum().alias("_s"), pl.len().alias("_n"))
        j = out.join(st, on=keys, how="left").with_columns(pl.col("_s").fill_null(0.0), pl.col("_n").fill_null(0))
        j = j.select("MVT_ID_mvt", ((pl.col("_s") + K_SMOOTH * pl.col("_prior")) / (pl.col("_n") + K_SMOOTH)
                                    - pl.col("_prior")).cast(pl.Float32).alias(name),
                     pl.col("_n").cast(pl.Float32).alias(name + "_n"))
        res = res.join(j, on="MVT_ID_mvt", how="left")
    return res


def target_encodings(f: pl.DataFrame, fold: str) -> pl.DataFrame:
    if fold == "final":
        is_train = pl.col("target").is_not_null()
        is_target = pl.col("target").is_null()
    else:
        F = FOLDS[fold]
        is_train = F.train() & pl.col("target").is_not_null()
        is_target = ~F.train() | pl.col("target").is_null()
    train = f.filter(is_train).with_columns((pl.col("event_time").dt.date().hash(7) % 5).alias("_k"))
    parts = [encode(train.filter(pl.col("_k") != k), train.filter(pl.col("_k") == k)) for k in range(5)]
    parts.append(encode(train, f.filter(is_target)))
    return pl.concat(parts)


def rotation() -> pl.DataFrame:
    mv = mask_departures(pl.concat([load_training(), load_ranking()], how="vertical_relaxed"))
    al = pl.coalesce(pl.col("FLIGHT_mvt").str.extract(r"^([A-Z]{3})", 1),
                     pl.col("FLIGHT_mvt").str.extract(r"^([A-Z0-9]{2})[A-Z]?\d", 1)).fill_null("?")
    dep = mv.filter(pl.col("PHASE_mvt") == "DEP").select(
        "MVT_ID_mvt", pl.col("ADEP_mvt").alias("ap"), pl.col("STAND_mvt").alias("stand"), al.alias("al"),
        pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").alias("t"), pl.col("SCHED_TIME_UTC_mvt").dt.epoch("s").alias("s"))
    arr = mv.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("BLOCK_TIME_UTC_mvt").is_not_null()).select(
        pl.col("ADES_mvt").alias("ap"), pl.col("STAND_mvt").alias("stand"), al.alias("al"),
        pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("s").alias("u"), pl.col("SCHED_TIME_UTC_mvt").dt.epoch("s").alias("us"),
        pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").alias("land"), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("ti"))
    out = []
    for same_al in (True, False):
        keys = ["ap", "stand", "al"] if same_al else ["ap", "stand"]
        d = dep.drop_nulls("stand").with_columns((pl.col("t") - 600).alias("key")).sort("key")
        a = arr.drop_nulls("stand").rename({"u": "key"}).with_columns(pl.col("key").alias("u")).sort("key")
        if not same_al:
            a = a.drop("al")
        j = d.join_asof(a, on="key", by=keys, strategy="backward", tolerance=72000)
        tag = "rot" if same_al else "rotst"
        j = j.select("MVT_ID_mvt",
                     (pl.col("s") - pl.col("u")).cast(pl.Float32).alias(f"{tag}_slack"),       # schedule - inbound in-block
                     (pl.col("t") - pl.col("u")).cast(pl.Float32).alias(f"{tag}_to_minus_ib"),  # take-off - inbound in-block
                     (pl.col("u") - pl.col("us")).cast(pl.Float32).alias(f"{tag}_inb_late"),    # inbound in-block lateness
                     pl.col("ti").cast(pl.Float32).alias(f"{tag}_inb_taxiin"))
        out.append(j)
    return out[0].join(out[1], on="MVT_ID_mvt", how="left")


if __name__ == "__main__":
    if not (OUT / "rot.parquet").exists():
        rotation().write_parquet(OUT / "rot.parquet")
        print("rot done", flush=True)
    f = base_frame()
    for fold in FOLD_NAMES:
        te = target_encodings(f, fold)
        te.write_parquet(OUT / f"te_{fold}.parquet")
        print(fold, te.shape, flush=True)
