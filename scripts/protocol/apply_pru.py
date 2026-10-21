"""EXP-055: shift airport-month means toward EUROCONTROL's published taxi-out averages.

Published ATXOT means (taxi_out_additional_time_<year>.csv, ansperformance.eu; use confirmed by the
organisers, 2026-09-27; source: EUROCONTROL) cover our population (stand, runway, taxi-out
<= 120 min, local months). Implied 2026 mean = published 2026 + (our 2025 truth - published 2025), same
calendar month. Rule fixed before any upload: only airports where the 2025 gap transfers from a
neighbouring month within 7 s rms (EGLL, LEBL, EHAM, LIRF, LEMD; not the de-icing airports EDDF,
EDDM, LSZH, LFPG, LTFM), only cells with |bias| > 2 x that rms, shift = -0.6 x bias on the rows of the
population (stand and runway present, prediction <= 7,200 s).

Usage: uv run python scripts/protocol/apply_pru.py <candidate.parquet> <out-name>
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import polars as pl
sys.path.insert(0, str(Path(__file__).parents[1] / "lab"))
import pru_calib as P  # noqa: E402
import glob  # noqa: E402
from prc.data import load_movements  # noqa: E402

RELIABLE = {"EGLL": 0.8, "LEBL": 1.5, "EHAM": 3.7, "LIRF": 5.3, "LEMD": 6.1}  # 2025 gap-transfer rms (s)
SHRINK = 0.6


def main(cand: str, out: str) -> None:
    tr = pl.concat([load_movements(p) for p in sorted(glob.glob("data/raw/training_2025-*.parquet"))], how="diagonal_relaxed")
    tr = tr.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("STAND_mvt").is_not_null() & pl.col("RUNWAY_mvt").is_not_null()
                   & (pl.col("TAXITIME_SEC_mvt") <= 7200)).select(pl.col("ADEP_mvt").alias("ap"), pl.col("MVT_TIME_UTC_mvt").alias("t"),
                   pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"))
    gap = P.local_month(tr).group_by("ap", "m").agg(pl.col("y").mean().alias("true2025")).join(P.pru(2025), on=["ap", "m"]).select(
        "ap", "m", (pl.col("true2025") - pl.col("pru2025")).alias("gap"))
    sub = pl.read_parquet(cand)
    rk = load_movements("data/raw/ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP").select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("ADEP_mvt").alias("ap"), pl.col("MVT_TIME_UTC_mvt").alias("t"),
        (pl.col("STAND_mvt").is_not_null() & pl.col("RUNWAY_mvt").is_not_null()).alias("has_sr"))
    x = P.local_month(sub.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64).alias("_id")).join(
        rk.rename({"MVT_ID_mvt": "_id"}), on="_id", how="left"))
    x = sub.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64).alias("_id")).join(x.select("_id", "ap", "m", "has_sr"), on="_id", how="left")
    pred = x["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    inpop = (x["has_sr"].to_numpy() & (pred <= 7200))
    x = x.with_columns(pl.Series("pred", pred), pl.Series("inpop", inpop))
    cells = x.filter(pl.col("inpop")).group_by("ap", "m").agg(pl.col("pred").mean().alias("pred_mean"), pl.len().alias("n")).join(
        P.pru(2026), on=["ap", "m"]).join(gap, on=["ap", "m"]).with_columns(
        (pl.col("pred_mean") - pl.col("pru2026") - pl.col("gap")).alias("bias"))
    cells = cells.with_columns(pl.col("ap").replace_strict(RELIABLE, default=None).alias("rms")).with_columns(
        pl.when(pl.col("rms").is_not_null() & (pl.col("bias").abs() > 2 * pl.col("rms"))).then(-SHRINK * pl.col("bias")).otherwise(0.0).alias("shift"))
    x = x.join(cells.select("ap", "m", "shift"), on=["ap", "m"], how="left").with_columns(pl.col("shift").fill_null(0.0))
    new = np.where(x["inpop"].to_numpy(), pred + x["shift"].to_numpy(), pred)
    assert x["_id"].equals(sub["MVT_ID_mvt"].cast(pl.Int64))
    res = sub.with_columns(pl.Series("TAXITIME_SEC_mvt", np.rint(np.maximum(new, 0)).astype(np.int32)))
    path = Path("data/submissions") / f"candidate_{out}.parquet"
    res.write_parquet(path)
    rep = cells.filter(pl.col("shift") != 0).select("ap", "m", "n", pl.col("bias").round(1), pl.col("shift").round(1)).sort("ap", "m")
    print(rep)
    Path(str(path).replace(".parquet", ".check.json")).write_text(json.dumps(
        {"file": str(path), "base": cand, "rows": res.height, "shifts": rep.to_dicts()}, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
