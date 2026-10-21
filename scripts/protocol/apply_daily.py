"""EXP-056: apply the daily pre-departure-delay calibration (scripts/lab/daily_delay.py) to a candidate.

Per airport and take-off day of 2026, the shift that makes the candidate reproduce EUROCONTROL's
published DLY_ALL_PRE_2 is solved, and half of it applied (lam = 0.5, chosen on 2025: core 228.48 ->
227.57, better in both months and at every airport). Rows predicted above 7,200 s are not moved. At LIRF the fill-aware
variant (EXP-056b) replaces the uniform shift.

Usage: uv run python scripts/protocol/apply_daily.py <candidate.parquet> <out-name>
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import polars as pl
sys.path.insert(0, str(Path(__file__).parents[1] / "lab"))
import daily_delay as D  # noqa: E402
from prc.data import load_movements  # noqa: E402

LAM = 0.5
FILL_LAM = 0.75  # EXP-056b, LIRF fill-aware variant (2025: LIRF all 522.5 -> 513.4, both months better)
LATENT_FINAL = "data/processed/latent_set_v3full_avg3/final.parquet"


def main(cand: str, out: str) -> None:
    sub = pl.read_parquet(cand)
    rk = load_movements("data/raw/ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP").select(
        pl.col("MVT_ID_mvt").cast(pl.Int64).alias("_id"), pl.col("ADEP_mvt").alias("airport"),
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("ts"),
        pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day"))
    x = sub.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64).alias("_id"), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("p")).join(
        rk, on="_id", how="left").join(D.published(2026), on=["airport", "day"], how="left")
    assert x["_id"].equals(sub["MVT_ID_mvt"].cast(pl.Int64))
    sh = D.solve_shifts(x.filter(pl.col("pub").is_not_null() & (pl.col("pub") > 0)), "p")
    new = D.apply(x, "p", sh, LAM)
    # LIRF: fill-aware daily calibration instead of the uniform shift
    lat = pl.read_parquet(LATENT_FINAL).select(pl.col("MVT_ID_mvt").cast(pl.Int64).alias("_id"), "nn_pred")
    L = x.filter((pl.col("airport") == "LIRF") & pl.col("pub").is_not_null() & (pl.col("pub") > 0)).join(lat, on="_id", how="left")
    fc = D.fill_calibrate(L.drop("MVT_ID_mvt").rename({"_id": "MVT_ID_mvt"}), "p", FILL_LAM).rename({"MVT_ID_mvt": "_id"})
    y = x.select("_id").with_columns(pl.Series("uni", new)).join(fc, on="_id", how="left")
    assert y["_id"].equals(x["_id"])
    new = y.with_columns(pl.coalesce("fc", "uni").alias("v"))["v"].to_numpy()
    res = sub.with_columns(pl.Series("TAXITIME_SEC_mvt", np.rint(np.maximum(new, 0)).astype(np.int32)))
    path = Path("data/submissions") / f"candidate_{out}.parquet"
    res.write_parquet(path)
    d = new - x["p"].to_numpy()
    summ = sh.group_by("airport").agg(pl.len().alias("days"), pl.col("shift").mean().round(1).alias("mean_shift"),
                                       pl.col("shift").std().round(1).alias("sd"), (pl.col("shift").abs() >= 899).sum().alias("clipped")).sort("airport")
    print(summ); print("rms change", float(np.sqrt((d ** 2).mean())), "days missing", x.filter(pl.col("pub").is_null())["day"].n_unique())
    Path(str(path).replace(".parquet", ".check.json")).write_text(json.dumps(
        {"file": str(path), "base": cand, "lam": LAM, "rms_change": float(np.sqrt((d ** 2).mean())), "per_airport": summ.to_dicts()}, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
