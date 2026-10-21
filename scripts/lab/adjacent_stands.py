"""EXP-048: arrivals at ADJACENT stands as observed pushback evidence.

A departure pushing back blocks the taxilane behind neighbouring stands; arrivals heading there
wait, which shows as an excess taxi-in that ends shortly after the pushback. Arrivals' taxi-in and
in-block are observed in the ranking file, so these are legitimate, label-free features.

Features for departure i (take-off T, stand at OSM coordinates from the geo block):
- next arrival at my own stand: its taxi-in excess (over the airport x stand x runway median) and its
  in-block minus my take-off (did it wait for me?);
- arrivals at other stands within R m, in-block in [T - 45 min, T]: count, max taxi-in excess, time
  of that arrival's in-block relative to T, count with excess > 180 s.
Test: one LightGBM (clip 3 h, ff 0.5 + Huber 900), blocks + latent, with vs without, jan+jul,
ordinary rows (NM non-fill <= 2 h).
"""
import json
import os
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "4")
import lightgbm as lgb
import numpy as np
import polars as pl

from prc.data import load_ranking, load_training, mask_departures
from prc.features.flight import block_columns
from prc.models.gbdt import DEFAULT_PARAMS, Frame, proxy_offset
from prc.validation.protocol import FOLDS

FEAT = Path("data/processed/features")
LAT = "data/processed/latent_set_noarr_wx_avg3/{fold}.parquet"
BLOCKS = ["base", "airline", "seq", "stand", "schedrel", "wx", "geo", "pxdev"]
THREADS = int(os.environ.get("LAB_THREADS", "4"))
OUT = Path("experiments/lab-adjacent"); OUT.mkdir(parents=True, exist_ok=True)


def build(radius: float = 120.0) -> pl.DataFrame:
    mv = mask_departures(pl.concat([load_training(), load_ranking()], how="vertical_relaxed"))
    feats = pl.concat([pl.read_parquet(FEAT / "train.parquet", columns=["MVT_ID_mvt", "airport", "stand", "geo_stand_x", "geo_stand_y"]),
                       pl.read_parquet(FEAT / "ranking.parquet", columns=["MVT_ID_mvt", "airport", "stand", "geo_stand_x", "geo_stand_y"])])
    coords = feats.drop_nulls(["geo_stand_x"]).group_by("airport", "stand").agg(pl.col("geo_stand_x").first().alias("x"),
                                                                                pl.col("geo_stand_y").first().alias("y"))
    arr = mv.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("BLOCK_TIME_UTC_mvt").is_not_null()).select(
        pl.col("ADES_mvt").alias("airport"), pl.col("STAND_mvt").alias("stand"), pl.col("RUNWAY_mvt").alias("rw"),
        pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("s").alias("u"), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("ti"))
    arr = arr.filter(pl.col("ti").is_between(0, 3600))
    arr = arr.with_columns((pl.col("ti") - pl.col("ti").median().over("airport", "stand", "rw")).alias("xs"))
    arr = arr.join(coords, on=["airport", "stand"], how="left")
    dep = feats.join(mv.filter(pl.col("PHASE_mvt") == "DEP").select(pl.col("MVT_ID_mvt").cast(pl.Int64),
                     pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").alias("t")), on="MVT_ID_mvt")
    out = []
    for (ap,), g in dep.group_by(["airport"]):
        A = arr.filter(pl.col("airport") == ap).sort("u")
        au, ax, ay, axs, ast = (A[c].to_numpy() for c in ("u", "x", "y", "xs", "stand"))
        n = g.height
        f = {k: np.full(n, np.nan) for k in ("adj_n", "adj_max_xs", "adj_max_dt", "adj_n_wait", "own_next_xs", "own_next_dt")}
        t, gx, gy, gs = (g[c].to_numpy() for c in ("t", "geo_stand_x", "geo_stand_y", "stand"))
        lo_i, hi_i = np.searchsorted(au, t - 2700, "left"), np.searchsorted(au, t, "right")
        own = {}
        for s in np.unique(ast[ast != None]):  # noqa: E711
            own[s] = np.where(ast == s)[0]
        for i in range(n):
            if gs[i] is not None and gs[i] in own:
                k = own[gs[i]]; j = np.searchsorted(au[k], t[i] - 3600, "left")  # first in-block after T - 1 h
                if j < len(k):
                    f["own_next_xs"][i] = axs[k[j]]; f["own_next_dt"][i] = au[k[j]] - t[i]
            if np.isnan(gx[i]) or hi_i[i] <= lo_i[i]:
                continue
            sl = slice(lo_i[i], hi_i[i])
            d = np.hypot(ax[sl] - gx[i], ay[sl] - gy[i])
            m = (d <= radius) & (ast[sl] != gs[i]) & ~np.isnan(axs[sl])
            if not m.any():
                f["adj_n"][i] = 0; continue
            xs = axs[sl][m]; uu = au[sl][m]
            f["adj_n"][i] = m.sum(); j = np.argmax(xs)
            f["adj_max_xs"][i] = xs[j]; f["adj_max_dt"][i] = t[i] - uu[j]; f["adj_n_wait"][i] = (xs > 180).sum()
        out.append(pl.DataFrame({"MVT_ID_mvt": g["MVT_ID_mvt"].to_numpy(), **f}))
    return pl.concat(out)


def main():
    adj = build()
    adj.write_parquet(OUT / "adjacent.parquet")
    print("features", adj.shape, "coverage adj_n>0", float((adj["adj_n"] > 0).mean()), flush=True)
    base = pl.read_parquet(FEAT / "train.parquet").join(pl.read_parquet(FEAT / "labels.parquet"), on="MVT_ID_mvt")
    b = block_columns(base); blocks = list(dict.fromkeys(c for k in BLOCKS for c in b[k]))
    acols = [c for c in adj.columns if c != "MVT_ID_mvt"]
    res = {}
    for arm in ("ref", "adjacent"):
        sse = n = 0.0
        for fold in ("jan", "jul"):
            lat = pl.read_parquet(LAT.format(fold=fold))
            f = base.join(lat, on="MVT_ID_mvt", how="left").join(adj, on="MVT_ID_mvt", how="left")
            cols = blocks + [c for c in lat.columns if c != "MVT_ID_mvt"] + (acols if arm == "adjacent" else [])
            F = FOLDS[fold]
            tr = f.filter(F.train() & (pl.col("fill") == 0)); ho = f.filter(F.holdout())
            fr = Frame(cols)
            prm = dict(DEFAULT_PARAMS, seed=20260926, objective="huber", alpha=900.0, feature_fraction=0.5, num_threads=THREADS)
            m = lgb.train(prm, lgb.Dataset(fr.fit_transform(tr), np.minimum(tr["target"].to_numpy(), 10800),
                                           init_score=proxy_offset(tr)), 1500)
            pred = m.predict(fr.transform(ho)) + proxy_offset(ho)
            mask = ((ho["no_nm"] == 0) & (ho["fill"] == 0) & (ho["target"] <= 7200)).to_numpy()
            e = pred[mask] - ho["target"].to_numpy()[mask]; sse += float((e ** 2).sum()); n += mask.sum()
            if arm == "adjacent" and fold == "jan":
                g = dict(zip(cols, m.feature_importance("gain"))); tot = sum(g.values())
                print("adjacent gain share", {c: round(g[c] / tot, 4) for c in acols}, flush=True)
        res[arm] = float(np.sqrt(sse / n)); print(f"{arm:9s} ordinary RMSE {res[arm]:.2f}", flush=True)
    (OUT / "metrics.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
