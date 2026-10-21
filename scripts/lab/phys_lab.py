"""Physics lab (EXP-037): does the non-fill regression gain from target encodings, inbound rotation
features, or other LightGBM settings? Regression only, one seed, jan + jul folds; scored on clean
non-fill holdout rows (the physics population) and, with a fixed fill probability from
experiments/lab-fill/parts.parquet, on core.

Usage: uv run python scripts/lab/phys_lab.py <out_dir> <arm> [<arm> ...]
Arms: ref, te, rot, te_rot, and hyper-parameter arms (see ARMS).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "12")
import lightgbm as lgb
import numpy as np
import polars as pl

from prc.features.flight import block_columns
from prc.models.gbdt import DEFAULT_PARAMS, Frame, proxy_offset
from prc.validation.protocol import FOLDS

FEAT = Path("data/processed/features")
PHYS = Path(os.environ.get("LAB_PHYS", "data/processed/lab_phys"))
LATENT = "data/processed/latent_set_noarr_wx_avg3/{fold}.parquet"
BLOCKS = ["base", "airline", "seq", "stand", "schedrel", "wx", "geo", "pxdev"]
THREADS = int(os.environ.get("LAB_THREADS", "12"))
YCLIP = float(os.environ.get("LAB_YCLIP", "10800"))

ARMS = {
    "ref": dict(te=False, rot=False, params={}, rounds=1500),
    "te": dict(te=True, rot=False, params={}, rounds=1500),
    "rot": dict(te=False, rot=True, params={}, rounds=1500),
    "te_rot": dict(te=True, rot=True, params={}, rounds=1500),
    "slow": dict(te=False, rot=False, params=dict(learning_rate=0.025), rounds=3000),
    "l127": dict(te=False, rot=False, params=dict(num_leaves=127, min_data_in_leaf=50), rounds=2500),
    "l511": dict(te=False, rot=False, params=dict(num_leaves=511, min_data_in_leaf=200), rounds=1200),
    "ff5": dict(te=False, rot=False, params=dict(feature_fraction=0.5), rounds=1500),
    "huber": dict(te=False, rot=False, params=dict(objective="huber", alpha=900.0), rounds=1500),
    "ff3": dict(te=False, rot=False, params=dict(feature_fraction=0.3), rounds=1500),
    "ff5_hub": dict(te=False, rot=False, params=dict(feature_fraction=0.5, objective="huber", alpha=900.0), rounds=1500),
    "ff5_hub600": dict(te=False, rot=False, params=dict(feature_fraction=0.5, objective="huber", alpha=600.0), rounds=1500),
    "ff5_hub1500": dict(te=False, rot=False, params=dict(feature_fraction=0.5, objective="huber", alpha=1500.0), rounds=1500),
    "ff5_hub_slow": dict(te=False, rot=False, params=dict(feature_fraction=0.5, objective="huber", alpha=900.0,
                                                           learning_rate=0.025), rounds=3000),
    # CatBoost regression arms (EXP-043): v8/v10 CatBoost is depth 8, lr 0.08, 3,000 iterations
    "cb_ref": dict(te=False, rot=False, cb=dict(depth=8, learning_rate=0.08, rounds=3000)),
    "cb_d10": dict(te=False, rot=False, cb=dict(depth=10, learning_rate=0.08, rounds=3000)),
    "cb_slow": dict(te=False, rot=False, cb=dict(depth=8, learning_rate=0.04, rounds=6000)),
    "cb_l2": dict(te=False, rot=False, cb=dict(depth=8, learning_rate=0.08, rounds=3000, l2_leaf_reg=10.0)),
}


def load(fold: str, arm: dict):
    f = pl.read_parquet(FEAT / "train.parquet").join(pl.read_parquet(FEAT / "labels.parquet"), on="MVT_ID_mvt")
    b = block_columns(f)
    cols = [c for k in BLOCKS for c in b[k]]
    extras = [LATENT.format(fold=fold)]
    if arm["te"]:
        extras.append(str(PHYS / f"te_{fold}.parquet"))
    if arm["rot"]:
        extras.append(str(PHYS / "rot.parquet"))
    for e in extras:
        ex = pl.read_parquet(e)
        f = f.join(ex, on="MVT_ID_mvt", how="left")
        cols += [c for c in ex.columns if c != "MVT_ID_mvt"]
    return f, list(dict.fromkeys(cols))


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    parts = pl.read_parquet("experiments/lab-fill/parts.parquet").select("MVT_ID_mvt", "p_base", "q24")
    for arm_name in sys.argv[2:]:
        arm = ARMS[arm_name]
        rows = []
        t0 = time.time()
        for fold in ("jan", "jul"):
            frame, cols = load(fold, arm)
            F = FOLDS[fold]
            train, hold = frame.filter(F.train()), frame.filter(F.holdout())
            tr = train.filter(pl.col("fill") == 0)
            y = np.minimum(tr["target"].to_numpy(), YCLIP)
            if "cb" in arm:
                from catboost import CatBoostRegressor, Pool
                from prc.features.flight import CATEGORICAL
                cat = [c for c in cols if c in CATEGORICAL]
                def X(fr_):
                    pdf = fr_.select(cols).to_pandas()
                    for c in cat:
                        pdf[c] = pdf[c].astype(str)
                    return pdf
                c = dict(arm["cb"]); rounds = c.pop("rounds")
                m = CatBoostRegressor(iterations=rounds, loss_function="RMSE", random_seed=20260926, thread_count=THREADS,
                                      verbose=0, allow_writing_files=False, **c)
                m.fit(Pool(X(tr), y, cat_features=cat, baseline=proxy_offset(tr)))
                reg = m.predict(X(hold)) + proxy_offset(hold)
                print(arm_name, fold, "fitted", flush=True)
                rows.append(hold.select("MVT_ID_mvt", "target", "fill", "no_nm", "airport", "mvt_vs_schedule_sec")
                            .with_columns(pl.Series("reg", reg), pl.lit(fold).alias("fold")))
                continue
            fr = Frame(cols)
            params = dict(DEFAULT_PARAMS, seed=20260926, objective="regression", num_threads=THREADS)
            params.update(arm["params"])
            m = lgb.train(params, lgb.Dataset(fr.fit_transform(tr), y, init_score=proxy_offset(tr)), arm["rounds"])
            reg = m.predict(fr.transform(hold)) + proxy_offset(hold)
            rows.append(hold.select("MVT_ID_mvt", "target", "fill", "no_nm", "airport", "mvt_vs_schedule_sec")
                        .with_columns(pl.Series("reg", reg), pl.lit(fold).alias("fold")))
            imp = sorted(zip(m.feature_importance("gain"), cols), reverse=True)[:12]
            print(arm_name, fold, "top", [c for _, c in imp], flush=True)
        p = pl.concat(rows).join(parts, on="MVT_ID_mvt", how="left")
        p.write_parquet(out / f"{arm_name}.parquet")
        y = p["target"].to_numpy(); reg = p["reg"].to_numpy(); ms = p["mvt_vs_schedule_sec"].fill_null(0).to_numpy()
        pf = p["p_base"].to_numpy(); q = p["q24"].to_numpy()
        mix = q * (86400 + reg) + (1 - q) * (pf * ms + (1 - pf) * reg)
        phys = (p["no_nm"].to_numpy() == 0) & (y <= 7200) & (p["fill"].to_numpy() == 0)
        core = y <= 10800
        clean = (p["no_nm"].to_numpy() == 0) & (y <= 7200)
        r = dict(phys=float(np.sqrt(np.mean((reg[phys] - y[phys]) ** 2))),
                 core=float(np.sqrt(np.mean((mix[core] - y[core]) ** 2))),
                 clean=float(np.sqrt(np.mean((mix[clean] - y[clean]) ** 2))),
                 all=float(np.sqrt(np.mean((mix - y) ** 2))), minutes=round((time.time() - t0) / 60, 1))
        ap = p.filter(pl.Series(phys)).group_by("airport").agg(((pl.col("reg") - pl.col("target")) ** 2).mean().sqrt().alias("r")).sort("airport")
        r["phys_by_airport"] = dict(zip(ap["airport"].to_list(), [round(v, 1) for v in ap["r"].to_list()]))
        print(arm_name, json.dumps(r), flush=True)
        (out / f"{arm_name}.json").write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
