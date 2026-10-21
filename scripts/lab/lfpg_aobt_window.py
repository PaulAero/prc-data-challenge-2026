"""Matched departures whose NM AOBT_3 lies outside the 60-min LOBT window: the recorded off-block
cannot be the AOBT_3 pushback. Test rule: pred = take-off - LOBT - c. All airports, LFPG focus.
Evaluated on (a) jan+jul 8-model OOF ensemble, (b) full-year two-part parts (lab-year)."""
import glob, os, numpy as np, polars as pl
S = os.environ["LFPG_OUT"]
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)
N = 344_841
rows = []
for f in sorted(glob.glob("data/raw/training_2025-*.parquet")):
    x = pl.scan_parquet(f).filter((pl.col("PHASE_mvt") == "DEP") & pl.col("LOBT_flt").is_not_null()
        & ((pl.col("AOBT_3_flt") - pl.col("LOBT_flt")).dt.total_seconds() > 2400)).collect()
    rows.append(x)
d = pl.concat(rows).with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64),
    tl=(pl.col("MVT_TIME_UTC_mvt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64),
    ul=(pl.col("AOBT_3_flt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64),
    y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64), m=pl.col("MVT_TIME_UTC_mvt").dt.month())
d = d.with_columns(bl=pl.col("tl") - pl.col("y"))
print("rows ul>40min", d.height)
d = d.with_columns(ulb=(pl.col("ul") // 600 * 10).clip(40, 150))
print(d.group_by("ulb").agg(pl.len(), pl.col("bl").quantile(0.1).alias("bl10"), pl.col("bl").quantile(0.25).alias("bl25"), pl.col("bl").median().alias("bl50"),
      pl.col("bl").quantile(0.75).alias("bl75"), (pl.col("ADEP_mvt") == "LFPG").sum().alias("lfpg")).sort("ulb"))
print("ul>65min by airport", d.filter(pl.col("ul") > 3900).group_by("ADEP_mvt").agg(pl.len(), pl.col("bl").median().alias("bl50"), (pl.col("tl") - pl.col("ul")).median().alias("taxi_px")).sort("len", descending=True))
d.write_parquet(f"{S}/aobt_outside.parquet")

def floor_rule(pred, tl):
    return np.where(tl > 7200, np.maximum(pred, tl - 1200), pred)

# (a) jan+jul ensemble
o = pl.read_parquet(f"{S}/oof_ens.parquet").select("MVT_ID_mvt", "fold", "pred")
a = d.join(o, on="MVT_ID_mvt", how="inner")
# (b) full-year parts
p = pl.read_parquet("experiments/lab-year/parts.parquet").select("MVT_ID_mvt", "block", "mvt_vs_schedule_sec", "reg", "p_base", "p_wsq", "win_lo", "win_hi")
b = d.join(p, on="MVT_ID_mvt", how="inner").with_columns(pf=(pl.col("p_base") + pl.col("p_wsq")) / 2)
b = b.with_columns(pred=(pl.col("pf") * pl.col("mvt_vs_schedule_sec") + (1 - pl.col("pf")) * pl.col("reg")).clip(pl.col("win_lo").clip(0), pl.col("win_hi")))
for name, df, grp in [("janjul-ens", a, "fold"), ("year-parts", b, "block")]:
    print("=====", name, df.height)
    for thr in [3600, 3900, 4200, 4800]:
        for c in ["cv", 1200, 1500, 1800]:
            tot = {"all": [0, 0, 0], "LFPG": [0, 0, 0]}
            blocks_better = 0; nb = 0
            for gval in df[grp].unique().to_list():
                te = df.filter((pl.col(grp) == gval) & (pl.col("ul") > thr))
                if te.height == 0:
                    continue
                if c == "cv":
                    tr = df.filter((pl.col(grp) != gval) & (pl.col("ul") > thr))
                    cc = float(tr["bl"].median())
                else:
                    cc = c
                y = te["y"].to_numpy(); tl = te["tl"].to_numpy(); pr = floor_rule(te["pred"].to_numpy(), tl)
                new = np.maximum(tl - cc, 0)
                dse = ((new - y) ** 2 - (pr - y) ** 2)
                nb += 1; blocks_better += dse.sum() < 0
                for k, msk in [("all", np.ones(len(y), bool)), ("LFPG", (te["ADEP_mvt"] == "LFPG").to_numpy())]:
                    tot[k][0] += msk.sum(); tot[k][1] += dse[msk].sum(); tot[k][2] += ((pr - y) ** 2)[msk].sum()
            print(f"thr {thr} c {c}: all n={tot['all'][0]} dSSE/N {tot['all'][1]/N:+.0f} (base {tot['all'][2]/N:.0f}); LFPG n={tot['LFPG'][0]} {tot['LFPG'][1]/N:+.0f} (base {tot['LFPG'][2]/N:.0f}); blocks better {blocks_better}/{nb}")
