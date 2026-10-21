"""Scouting run 2 (not protocol-grade): stand occupancy + runway busy period + fill-regime two-part model."""
# Run: PYTHONPATH=scripts/scouting uv run python scripts/scouting/exp009_stand_busy_fill.py <outdir>
import glob, sys, time
import numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0, 'src')
from prc.models.flight_features import load_departure_features, CATEGORICAL_FEATURES, NUMERIC_FEATURES
SP = sys.argv[1]
from queue_feats import build_seq  # same features as EXP-008

paths = sorted(glob.glob('data/raw/training_*.parquet'))
base = load_departure_features(paths)
raw = pl.scan_parquet(paths).select('MVT_ID_mvt','PHASE_mvt','ADEP_mvt','ADES_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','AOBT_3_flt','RUNWAY_mvt','STAND_mvt','FLIGHT_ID_mvt','SCHED_TIME_UTC_mvt','TAXITIME_SEC_mvt').collect()
ep = lambda c: pl.col(c).dt.epoch('s')
dep = raw.filter(pl.col('PHASE_mvt')=='DEP').with_columns(ep('MVT_TIME_UTC_mvt').alias('t'), ep('AOBT_3_flt').alias('a'), ep('SCHED_TIME_UTC_mvt').alias('s'))
arr = raw.filter(pl.col('PHASE_mvt')=='ARR').with_columns(ep('MVT_TIME_UTC_mvt').alias('t'), ep('BLOCK_TIME_UTC_mvt').alias('u'))
seq = build_seq(dep, arr)

# stand occupancy features (all ranking-available: arrivals keep in-block time and stand)
st = []
for (ap,), g in dep.group_by(['ADEP_mvt']):
    A = arr.filter(pl.col('ADES_mvt')==ap).drop_nulls(['STAND_mvt','u'])
    f = {k: np.full(g.height, np.nan) for k in ['st_gap_last_arr','st_gap_2nd_arr','st_n_arr_since_prevdep','st_gap_prevdep','st_next_arr_after']}
    f['MVT_ID_mvt'] = g['MVT_ID_mvt'].to_numpy()
    gs = g.with_row_index('i').drop_nulls('STAND_mvt')
    amap = {s: np.sort(x['u'].to_numpy()) for (s,), x in A.group_by(['STAND_mvt'])}
    for (s,), x in gs.group_by(['STAND_mvt']):
        idx = x['i'].to_numpy(); t = x['t'].to_numpy(); o = np.argsort(t); idx, t = idx[o], t[o]
        u = amap.get(s, np.array([], dtype=np.int64))
        k = np.searchsorted(u, t, 'left')  # arrivals strictly before t: u[:k]
        last = np.where(k >= 1, u[np.maximum(k-1, 0)] if len(u) else 0, np.nan)
        sec = np.where(k >= 2, u[np.maximum(k-2, 0)] if len(u) else 0, np.nan)
        nxt = np.where(k < len(u), u[np.minimum(k, max(len(u)-1, 0))] if len(u) else 0, np.nan)
        prevdep = np.concatenate([[np.nan], t[:-1]])
        kp = np.searchsorted(u, np.nan_to_num(prevdep, nan=-1), 'right')
        f['st_gap_last_arr'][idx] = t - last
        f['st_gap_2nd_arr'][idx] = t - sec
        f['st_next_arr_after'][idx] = nxt - t
        f['st_gap_prevdep'][idx] = t - prevdep
        f['st_n_arr_since_prevdep'][idx] = np.where(np.isnan(prevdep), np.nan, k - kp)
    st.append(pl.DataFrame(f))
st = pl.concat(st, how='vertical_relaxed')

# runway busy-period features: time since the runway last had an idle gap > thr, before my take-off
bp = []
for (ap, r), g in dep.group_by(['ADEP_mvt', 'RUNWAY_mvt']):
    g = g.sort('t'); t = g['t'].to_numpy().astype(float); gap = np.diff(t, prepend=-1e12)
    f = {'MVT_ID_mvt': g['MVT_ID_mvt'].to_numpy()}
    for thr in (90, 150, 300):
        start = np.maximum.accumulate(np.where(gap > thr, t, -np.inf))
        f[f'rw_busy_len_{thr}'] = t - start
        pos = np.arange(len(t)); lastbreak = np.maximum.accumulate(np.where(gap > thr, pos, 0))
        f[f'rw_busy_pos_{thr}'] = pos - lastbreak
    bp.append(pl.DataFrame(f))
bp = pl.concat(bp, how='vertical_relaxed')

extra = dep.select('MVT_ID_mvt', pl.col('FLIGHT_ID_mvt').is_null().cast(pl.Float32).alias('no_nm'),
                   ((pl.col('t') - pl.col('s'))).cast(pl.Float32).alias('ms'),
                   (((pl.col('BLOCK_TIME_UTC_mvt') - pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds().abs() <= 10)).cast(pl.Int8).alias('fill'))
df = base.join(seq, on='MVT_ID_mvt').join(st, on='MVT_ID_mvt').join(bp, on='MVT_ID_mvt').join(extra, on='MVT_ID_mvt')
df = df.with_columns([pl.col(c).cast(pl.Float32) for c in st.columns + bp.columns if c != 'MVT_ID_mvt'])
SEQ = [c for c in seq.columns if c != 'MVT_ID_mvt' and not c.startswith('orc_')]
ST = [c for c in st.columns if c != 'MVT_ID_mvt']; BP = [c for c in bp.columns if c != 'MVT_ID_mvt']
q4 = pl.col('event_time').dt.month() >= 10
trn, val = df.filter(~q4), df.filter(q4)
clean = (val['no_nm'] == 0).to_numpy() & (val['target'] <= 7200).to_numpy()
lir = (val['airport'] == 'LIRF').to_numpy()

def prep(frame, cols, ref=None):
    pdf = frame.select(cols).to_pandas()
    for c in CATEGORICAL_FEATURES:
        if c in pdf:
            pdf[c] = pdf[c].astype('category')
            if ref is not None: pdf[c] = pdf[c].cat.set_categories(ref[c].cat.categories)
    return pdf
P = dict(learning_rate=0.05, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
         cat_smooth=10, max_cat_to_onehot=8, num_threads=12, seed=20260926, verbose=-1)
ytr = trn['target'].to_numpy(); yv = val['target'].to_numpy()
def report(name, p, t0):
    se = (p - yv) ** 2
    print(f'{name:28s} RMSE all {se.mean()**.5:7.2f} | clean {se[clean].mean()**.5:7.2f} | clean excl LIRF {se[clean & ~lir].mean()**.5:7.2f} | LIRF all {se[lir].mean()**.5:7.1f} | {time.time()-t0:4.0f}s', flush=True)
    np.save(f'{SP}/pred2_{name}.npy', p)

B = CATEGORICAL_FEATURES + NUMERIC_FEATURES + ['no_nm'] + SEQ
arms = {'B1_seq': B, 'B2_seq+stand': B + ST, 'B3_seq+busy': B + BP, 'B4_seq+stand+busy': B + ST + BP}
for name, cols in arms.items():
    t0 = time.time(); Xtr = prep(trn, cols); Xv = prep(val, cols, Xtr)
    m = lgb.train(dict(P, objective='regression'), lgb.Dataset(Xtr, ytr), 1500)
    report(name, m.predict(Xv), t0)
    if name == 'B4_seq+stand+busy':
        imp = sorted(zip(m.feature_importance('gain'), cols), reverse=True)[:15]; print('   top gain:', [c for _, c in imp])
# two-part fill model on B4 features: p(fill) * ms + (1-p) * E[y | not fill]
cols = B + ST + BP; t0 = time.time()
Xtr = prep(trn, cols); Xv = prep(val, cols, Xtr); ftr = trn['fill'].to_numpy()
clf = lgb.train(dict(P, objective='binary'), lgb.Dataset(Xtr, ftr), 800)
reg = lgb.train(dict(P, objective='regression'), lgb.Dataset(Xtr[ftr == 0], ytr[ftr == 0]), 1500)
pf = clf.predict(Xv); msv = val['ms'].to_numpy()
from sklearn.metrics import roc_auc_score
print('fill AUC', roc_auc_score(val['fill'].to_numpy(), pf), 'val fill rate', val['fill'].mean())
report('C1_twopart_fill', pf * msv + (1 - pf) * reg.predict(Xv), t0)
