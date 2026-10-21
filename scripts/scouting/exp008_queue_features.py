"""Scouting run (not protocol-grade): do runway-sequence / queue features help? Q1-Q3 train, Q4 eval."""
# Run: uv run python scripts/scouting/exp008_queue_features.py <outdir>
import glob, sys, time
import numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0, 'src')
from prc.models.flight_features import load_departure_features, CATEGORICAL_FEATURES, NUMERIC_FEATURES

paths = sorted(glob.glob('data/raw/training_*.parquet'))
base = load_departure_features(paths)
raw = pl.scan_parquet(paths).select('MVT_ID_mvt','PHASE_mvt','ADEP_mvt','ADES_mvt','MVT_TIME_UTC_mvt','AOBT_3_flt','RUNWAY_mvt','FLIGHT_ID_mvt','SCHED_TIME_UTC_mvt','TAXITIME_SEC_mvt').collect()
ep = lambda c: pl.col(c).dt.epoch('s')
dep = raw.filter(pl.col('PHASE_mvt')=='DEP').with_columns(ep('MVT_TIME_UTC_mvt').alias('t'), ep('AOBT_3_flt').alias('a'), ep('SCHED_TIME_UTC_mvt').alias('s'))
arr = raw.filter(pl.col('PHASE_mvt')=='ARR').with_columns(ep('MVT_TIME_UTC_mvt').alias('t'))

def cnt(sorted_times, lo, hi):
    return np.searchsorted(sorted_times, hi, 'left') - np.searchsorted(sorted_times, lo, 'left')

out = []
for (ap,), g in dep.group_by(['ADEP_mvt']):
    g = g.sort('t'); t = g['t'].to_numpy(); a = g['a'].to_numpy().astype('float'); s = g['s'].to_numpy()
    y = g['TAXITIME_SEC_mvt'].to_numpy().astype(float)
    at = np.sort(arr.filter(pl.col('ADES_mvt')==ap)['t'].to_numpy())
    asort = np.sort(a[~np.isnan(a)])
    f = {'MVT_ID_mvt': g['MVT_ID_mvt'].to_numpy()}
    for w in (300, 600, 1200, 1800):
        f[f'dep_all_prev{w}'] = cnt(t, t - w, t)
        f[f'dep_all_next{w}'] = cnt(t, t + 1, t + w + 1)
        f[f'arr_all_prev{w}'] = cnt(at, t - w, t)
    # pushbacks (NM AOBT) of others around my take-off and before it
    for w in (600, 1200, 1800):
        f[f'push_prev{w}'] = cnt(asort, t - w, t)
    # take-offs between my AOBT proxy and my take-off (queue ahead proxy)
    an = np.where(np.isnan(a), t - 900, a)
    f['dep_between_aobt_to'] = np.where(np.isnan(a), np.nan, cnt(t, an, t))
    f['push_between_aobt_to'] = np.where(np.isnan(a), np.nan, cnt(asort, an, t))
    # schedule pressure
    ss = np.sort(s)
    f['sched_dep_pm30'] = cnt(ss, t - 1800, t + 1800)
    # same-runway sequence
    rw = g['RUNWAY_mvt'].fill_null('NA').to_numpy()
    for k in ['rw_prev600','rw_prev1800','rw_gap_prev','rw_gap_next','rw_gap_prev5_med','rw_between_aobt_to','orc_rw_prev5_mean','orc_all_pm600_mean']:
        f[k] = np.full(len(t), np.nan)
    for r in np.unique(rw):
        idx = np.where(rw == r)[0]; tr_ = t[idx]
        f['rw_prev600'][idx] = cnt(tr_, tr_ - 600, tr_)
        f['rw_prev1800'][idx] = cnt(tr_, tr_ - 1800, tr_)
        gp = np.diff(tr_, prepend=np.nan); f['rw_gap_prev'][idx] = gp
        f['rw_gap_next'][idx] = np.append(np.diff(tr_), np.nan)
        # median of last 5 gaps
        gps = pl.Series(gp).rolling_median(5, min_samples=1).to_numpy(); f['rw_gap_prev5_med'][idx] = gps
        ai = an[idx]; f['rw_between_aobt_to'][idx] = np.where(np.isnan(a[idx]), np.nan, cnt(tr_, ai, tr_))
        # ORACLE: mean true taxi of previous 5 take-offs same runway (excluding self)
        yi = y[idx]; cs = np.concatenate([[0], np.cumsum(yi)])
        j = np.arange(len(idx)); lo = np.maximum(j - 5, 0)
        with np.errstate(invalid='ignore'):
            f['orc_rw_prev5_mean'][idx] = np.where(j > 0, (cs[j] - cs[lo]) / np.maximum(j - lo, 1), np.nan)
    # ORACLE: mean true taxi of all deps within +-10 min excluding self
    cs = np.concatenate([[0], np.cumsum(y)]); lo = np.searchsorted(t, t - 600, 'left'); hi = np.searchsorted(t, t + 600, 'right')
    nn = hi - lo - 1
    f['orc_all_pm600_mean'] = np.where(nn > 0, (cs[hi] - cs[lo] - y) / np.maximum(nn, 1), np.nan)
    out.append(pl.DataFrame(f))
seq = pl.concat(out, how='vertical_relaxed')
seq = seq.with_columns([pl.col(c).cast(pl.Float32) for c in seq.columns if c != 'MVT_ID_mvt'])
nm = dep.select('MVT_ID_mvt', pl.col('FLIGHT_ID_mvt').is_null().cast(pl.Float32).alias('no_nm'))
df = base.join(seq, on='MVT_ID_mvt').join(nm, on='MVT_ID_mvt')
SEQ = [c for c in seq.columns if c != 'MVT_ID_mvt' and not c.startswith('orc_')]
ORC = [c for c in seq.columns if c.startswith('orc_')]
q4 = pl.col('event_time').dt.month() >= 10
trn, val = df.filter(~q4), df.filter(q4)
clean = (val['no_nm'] == 0).to_numpy() & (val['target'] <= 7200).to_numpy()

def prep(frame, cols):
    pdf = frame.select(cols).to_pandas()
    for c in CATEGORICAL_FEATURES:
        if c in pdf: pdf[c] = pdf[c].astype('category')
    return pdf

arms = {
  'A0_base': CATEGORICAL_FEATURES + NUMERIC_FEATURES,
  'A0b_base+no_nm': CATEGORICAL_FEATURES + NUMERIC_FEATURES + ['no_nm'],
  'A1_+seq': CATEGORICAL_FEATURES + NUMERIC_FEATURES + ['no_nm'] + SEQ,
  'A2_+seq+ORACLE': CATEGORICAL_FEATURES + NUMERIC_FEATURES + ['no_nm'] + SEQ + ORC,
}
params = dict(objective='regression', learning_rate=0.05, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, cat_smooth=10, max_cat_to_onehot=8, num_threads=12, seed=20260926, verbose=-1)
ytr = trn['target'].to_numpy(); yv = val['target'].to_numpy()
for name, cols in arms.items():
    t0 = time.time(); Xtr = prep(trn, cols); Xv = prep(val, cols)
    for c in CATEGORICAL_FEATURES:
        Xv[c] = Xv[c].cat.set_categories(Xtr[c].cat.categories)
    m = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=1500)
    p = m.predict(Xv); se = (p - yv) ** 2
    lir = (val['airport'] == 'LIRF').to_numpy()
    print(f'{name:18s} RMSE all {se.mean()**.5:7.2f} | clean {se[clean].mean()**.5:7.2f} | clean excl LIRF {se[clean & ~lir].mean()**.5:7.2f} | {time.time()-t0:5.0f}s', flush=True)
    if name == 'A2_+seq+ORACLE' or name == 'A1_+seq':
        imp = sorted(zip(m.feature_importance('gain'), cols), reverse=True)[:12]
        print('   top gain:', [c for _, c in imp])
    np.save(f'{sys.argv[1]}/pred_{name}.npy', p)
val.select('MVT_ID_mvt','airport','target').write_parquet(f'{sys.argv[1]}/val_ids.parquet')
