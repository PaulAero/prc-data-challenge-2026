import numpy as np, polars as pl
def cnt(sorted_times, lo, hi):
    return np.searchsorted(sorted_times, hi, 'left') - np.searchsorted(sorted_times, lo, 'left')

def build_seq(dep, arr):
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
    return seq.with_columns([pl.col(c).cast(pl.Float32) for c in seq.columns if c != 'MVT_ID_mvt'])
