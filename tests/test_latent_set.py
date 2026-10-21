from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from prc.data import mask_departures
from prc.features.flight import build_features
from prc.latent_set.tokens import CENTRAL, CONTEXT, build_events, build_windows
from prc.latent_set.train import oof_groups, split_val

T0 = datetime(2025, 1, 31, 20, tzinfo=timezone.utc)


def _movements(offset_block: int = 600, n: int = 30) -> pl.DataFrame:
    rows = []
    for i in range(n):
        t = T0 + timedelta(minutes=7 * i)
        rows.append(dict(MVT_ID_mvt=i, FLIGHT_ID_mvt=100 + i, PHASE_mvt="DEP", ADEP_mvt="EDDF", ADES_mvt="LFPG",
                         MVT_TIME_UTC_mvt=t, BLOCK_TIME_UTC_mvt=t - timedelta(seconds=offset_block + 13 * i),
                         SCHED_TIME_UTC_mvt=t - timedelta(minutes=20), TAXITIME_SEC_mvt=offset_block + 13 * i,
                         RUNWAY_mvt="18", STAND_mvt="A1" if i % 2 else "B2", AIRCRAFT_TYPE_mvt="A320",
                         AOBT_3_flt=t - timedelta(minutes=12), LOBT_flt=t - timedelta(minutes=20),
                         IOBT_flt=t - timedelta(minutes=20), EOBT_1_flt=t - timedelta(minutes=20),
                         ARVT_1_flt=None, ARVT_3_flt=None))
        rows.append(dict(MVT_ID_mvt=500 + i, FLIGHT_ID_mvt=200 + i, PHASE_mvt="ARR", ADEP_mvt="LFPG", ADES_mvt="EDDF",
                         MVT_TIME_UTC_mvt=t - timedelta(minutes=5), BLOCK_TIME_UTC_mvt=t - timedelta(minutes=1),
                         SCHED_TIME_UTC_mvt=t, TAXITIME_SEC_mvt=240, RUNWAY_mvt="25C", STAND_mvt="A1",
                         AIRCRAFT_TYPE_mvt="A320", AOBT_3_flt=None, LOBT_flt=None, IOBT_flt=None, EOBT_1_flt=None,
                         ARVT_1_flt=t - timedelta(minutes=6), ARVT_3_flt=t - timedelta(minutes=5)))
    f = pl.DataFrame(rows)
    for c in ["MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt", "FLIGHT_TYPE_flt", "FLIGHT_RULE_flt"]:
        f = f.with_columns(pl.lit("x").alias(c))
    return f.with_columns(pl.lit("AFR123").alias("FLIGHT_mvt"))


def _events(offset_block: int = 600):
    masked = mask_departures(_movements(offset_block))
    return build_events(masked, build_features(masked))


def test_events_refuse_unmasked_input():
    raw = _movements()
    with pytest.raises(ValueError):
        build_events(raw, build_features(mask_departures(raw)))


def test_tokens_do_not_depend_on_departure_block_times():
    (a, _), (b, _) = _events(600), _events(2400)
    for f in ("mvt_id", "is_dep", "t_ref", "t_stand", "cats", "num", "rw_pair", "st_pair"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f


def test_arrival_stand_time_is_in_block():
    ev, _ = _events()
    arr = ~ev.is_dep
    assert np.all(ev.t_stand[arr] - ev.t_ref[arr] == 240)
    assert np.all(ev.t_stand[ev.is_dep] == ev.t_ref[ev.is_dep])


def test_window_shapes_and_each_departure_central_once():
    ev, enc = _events()
    assert ev.n == 60 and ev.num.shape == (60, len(enc.num_names))
    assert ev.cats.shape == (60, 7)
    win = build_windows(ev)
    seen = np.zeros(ev.n, dtype=int)
    for w in range(win.n):
        idx = np.arange(win.lo[w], win.hi[w])
        t = ev.t_ref[idx]
        assert t.min() >= win.c0[w] - CONTEXT and t.max() < win.c0[w] + CENTRAL + CONTEXT
        central = (t >= win.c0[w]) & (t < win.c0[w] + CENTRAL) & ev.is_dep[idx]
        seen[idx[central]] += 1
    assert np.all(seen[ev.is_dep] == 1) and np.all(seen[~ev.is_dep] == 0)


def test_window_trim_keeps_central_slice():
    ev, _ = _events()
    win = build_windows(ev, max_tokens=10)
    for w in range(win.n):
        idx = np.arange(win.lo[w], win.hi[w])
        t = ev.t_ref[idx]
        n_central = int(((ev.t_ref >= win.c0[w]) & (ev.t_ref < win.c0[w] + CENTRAL)).sum())
        assert int(((t >= win.c0[w]) & (t < win.c0[w] + CENTRAL)).sum()) == n_central
        assert len(idx) <= max(10, n_central)


def test_oof_groups_have_no_day_overlap_and_cover_all_days():
    days = np.arange(20000, 20300)
    g = oof_groups(days, k=3)
    allg = np.concatenate(list(g.values()))
    assert len(allg) == len(days) and len(np.unique(allg)) == len(days)
    for a in range(3):
        for b in range(a + 1, 3):
            assert len(np.intersect1d(g[a], g[b])) == 0
    # 7-day blocks are never split across groups
    blk = {d // 7: k for k, ds in g.items() for d in ds}
    assert all(blk[d // 7] == k for k, ds in g.items() for d in ds)
    fit, val = split_val(np.setdiff1d(days, g[0]))
    assert len(np.intersect1d(fit, val)) == 0 and len(np.intersect1d(fit, g[0])) == 0


def test_model_forward_shapes():
    torch = pytest.importorskip("torch")
    from prc.latent_set.model import WindowEncoder
    from prc.latent_set.train import Device
    ev, _ = _events()
    target = np.where(ev.is_dep, 600.0, np.nan)
    D = Device(ev, target, np.zeros(ev.n), dev="cpu")
    win = build_windows(ev)
    inp, idx, central, pad = D.batch(win, np.arange(win.n))
    m = WindowEncoder(ev.num.shape[1], [int(ev.cats[:, k].max()) for k in range(7)], d=32, heads=4, layers=1,
                      ff=64, z_dim=8)
    out, z = m(**inp)
    assert out.shape == (*idx.shape, 5) and z.shape == (*idx.shape, 8)
    assert int(central.sum()) == int(ev.is_dep.sum())
    # without context a token's output must not depend on other tokens
    m.context = False
    m.eval()
    o1, _ = m(**inp)
    inp2 = dict(inp, num=inp["num"] + torch.randn_like(inp["num"]) * (~central)[..., None])
    o2, _ = m(**inp2)
    assert torch.allclose(o1[central], o2[central], atol=1e-5)


def test_concat_events_sorted_and_windows_cross_the_file_boundary():
    from prc.latent_set.tokens import concat_events, subset_events
    ev, enc = _events()
    first = ev.t_ref < np.median(ev.t_ref)
    a, b = subset_events(ev, first), subset_events(ev, ~first)
    merged = concat_events(b, a)
    for f in ("mvt_id", "t_ref", "cats", "num", "st_pair"):
        assert np.array_equal(getattr(merged, f), getattr(ev, f)), f
    wa, wm = build_windows(a), build_windows(merged)
    # the last window of the first half sees tokens from the second half once merged
    last = int(np.argmax(wm.c0 == wa.c0.max()))
    assert wm.hi[last] - wm.lo[last] > wa.hi[-1] - wa.lo[-1]


def test_ranking_encoding_maps_unseen_categories_to_zero():
    masked = mask_departures(_movements())
    ev, enc = build_events(masked, build_features(masked))
    other = masked.with_columns(pl.lit("Z999").alias("STAND_mvt"))
    ev2, _ = build_events(other, build_features(other), enc)
    assert (ev2.cats[:, 2] == 0).all()
    assert ev2.num.shape == ev.num.shape


def test_dt_bucket_is_device_independent_and_matches_cuda_training_semantics():
    torch = pytest.importorskip("torch")
    from prc.latent_set.model import DT_EDGES_MIN, dt_bucket
    edges = torch.tensor(DT_EDGES_MIN, dtype=torch.float32)
    dt = torch.arange(-20000, 20000).float()
    inv = torch.tensor(1.0, dtype=torch.float32) / torch.tensor(60.0, dtype=torch.float32)
    expected = torch.bucketize(dt.abs() * inv, edges) + (dt < 0).long() * (len(DT_EDGES_MIN) + 1)
    assert torch.equal(dt_bucket(dt, edges), expected)
    assert int(dt_bucket(torch.tensor([600.0]), edges)) == 6  # 600 s sits just above the 10-min edge
    if torch.cuda.is_available():
        assert torch.equal(dt_bucket(dt.cuda(), edges.cuda()).cpu(), expected)


def test_weather_inputs_join_every_token_at_its_hour():
    from prc.latent_set.tokens import WX_TRANSFORMS
    masked = mask_departures(_movements())
    hours = pl.datetime_range(T0 - timedelta(hours=2), T0 + timedelta(hours=6), "1h", eager=True)
    w = pl.DataFrame({"airport": "EDDF", "wx_hour": hours}).with_columns(
        [pl.lit(0.0).alias(c) for c in WX_TRANSFORMS]).with_columns(
        pl.Series("wx_temp", np.arange(len(hours)) * 10.0))
    ev, enc = build_events(masked, build_features(masked), weather=w)
    base, _ = build_events(masked, build_features(masked))
    assert ev.num.shape[1] == base.num.shape[1] + len(WX_TRANSFORMS) + 1 == len(enc.num_names)
    temp = ev.num[:, base.num.shape[1]]  # wx_temp / 10 = hour index since T0 - 2 h
    hour_idx = (ev.t_ref - int((T0 - timedelta(hours=2)).timestamp())) // 3600
    assert np.allclose(temp, hour_idx) and (ev.num[:, -1] == 0).all()
    with pytest.raises(ValueError):
        build_events(masked, build_features(masked), enc)
