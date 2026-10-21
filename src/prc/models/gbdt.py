"""LightGBM regressors used under VAL-MONTH-V2, including the two-part schedule-fill model."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from prc.features.flight import CATEGORICAL

DEFAULT_PARAMS = dict(learning_rate=0.05, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=1, cat_smooth=10, max_cat_to_onehot=8,
                      num_threads=12, verbose=-1, deterministic=True, force_row_wise=True)


class Frame:
    """Polars -> pandas with categories frozen to the training vocabulary."""

    def __init__(self, cols: list[str]):
        self.cols = cols
        self.cats: dict[str, pd.Index] = {}

    def fit_transform(self, frame: pl.DataFrame) -> pd.DataFrame:
        pdf = frame.select(self.cols).to_pandas()
        for c in self.cols:
            if c in CATEGORICAL:
                pdf[c] = pdf[c].astype("category")
                self.cats[c] = pdf[c].cat.categories
        return pdf

    def transform(self, frame: pl.DataFrame) -> pd.DataFrame:
        pdf = frame.select(self.cols).to_pandas()
        for c, cats in self.cats.items():
            pdf[c] = pd.Categorical(pdf[c], categories=cats)
        return pdf


OFFSET_CLIP = 10800.0


def proxy_offset(frame: pl.DataFrame) -> np.ndarray:
    """Take-off minus NM AOBT_3, clipped; airport median where missing. Used as a boosting start
    point so the trees learn a correction to an observed quantity that can exceed the 2025 range."""
    px = frame.select(
        pl.col("aobt_proxy_sec").clip(0, OFFSET_CLIP)
        .fill_null(pl.col("aobt_proxy_sec").clip(0, OFFSET_CLIP).median().over("airport"))
        .fill_null(900.0)
    ).to_series()
    return px.to_numpy().astype(float)


DEFAULT_SHARE = 0.3  # airports whose median daily share of default-valued AOBT_3 exceeds this


class ProxyOffset:
    """Boosting start point. `aobt_proxy`: take-off minus AOBT_3 everywhere (EXP-021).
    `aobt_proxy_dev` (EXP-030): same, except at airports where AOBT_3 is mostly a default value;
    there, the airport's median training target plus the deviation from the day's default."""

    def __init__(self, mode: str | None):
        self.mode = mode

    def fit(self, train: pl.DataFrame) -> "ProxyOffset":
        if self.mode == "aobt_proxy_dev":
            share = train.group_by("airport").agg(pl.col("px_day_default_share").median())
            self.default_airports = set(share.filter(pl.col("px_day_default_share") > DEFAULT_SHARE)["airport"].to_list())
            med = train.filter(pl.col("target") < 3600).group_by("airport").agg(pl.col("target").median())
            self.median = dict(zip(med["airport"].to_list(), med["target"].to_list()))
        return self

    def __call__(self, frame: pl.DataFrame) -> np.ndarray | None:
        if self.mode is None:
            return None
        px = proxy_offset(frame)
        if self.mode == "aobt_proxy":
            return px
        dev = frame["px_dev"].fill_null(0).clip(-OFFSET_CLIP, OFFSET_CLIP).to_numpy().astype(float)
        ap = frame["airport"].to_numpy()
        base = np.array([self.median.get(a, 900.0) for a in ap])
        is_def = np.isin(ap, list(self.default_airports))
        return np.where(is_def, np.clip(base + dev, 0, OFFSET_CLIP), px)


class LGBMRegressor:
    def __init__(self, cols: list[str], rounds: int = 1500, seed: int = 20260926, offset: str | None = None, **params):
        self.frame = Frame(cols)
        self.rounds = rounds
        self.offset = ProxyOffset(offset)
        self.params = {**DEFAULT_PARAMS, "seed": seed, "objective": "regression", **params}

    def _init(self, frame: pl.DataFrame) -> np.ndarray | None:
        return self.offset(frame)

    def fit(self, train: pl.DataFrame) -> "LGBMRegressor":
        self.offset.fit(train)
        X = self.frame.fit_transform(train)
        self.model = lgb.train(self.params, lgb.Dataset(X, train["target"].to_numpy(), init_score=self._init(train)),
                               self.rounds)
        return self

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        base = self._init(frame)
        p = self.model.predict(self.frame.transform(frame))
        return p if base is None else p + base

    def importance(self, k: int = 20) -> list[str]:
        g = self.model.feature_importance("gain")
        return [c for _, c in sorted(zip(g, self.frame.cols), reverse=True)[:k]]


class CatBoostReg:
    """CatBoost regression with native categoricals; same interface and offset option as LGBMRegressor."""

    def __init__(self, cols: list[str], rounds: int = 3000, seed: int = 20260926, offset: str | None = None,
                 depth: int = 8, learning_rate: float = 0.08, num_threads: int = 8, loss: str = "RMSE",
                 rsm: float | None = None, **_):
        self.cols, self.rounds, self.seed, self.offset = cols, rounds, seed, ProxyOffset(offset)
        self.depth, self.lr, self.threads = depth, learning_rate, num_threads
        self.loss, self.rsm = loss, rsm  # EXP-042: e.g. "Huber:delta=900", rsm 0.5
        self.cat = [c for c in cols if c in CATEGORICAL]

    def _X(self, frame: pl.DataFrame) -> pd.DataFrame:
        pdf = frame.select(self.cols).to_pandas()
        for c in self.cat:
            pdf[c] = pdf[c].astype(str)
        return pdf

    def _init(self, frame: pl.DataFrame) -> np.ndarray | None:
        return self.offset(frame)

    def fit(self, train: pl.DataFrame) -> "CatBoostReg":
        from catboost import CatBoostRegressor, Pool

        self.offset.fit(train)
        extra = {} if self.rsm is None else {"rsm": self.rsm}
        self.model = CatBoostRegressor(iterations=self.rounds, depth=self.depth, learning_rate=self.lr,
                                       loss_function=self.loss, random_seed=self.seed, thread_count=self.threads,
                                       verbose=0, allow_writing_files=False, **extra)
        self.model.fit(Pool(self._X(train), train["target"].to_numpy(), cat_features=self.cat,
                            baseline=self._init(train)))
        return self

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        p = self.model.predict(self._X(frame))
        base = self._init(frame)
        return p if base is None else p + base

    def importance(self, k: int = 20) -> list[str]:
        g = self.model.get_feature_importance()
        return [c for _, c in sorted(zip(g, self.cols), reverse=True)[:k]]


NM_WINDOW = 3600.0  # airport off-block is always within 60 min of NM LOBT for matched flights (EXP-033)


class NMWindow:
    """Hard constraint from how NM records were joined to movements (EXP-033): for a matched
    departure, |off-block - LOBT| <= 60 min (0 exceptions in 2,062,623 matched 2025 departures).
    Hence target in [take-off - LOBT - 3600, take-off - LOBT + 3600], and a schedule-filled off-block
    is impossible when the schedule is more than 60 min from LOBT."""

    def fit(self, train: pl.DataFrame) -> "NMWindow":
        m = train.filter((pl.col("no_nm") == 0) & pl.col("mvt_vs_lobt_sec").is_not_null())
        # off-block minus LOBT = (take-off - LOBT) - target
        med = m.group_by("airport").agg((pl.col("mvt_vs_lobt_sec") - pl.col("target")).median().alias("m"))
        self.offset = dict(zip(med["airport"].to_list(), med["m"].to_list()))
        return self

    def arrays(self, frame: pl.DataFrame):
        lobt = frame["mvt_vs_lobt_sec"].cast(pl.Float64).to_numpy()
        matched = (frame["no_nm"].to_numpy() == 0) & ~np.isnan(lobt)
        return matched, lobt - NM_WINDOW, lobt + NM_WINDOW, lobt

    def fill_possible(self, frame: pl.DataFrame) -> np.ndarray:
        matched, lo, hi, _ = self.arrays(frame)
        ms = frame["mvt_vs_schedule_sec"].cast(pl.Float64).fill_null(0).to_numpy()
        return ~matched | ((ms >= lo) & (ms <= hi))

    def apply(self, frame: pl.DataFrame, pred: np.ndarray) -> np.ndarray:
        """Clip matched flights' predictions to their window (nearest edge)."""
        matched, lo, hi, _ = self.arrays(frame)
        clipped = np.clip(pred, np.maximum(np.nan_to_num(lo, nan=0.0), 0.0), np.nan_to_num(hi, nan=np.inf))
        return np.where(matched, clipped, pred)


D24_BUCKETS = [0, 3, 10, 24]  # local take-off hour buckets (EXP-034)


def _crossing_and_bucket(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """No-NM departures whose local take-off date is after the schedule's local date, and the local
    take-off hour bucket."""
    from prc.features.flight import LOCAL_TZ

    t = pl.col("event_time")
    s = t - pl.duration(seconds=pl.col("mvt_vs_schedule_sec").fill_null(0).cast(pl.Int64))
    t_local_date = pl.lit(None, dtype=pl.Date)
    s_local_date = pl.lit(None, dtype=pl.Date)
    for ap, tz in LOCAL_TZ.items():
        t_local_date = pl.when(pl.col("airport") == ap).then(t.dt.convert_time_zone(tz).dt.date()).otherwise(t_local_date)
        s_local_date = pl.when(pl.col("airport") == ap).then(s.dt.convert_time_zone(tz).dt.date()).otherwise(s_local_date)
    x = frame.select(((pl.col("no_nm") == 1) & (t_local_date > s_local_date)).fill_null(False).alias("c"),
                     pl.col("local_hour").fill_null(0).alias("h"))
    bucket = np.digitize(x["h"].to_numpy(), D24_BUCKETS[1:-1])
    return x["c"].to_numpy(), bucket


class D24:
    """24-hour date artefact (EXP-034): some no-NM departures that left on a later local date than
    scheduled have their off-block stored with the schedule's local date and the real time of day,
    so target = 86,400 s + taxi. Probability by airport group (LIRF vs other) and local take-off
    hour bucket, learned from training rows with a prior pulling toward 0."""

    PRIOR = 2.0  # pseudo-observations of "not D24"

    def fit(self, train: pl.DataFrame) -> "D24":
        c, b = _crossing_and_bucket(train)
        y = train["target"].to_numpy()
        is_d24 = (y > 84000) & (y < 90000)
        lirf = train["airport"].to_numpy() == "LIRF"
        self.q = {}
        for g in (True, False):
            for k in range(len(D24_BUCKETS) - 1):
                m = c & (b == k) & (lirf == g)
                self.q[(g, k)] = float(is_d24[m].sum() / (m.sum() + self.PRIOR))
        return self

    def prob(self, frame: pl.DataFrame) -> np.ndarray:
        c, b = _crossing_and_bucket(frame)
        lirf = frame["airport"].to_numpy() == "LIRF"
        q = np.array([self.q[(bool(g), int(k))] for g, k in zip(lirf, b)])
        return np.where(c, q, 0.0)


class D24Zone:
    """24-hour date artefact as a rule (EXP-039, replaces the hour-bucket table of EXP-034).

    The recorded off-block keeps the real local time of day and takes the calendar date nearest to
    the schedule. A LIRF no-NM departure whose real off-block falls on a later local date than the
    schedule, at a time of day within 12 h of the scheduled time of day, is stored 24 h early
    (target = 86,400 s + taxi). 2025: 11 of 11 such non-fills are D24; night crossings (gap > 12 h)
    23 normal, 1 D24; matched LIRF flights and the other airports 0 D24 in 984 comparable rows.
    In the zone the only uncertainty left is fill (target = take-off - schedule) versus D24."""

    TZ = "Europe/Rome"
    NIGHT_RATE = 1.0 / 25.0  # LIRF no-NM later-date rows outside the zone (ITY332, 1 of 24)
    LATE = 6 * 3600.0  # EXP-041: LIRF no-NM departures this late are fills (37 of 38 in 2025)
    ZONE_BINS = [14 * 3600.0, 18 * 3600.0]  # zone fill rate falls with lateness (D24 takes over)

    def __init__(self, late_rule: bool = False):
        self.late_rule = late_rule

    def fit(self, train: pl.DataFrame) -> "D24Zone":
        """Learn the late-fill rate and the zone fill rates by lateness from the training rows."""
        if not self.late_rule:
            return self
        est = train["nn_pred"].fill_null(900.0).to_numpy() if "nn_pred" in train.columns else np.full(train.height, 900.0)
        z, _ = self.zones(train, est)
        late = self.late_mask(train, z)
        f = train["fill"].to_numpy()
        self.late_rate = float((f[late].sum() + 0.5) / (late.sum() + 1.0))
        zb = np.digitize(train["mvt_vs_schedule_sec"].fill_null(0).to_numpy(), self.ZONE_BINS)
        self.zone_rates = np.array([(f[z & (zb == k)].sum() + 1.0) / ((z & (zb == k)).sum() + 2.0) for k in range(3)])
        return self

    def late_mask(self, frame: pl.DataFrame, z: np.ndarray) -> np.ndarray:
        """LIRF no-NM departures more than 6 h late, outside the D24 zone. A delay that large is always
        filed, so NM's LOBT follows the real off-block: a no-NM record can only carry a substitute
        off-block (the schedule), not the real one."""
        ms = frame["mvt_vs_schedule_sec"].fill_null(0).to_numpy()
        cand = (frame["airport"].to_numpy() == "LIRF") & (frame["no_nm"].to_numpy() == 1)
        return cand & (ms > self.LATE) & ~z

    def adjust(self, frame: pl.DataFrame, p: np.ndarray, z: np.ndarray) -> np.ndarray:
        if not self.late_rule:
            return p
        late = self.late_mask(frame, z)
        zb = np.digitize(frame["mvt_vs_schedule_sec"].fill_null(0).to_numpy(), self.ZONE_BINS)
        p = np.where(late, np.maximum(p, self.late_rate), p)
        return np.where(z, self.zone_rates[zb], p)

    def zones(self, frame: pl.DataFrame, reg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        taxi = np.clip(np.nan_to_num(reg, nan=900.0), 300.0, 1500.0)  # robust taxi for the date test
        f = frame.select(
            pl.col("event_time").alias("t"),
            (pl.col("event_time") - pl.duration(seconds=pl.col("mvt_vs_schedule_sec").fill_null(0).cast(pl.Int64))).alias("s"),
            ((pl.col("airport") == "LIRF") & (pl.col("no_nm") == 1)).alias("cand"),
        ).with_columns(pl.Series("taxi", taxi))
        f = f.with_columns((pl.col("t") - pl.duration(seconds=pl.col("taxi").cast(pl.Int64))).alias("ob"))
        loc = lambda c: pl.col(c).dt.convert_time_zone(self.TZ).dt.replace_time_zone(None)  # noqa: E731
        tod = lambda c: (loc(c).dt.hour().cast(pl.Int32) * 3600 + loc(c).dt.minute().cast(pl.Int32) * 60  # noqa: E731
                         + loc(c).dt.second().cast(pl.Int32))
        f = f.with_columns(later=(loc("ob").dt.date() > loc("s").dt.date()) & pl.col("cand"),
                           gap=((tod("s") - tod("ob")) / 3600.0).abs())
        z = f.select((pl.col("later") & (pl.col("gap") < 12)).fill_null(False)).to_series().to_numpy()
        night = f.select((pl.col("later") & (pl.col("gap") >= 12)).fill_null(False)).to_series().to_numpy()
        return z, night


class TwoPartFill:
    """p(fill) * (take-off - schedule) + (1 - p(fill)) * E[target | not fill].

    `fill` marks recorded off-blocks equal to the schedule (see prc.data.departure_labels); for those
    rows the target is take-off minus schedule, which is known on ranking.
    """

    def __init__(self, cols: list[str], rounds: int = 1500, clf_rounds: int = 800, seed: int = 20260926,
                 offset: str | None = None, reg: str = "lgbm", d24: bool = False, recnm: bool = False,
                 yclip: float | None = None, clf_stake: bool = False, lirf_late: bool = False,
                 reg_params: dict | None = None, **params):
        self.lirf_late = lirf_late  # EXP-041, requires d24="zone"
        self.use_d24 = d24  # False, True (hour-bucket table, EXP-034) or "zone" (rule, EXP-039)
        # EXP-038: average the unweighted fill classifier with one weighted by the stake
        # (take-off - schedule - nn_pred)^2, the weight of each row in the mixture's squared error.
        self.clf_stake = clf_stake
        # EXP-036: clip the non-fill regression target (e.g. 10,800 s). The 30-odd 2025 non-fill rows
        # at 58,000-88,000 s (D24, overnight records) otherwise contaminate the leaves of ordinary
        # no-NM flights; the tail is modelled by the fill, D24 and window components instead.
        self.yclip = yclip
        self.use_recnm = recnm
        if reg == "catboost":
            self.reg = CatBoostReg(cols, seed=seed, offset=offset, num_threads=params.get("num_threads", 8),
                                   **(reg_params or {}))
        else:
            # reg_params (EXP-042): regression-only settings, e.g. Huber loss or feature_fraction
            self.reg = LGBMRegressor(cols, rounds, seed, offset=offset, **{**params, **(reg_params or {})})
        self.frame = Frame(cols)
        self.clf_params = dict(DEFAULT_PARAMS, seed=seed, objective="binary", **params)
        self.clf_rounds = clf_rounds

    def fit(self, train: pl.DataFrame) -> "TwoPartFill":
        self.window = NMWindow().fit(train)
        self.d24 = ((D24Zone(late_rule=self.lirf_late) if self.use_d24 == "zone" else D24()).fit(train)
                    if self.use_d24 else None)
        X = self.frame.fit_transform(train)
        self.clf = lgb.train(self.clf_params, lgb.Dataset(X, train["fill"].to_numpy()), self.clf_rounds)
        self.clf_w = None
        if self.clf_stake:
            nn = (train["nn_pred"] if "nn_pred" in train.columns else pl.Series([900.0] * train.height))
            ms = train["mvt_vs_schedule_sec"].fill_null(0).to_numpy().astype(float)
            w = np.minimum((ms - nn.fill_null(900.0).to_numpy().astype(float)) ** 2, 86400.0 ** 2)
            prm = dict(self.clf_params, num_leaves=63, min_data_in_leaf=200)
            self.clf_w = lgb.train(prm, lgb.Dataset(X, train["fill"].to_numpy(), weight=w / w.mean()), 500)
        reg_rows = train.filter(pl.col("fill") == 0)
        if self.yclip is not None:
            reg_rows = reg_rows.with_columns(pl.col("target").clip(upper_bound=self.yclip))
        self.reg.fit(reg_rows)
        return self

    def predict_parts(self, frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = self.frame.transform(frame)
        p = self.clf.predict(X)
        if self.clf_w is not None:
            p = 0.5 * (p + self.clf_w.predict(X))
        p = np.where(self.window.fill_possible(frame), p, 0.0)
        if self.use_recnm and "rec_to_minus_lobt" in frame.columns:
            # EXP-035, recovered NM record of no-NM departures (LIRF 2025):
            # A |schedule - LOBT| <= 60 min -> never a fill (76 rows, 0 fills);
            # B take-off - 15 min within 45 min of LOBT -> fill in 92% of 24 rows.
            c = frame["rec_to_minus_lobt"].cast(pl.Float64).to_numpy()
            ms = frame["mvt_vs_schedule_sec"].cast(pl.Float64).fill_null(0).to_numpy()
            has = ~np.isnan(c) & (frame["airport"].to_numpy() == "LIRF")  # rules established at LIRF only
            rule_a = has & (np.abs(c - ms) <= NM_WINDOW)   # sched - LOBT = (T - LOBT) - (T - sched)
            rule_b = has & ~rule_a & (np.abs(c - 900.0) <= 2700.0)
            p = np.where(rule_a, 0.0, np.where(rule_b, np.maximum(p, 0.9), p))
        return p, self.reg.predict(frame)

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        p, reg = self.predict_parts(frame)
        ms = frame["mvt_vs_schedule_sec"].fill_null(0).to_numpy()
        mix = p * ms + (1 - p) * reg
        if isinstance(self.d24, D24Zone):
            # in the zone a non-fill is a D24 record; outside it, a small residual rate at night
            z, night = self.d24.zones(frame, reg)
            p = self.d24.adjust(frame, p, z)
            nonfill = np.where(z, 86400.0 + reg, np.where(night, reg + D24Zone.NIGHT_RATE * 86400.0, reg))
            mix = p * ms + (1 - p) * nonfill
        elif self.d24 is not None:
            q = self.d24.prob(frame)
            mix = q * (86400.0 + reg) + (1 - q) * mix
        return self.window.apply(frame, mix)

    def importance(self, k: int = 20) -> list[str]:
        return self.reg.importance(k)


class SeedBag:
    """Average of the same model fitted with several seeds (reduces the per-flight instability of the
    two-part mixture, EXP-018)."""

    def __init__(self, factory, seeds: list[int]):
        self.models = [factory(s) for s in seeds]

    def fit(self, train: pl.DataFrame) -> "SeedBag":
        for m in self.models:
            m.fit(train)
        return self

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        return np.mean([m.predict(frame) for m in self.models], axis=0)

    def importance(self, k: int = 20) -> list[str]:
        return self.models[0].importance(k)


def build_model(cfg: dict, cols: list[str]):
    """Model from an experiment config: model type, rounds, seed(s), offset, reg, params."""
    kind = MODELS[cfg.get("model", "lgbm")]
    extra = {"reg": cfg["reg"]} if "reg" in cfg else {}
    if cfg.get("d24"):
        extra["d24"] = cfg["d24"]  # true (hour table) or "zone"
    if cfg.get("clf_stake"):
        extra["clf_stake"] = True
    if cfg.get("lirf_late"):
        extra["lirf_late"] = True
    if cfg.get("reg_params"):
        extra["reg_params"] = dict(cfg["reg_params"])
    if cfg.get("recnm"):
        extra["recnm"] = True
    if cfg.get("yclip"):
        extra["yclip"] = float(cfg["yclip"])
    make = lambda seed: kind(cols, rounds=cfg.get("rounds", 1500), seed=seed, offset=cfg.get("offset"),  # noqa: E731
                             **extra, **cfg.get("params", {}))
    seeds = cfg.get("seeds")
    return SeedBag(make, seeds) if seeds else make(cfg.get("seed", 20260926))


MODELS = {"lgbm": LGBMRegressor, "two_part": TwoPartFill}


def mixture_parts(model, frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(fill probability, non-fill value) of a two-part model or a seed bag of them, after the D24 zone
    logic (EXP-056b: exact fill weights for the fill-aware daily calibration). Window clipping ignored."""
    if isinstance(model, SeedBag):
        parts = [mixture_parts(m, frame) for m in model.models]
        return np.mean([p for p, _ in parts], axis=0), np.mean([n for _, n in parts], axis=0)
    p, reg = model.predict_parts(frame)
    if isinstance(model.d24, D24Zone):
        z, night = model.d24.zones(frame, reg)
        p = model.d24.adjust(frame, p, z)
        reg = np.where(z, 86400.0 + reg, np.where(night, reg + D24Zone.NIGHT_RATE * 86400.0, reg))
    return p, reg
