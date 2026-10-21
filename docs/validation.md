# Validation protocol

## VAL-Q4-MASKED-V1: fully masked fourth quarter

Purpose: first evaluation of airport-state forecasting, including TimesFM 3.

- Context: 1 January through 30 September 2025.
- Holdout: 1 October through 31 December 2025.
- The complete holdout taxi-out target is masked at once. No true taxi-out
  value from the holdout may be used as a lag, rolling statistic, or context.
- Time grid: 15 minutes, separately for each airport.
- Target: mean departure taxi-out time in each airport/time bin.
- Past-future covariates: actual departure count, actual arrival count, and
  scheduled departure count. These are retained in the holdout because
  equivalent fields can be constructed from the official ranking file.
- Context target gaps, mostly bins without departures overnight, are linearly
  interpolated using context data only and then forward/backward filled.
- Primary diagnostic metric: bucket-mean RMSE weighted by the number of
  departures in each bin. This is not directly comparable to the official
  per-flight RMSE. A promising state forecast must subsequently be joined to
  individual departures and evaluated with official per-flight RMSE.
- TimesFM context length: at most 15,360 bins, the model's documented maximum.
  At 15-minute resolution this retains the final 160 days before Q4.
- TimesFM forecast horizon: 8,832 bins (92 days). This deliberately reproduces
  a fully missing target block rather than an easier rolling forecast.

The ranking labels for January and July 2026 are private competition targets.
They must not be sought from external sources, inferred through leaderboard
probing, or otherwise accessed. Only the official files supplied by the
organisers are permitted as competition input for the current phase.

*Update 2026-10-05:* open external data allowed by the competition rules is used
(weather, OpenStreetMap, and adsb.lol ADS-B surface tracks, confirmed by the organisers);
features are built the same way on train and ranking and never read a ranking label.

### Limitations

- Q4 is neither January nor July, so this is a feasibility test rather than a
  complete estimate of ranking performance.
- July 2026 is separated from the end of training by six unobserved months.
  A later validation protocol must reproduce that gap if this direction is
  promising.
- The TimesFM 3 checkpoint is under a separate non-commercial/non-production
  licence. Results from it are evaluation-only and are not submission-eligible
  unless the licence is independently confirmed to allow competition use.


## VAL-MONTH-V2: whole-month holdouts (current protocol, 2026-09-26)

Purpose: the per-flight protocol for every modelling decision from 2026-09-26 on. Replaces
VAL-Q4-MASKED-V1 for model selection.

- **What the ranking file hides, and only that.** Features are built once from a movement table in
  which every departure's off-block and taxi time is blanked (`prc.data.mask_departures`), exactly
  the ranking condition. `build_features` refuses unmasked input, and a test checks that features do
  not change when departure off-blocks change. Because nothing in the features is label-derived,
  they are identical whichever fold is held out, and identical in construction on ranking.
- **Folds.** `jan` = January 2025, `jul` = July 2025 (the ranking months' seasonal positions),
  `q4` = Oct-Dec 2025 (legacy comparison). Each fold trains on all other 2025 departures except a
  3-day buffer on each side of the holdout.
- **Label-derived components** (the model itself, stage-1 estimates, target encodings, network
  outputs) are fitted on the fold's training rows only. When they feed another model, training rows
  get out-of-fold values (day-grouped K-fold) and other rows get the full-training-rows model, so
  both sides see estimates of the same quality.
- **Primary metric:** RMSE pooled over the `jan` and `jul` holdouts. Reported three ways:
  - `all`: every departure, the competition metric; very noisy, since two overnight records can hold
    half a fold's squared error;
  - `core`: departures with target <= 3 h, the robust view of the competition metric;
  - `clean`: NM-matched departures with target <= 2 h, the physics of taxiing.
  Plus per airport. Decisions rest on `core` and `clean`; `all` is reported, never optimised alone.
- **Uncertainty:** `paired_day_bootstrap` on clean flights for paired comparisons.
- **No early stopping or tuning on holdout rows.** Carve validation slices from training months.
- Commands: `uv run python scripts/protocol/build_features.py` once, then
  `uv run python scripts/protocol/run.py configs/<id>.yaml`; `scripts/protocol/rescore.py` prints a
  comparison table.

### Limits

- The one-year gap to 2026 cannot be reproduced inside 2025 (drift: take-off minus AOBT_3 median
  +50-60 s at EHAM, LFPG, EDDF). The first leaderboard submission measures it. Arrivals' taxi-in
  times are not blanked in ranking and give a label-free 2025 to 2026 congestion comparison.
- Training on months after the holdout is allowed: the task is post-operations and the final model
  is trained on all of 2025 anyway.

### Noise floor (measured 2026-09-26)

Same features, stage-2 seed 20260926 vs 7 (`v2-twopart-2s` vs `v2-twopart-2s-seed2`, jan+jul):
clean 229.77 vs 230.28, core 279.48 vs 279.74, all 398.74 vs 393.57. Treat clean or core differences
below about 0.7 s, and all-flight differences below about 5 s, as noise. Final models should average
several seeds.
