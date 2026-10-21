# Experiment log

One entry per meaningful experiment, newest at the bottom. Never edit a past entry except to add a
dated follow-up line. Machine-readable results also go to `experiments/<exp_id>/metrics.json`.

Template:

```
## EXP-NNN  <short title>  (YYYY-MM-DD, branch/worktree)
- Hypothesis:
- Data / features:            (feature set id, train months, filters)
- Validation:                 (protocol id from docs/validation.md)
- Model + key hyperparameters:
- Seed(s):
- RMSE (overall / per fold / per airport):
- Cost:                       (train wall time, inference time, threads)
- Interpretation:
- Decision:                   continue / modify / abandon
- Leaderboard (if submitted): file, RMSE, and the local RMSE it corresponds to
```

Historical state on 2026-09-24: no experiments yet; data had not been downloaded.


## EXP-001 TimesFM 3 airport-state forecast (2026-09-25, local workspace)

- Hypothesis: a pretrained multivariate forecaster can estimate the latent airport taxi-out state
  from its history plus observed future traffic, even when the complete target quarter is masked.
- Data / features: 15-minute airport series. Target is mean departure taxi-out. Past-future
  covariates are departure count, arrival count, and scheduled departure count. Context target gaps
  are interpolated using pre-holdout values only.
- Validation: VAL-Q4-MASKED-V1. Context 2025 Q1-Q3; every 2025 Q4 target masked simultaneously.
- Model + key hyperparameters: Google TimesFM 3.0.2, checkpoint
  `google/timesfm-3.0-pytorch`, zero-shot, CUDA, 15,360-bin context, 8,832-bin horizon, no symmetric
  averaging.
- Seed(s): none (deterministic inference).
- RMSE: 260.23 s on departure-weighted 15-minute bucket means. Official per-flight evaluation over
  513,683 departures: **454.55 s**. Q1-Q3-only weekly-profile baseline: 460.73 s; airport-mean
  baseline: 461.06 s. Oracle true Q4 15-minute bucket mean: 372.68 s.
- Per-airport per-flight RMSE: EDDF 330.20, EDDM 356.61, EGLL 383.54, EHAM 306.27, LEBL 312.06,
  LEMD 309.11, LFPG 424.44, LIRF 1122.67, LSZH 302.72, LTFM 379.54 seconds.
- Cost: 5.99 s inference for ten airports after model load, RTX 4070 Ti SUPER. Checkpoint download is
  1.32 GB and excluded from inference time.
- Interpretation: the airport-state forecast contains signal, improving the honest weekly profile by
  6.18 s overall and by 20.80 s at EGLL. It is not competitive as a standalone flight predictor.
  The 15-minute oracle remains at 372.68 s, showing that flight-level features are essential. The
  extreme LIRF target tail dominates its RMSE and needs separate error analysis.
- Licence: TimesFM 3 weights use the TimesFM Non-Commercial License v1.0. This run is private
  evaluation/research. Do not use the checkpoint or its outputs in a submission until competition
  and prize use is explicitly cleared.
- Decision: continue only as a candidate airport-state feature for a per-flight model; do not use as
  a standalone submission model. Compare against hand-engineered state features under the same
  flight-level protocol.


## EXP-002 Linear future-predictive latent state (2026-09-25, local workspace)

- Hypothesis: a compact airport state trained to predict taxi conditions 1, 3, and 6 hours ahead
  adds operational context beyond ordinary per-flight and calendar features.
- Data / features: six PLS latent coordinates learned from 52 ranking-available traffic features on
  a 15-minute grid. Downstream baseline uses airport, destination, runway, stand, aircraft, operator,
  wake category, flight type/rule, calendar features, and official NM time differences including
  `MVT_TIME_UTC_mvt - AOBT_3_flt`.
- Validation: VAL-Q4-MASKED-V1 at per-flight level. Encoder and CatBoost fit on Q1-Q3; Q4 labels are
  used only for evaluation. The two CatBoost arms are identical except for the six latent columns.
- Model + key hyperparameters: `PLSRegression`, 6 components, standardised internally; CatBoost 600
  trees, depth 9, learning rate 0.08, seed 20260925, 12 CPU threads.
- Seed(s): 20260925.
- RMSE: baseline **291.13 s**; baseline + latent **289.17 s**; improvement **1.96 s** over 513,683
  Q4 departures. All ten airports improve; largest gains are EDDM 6.21 s, EHAM 3.80 s, LSZH 3.69 s,
  and LTFM 3.58 s.
- Uncertainty: paired day-block bootstrap, 10,000 samples: median improvement 1.77 s, 95% interval
  [-3.66, 8.35] s, probability of improvement 71.8%. The gain is promising but not yet stable.
- Cost: encoder fit plus full-year transform 0.86 s. Baseline CatBoost 264.36 s; latent CatBoost
  273.21 s. CPU only for downstream models.
- Interpretation: CatBoost uses `latent_z_1` heavily (10.38% importance, fourth overall), so the
  representation is not ignored. Inspection shows the first two components partly encode static
  airport identity, especially LIRF/EGLL, as well as rolling traffic pressure. The downstream gain
  therefore mixes genuine dynamics with redundant identity information.
- Decision: **continue, modify**. Preserve this result, then fit within-airport or residualised
  predictive states so the latent coordinates represent deviations from each airport's normal
  operating regime. Do not escalate to a neural encoder until that cleaner linear ablation is run.


## EXP-003 Independent per-airport predictive state (2026-09-25, local workspace)

- Hypothesis: removing static airport identity and fitting a separate predictive state within each
  airport will isolate dynamic congestion and improve the EXP-002 representation.
- Data / features: four PLS coordinates per airport from the same ranking-available 15-minute traffic
  features and +1/+3/+6 hour taxi-state targets. No airport identity enters an encoder.
- Validation: VAL-Q4-MASKED-V1, with the exact saved EXP-002 baseline predictions reused.
- Model + key hyperparameters: ten independent `PLSRegression` encoders, 4 components each; the same
  600-tree CatBoost downstream configuration and seed as EXP-002.
- RMSE: baseline **291.13 s**; per-airport latent **292.05 s**; change **-0.92 s** (worse).
  Six airports improve modestly, led by LFPG 1.93 s, LEBL 1.91 s, and LTFM 1.81 s. LIRF worsens by
  8.75 s and LSZH by 3.43 s.
- Uncertainty: paired day-block bootstrap median change -0.84 s, 95% interval [-3.64, 1.97] s,
  probability of improvement 27.8%.
- Cost: ten encoders fit and transform in 1.69 s; downstream model 269.67 s.
- Interpretation: independently fitted PLS axes are not aligned across airports. A coordinate such as
  `latent_z_0` can encode different dynamics at EDDF and LIRF, while one downstream CatBoost feature
  name assumes a shared meaning. Removing identity also removed useful cross-airport structure.
- Decision: **abandon this representation, keep the evidence**. Return to one shared latent geometry,
  but residualise airport and seasonal baselines before fitting so shared coordinates describe
  comparable deviations from normal operations.


## EXP-004 Shared seasonal-residual predictive state (2026-09-25, local workspace)

- Hypothesis: one shared coordinate system trained on deviations from each airports normal 15-minute
  time-of-week profile will preserve cross-airport alignment while removing static and seasonal scale.
- Data / features: six PLS coordinates learned from 38 Q1-Q3-only airport-seasonal residual traffic
  features and future taxi-state residuals at +1/+3/+6 hours.
- Validation: VAL-Q4-MASKED-V1 with the exact saved EXP-002 baseline predictions.
- RMSE: baseline **291.13 s**; residual latent **289.56 s**; improvement **1.57 s**. This does not beat
  the original shared raw latent result of 289.17 s.
- Per-airport behavior: EHAM improves 9.23 s, LFPG 8.48 s, and LIRF 16.92 s; EDDM worsens 38.34 s and
  LSZH worsens 22.65 s. The representation is highly heterogeneous.
- Uncertainty: paired day-block bootstrap median improvement 1.49 s, 95% interval [-2.31, 5.40] s,
  probability of improvement 77.0%.
- Cost: encoder fit and transform 1.03 s; downstream model 278.77 s.
- Interpretation: residualisation exposes disruption-like deviations that are useful at some airports,
  especially EHAM/LFPG/LIRF, but removes or distorts stable scale information needed at EDDM/LSZH.
  Raw and residual states appear complementary rather than interchangeable.
- Decision: **do not replace EXP-002**. Test one combined raw + residual representation under the same
  fixed downstream model. If that fails, stop iterating on linear PLS geometry.


## EXP-005 Combined raw and residual predictive state (2026-09-25, local workspace)

- Hypothesis: absolute airport state and deviation-from-normal state are complementary views that a
  single downstream model can combine.
- Data / features: the six shared raw PLS coordinates from EXP-002 plus six shared seasonal-residual
  PLS coordinates from EXP-004; all encoders fit on Q1-Q3 only.
- Validation: VAL-Q4-MASKED-V1 with the exact saved EXP-002 baseline predictions.
- RMSE: baseline **291.13 s**; combined latent **288.67 s**; improvement **2.46 s**. This beats the
  previous best EXP-002 result (289.17 s) by 0.50 s.
- Per-airport behavior: LFPG improves 12.25 s, EDDM 11.19 s, EHAM 11.01 s, LSZH 5.77 s, LTFM 5.16 s,
  EDDF 4.29 s, and LEBL 3.14 s. EGLL (-0.87 s), LEMD (-0.08 s), and LIRF (-8.56 s) regress.
- Uncertainty: paired day-block bootstrap median improvement 2.52 s, 95% interval [-2.86, 7.33] s,
  probability of improvement 82.1%.
- Cost: downstream model 279.89 s; both latent encoders together take about two seconds.
- Interpretation: raw `latent_z_1` remains the dominant state coordinate, while residual coordinates
  add smaller complementary contributions. The improvement is broader than either view alone but is
  still unstable across days and at LIRF.
- Decision: **new best latent representation; continue**. Add Q1-Q3-only soft operating-regime
  probabilities over the combined state. Do not start a neural encoder yet.


## EXP-006 Soft operating regimes over combined state (2026-09-25, local workspace)

- Hypothesis: soft membership in nonlinear operating regimes adds information that continuous raw
  and residual PLS coordinates do not expose directly.
- Data / features: the 12 combined latent coordinates from EXP-005 plus an eight-component diagonal
  Gaussian mixture fit on Q1-Q3 only. The downstream model receives the most likely regime, all
  eight membership probabilities, membership entropy, and negative log-likelihood surprise.
- Validation: VAL-Q4-MASKED-V1 with the exact saved EXP-002 baseline predictions.
- Model + key hyperparameters: `GaussianMixture`, 8 components, diagonal covariance, 3 initialisations,
  seed 20260925; the same 600-tree CatBoost downstream configuration as earlier experiments.
- RMSE: baseline **291.13 s**; combined latent + regimes **288.01 s**; improvement **3.12 s**. This
  beats EXP-005 by **0.66 s** and is the best latent-state result so far.
- Per-airport behavior: EHAM improves 12.00 s, LFPG 12.07 s, LIRF 11.44 s, EDDF 5.45 s, and LTFM
  3.78 s. EDDM regresses 24.85 s and LSZH regresses 13.44 s; the other airports are nearly neutral.
- Uncertainty: paired day-block bootstrap median improvement 2.75 s, 95% interval [-6.35, 12.91] s,
  probability of improvement 71.4%. The aggregate win is not stable across days.
- Cost: regime fit plus full-year transform 2.78 s; downstream model 301.59 s. Eight regimes all
  retain substantial support, from 26,014 to 67,345 15-minute airport bins.
- Interpretation: multiple soft regime probabilities receive non-trivial importance, so the model
  uses the nonlinear partition. It notably reverses EXP-005's LIRF regression, but trades that for
  severe EDDM and LSZH regressions. This suggests that one global downstream response to regimes is
  too rigid even though the regime representation contains useful signal.
- Decision: **retain as current aggregate best, modify next**. Test airport-conditioned regime
  effects or per-airport model selection using nested time splits. Do not promote EXP-006 as a
  stable final model, and do not start a neural encoder until this heterogeneity is addressed.


## EXP-007 Airport-conditioned soft regimes (2026-09-25, local workspace)

- Hypothesis: the same latent regime has different implications at different airports, so exposing
  an explicit airport-regime interaction will resolve EXP-006's heterogeneous errors.
- Data / features: exactly EXP-006 plus one categorical feature joining airport identity and the
  maximum-probability GMM regime. No encoder, split, or model hyperparameter changed.
- Validation: VAL-Q4-MASKED-V1 with the exact saved EXP-002 baseline predictions.
- Model + key hyperparameters: the EXP-006 eight-component GMM and 600-tree CatBoost, plus the
  `airport_regime` categorical interaction; seed 20260925.
- RMSE: baseline **291.13 s**; airport-conditioned regimes **283.96 s**; improvement **7.17 s**.
  This beats EXP-006 by **4.05 s** and is the new best latent-state result.
- Per-airport behavior: seven airports improve, led by LIRF 19.44 s, LFPG 13.19 s, EHAM 12.98 s,
  EDDF 5.05 s, and LTFM 4.92 s. EGLL (-1.00 s), LEMD (-0.27 s), and LSZH (-2.40 s) regress only
  modestly. EDDM changes from a 24.85 s regression in EXP-006 to a 3.87 s improvement.
- Uncertainty: paired day-block bootstrap median improvement 7.06 s, 95% interval [-4.66, 17.86] s,
  probability of improvement 87.7%. Day-to-day uncertainty remains material.
- Cost: downstream fit and prediction 322.11 s. The joint category itself receives 0.79% feature
  importance and also changes the model's use of regime probabilities and airport identity.
- Interpretation: regime meaning is airport-dependent. A single compact interaction captures much
  of that dependency and makes the gains substantially broader, without a more complex encoder.
- Decision: **new best; continue cautiously**. Before adding representation complexity, validate
  the choice on earlier rolling time splits and test whether airport-specific fallback decisions
  improve stability without selecting on Q4.


## Re-score of EXP-002 to EXP-007 on the clean population (2026-09-26, analysis only)

- Question: are the latent-state gains real on ordinary flights, or driven by a few extreme rows?
- Method: recompute RMSE of the saved Q4 predictions on NM-matched departures with target <= 7200 s
  (508,759 of 513,683 flights, 99.0%).
- Result (baseline 228.37 s on this population): EXP-002 226.09, EXP-005 **224.25**, EXP-006 228.13,
  EXP-007 224.42. On clean LIRF, every latent arm is worse than the baseline (352.1 vs 351.4-356.2).
- Context: in EXP-007, 51 flights (0.01%) carry 33% of the Q4 squared error; LIRF is 7% of flights
  and 46% of squared error; excluding the 104 flights with target > 2 h drops RMSE from 283.96 to
  233.48 s.
- Interpretation: the regime steps (EXP-006, EXP-007) add nothing on the bulk of flights. EXP-007's
  aggregate lead over EXP-005 and its "LIRF +19 s" are movements on a handful of artefact rows. The
  combined PLS state (EXP-005) is worth about 4 s on clean flights. Day-block intervals were wide
  because tail rows dominate each day's error, not because the state signal is fragile.
- Decision: park the latent/regime line. Keep EXP-005 coordinates as a candidate feature block to
  re-test once the flight-level features below are in.


## EXP-008 Runway-sequence and queue features, with oracle ceiling (2026-09-26, scouting)

- Hypothesis: in post-operations data every departure's take-off time and NM AOBT_3 is known, so
  the take-off queue a flight experienced (Idris et al. 2002: number of take-offs between pushback
  and take-off) and runway sequencing can be measured per flight, not per 15-minute bin.
- Features (all ranking-available): take-offs and landings in the previous/next 5-30 min (all and
  same runway), gaps to previous/next take-off on the same runway, rolling median gap, NM pushbacks
  in the previous 10-30 min, take-offs and pushbacks between own AOBT_3 and own take-off (all and
  same runway), scheduled departures within +-30 min, `no_nm` flag. Oracle arm adds true neighbour
  taxi-outs (mean of previous 5 same-runway take-offs, mean of all take-offs within +-10 min).
- Validation: VAL-Q4-MASKED-V1. Scouting quality: LightGBM, fixed 1500 rounds, lr 0.05, 255 leaves,
  seed 20260926, no tuning, one seed. Metrics for all flights, clean flights (NM-matched, target
  <= 2 h), clean excluding LIRF.
- RMSE all / clean / clean excl LIRF: base 304.45 / 227.78 / 215.63; **+seq 302.36 / 221.41 /
  209.06**; +seq+ORACLE 288.00 / 205.99 / 192.43.
- Cost: about 70 s per arm on 12 threads, feature build under a minute.
- Interpretation: the flight-level queue block gives -6.4 s on clean flights, more than the whole
  latent line, for trivial cost. The oracle ceiling of a further -15.4 s says that estimating the
  neighbours' taxi-outs (two-stage, out-of-fold) is worth maturing. LightGBM base on all flights
  (304) is worse than CatBoost (291) because it fits the artefact tail worse; on clean flights they
  match (227.8 vs 228.4).
- Decision: **continue**. Promote the queue block into `src/prc/features/`; build the two-stage
  neighbour estimate.
- Script: `scripts/scouting/exp008_queue_features.py`.


## EXP-009 Stand occupancy, runway busy periods, two-part model for schedule-filled off-blocks (2026-09-26, scouting)

- Hypotheses:
  1. Arrivals keep in-block time and stand in the ranking file. If the next aircraft blocks in at the
     same stand before my take-off, I had pushed back before that, a hard lower bound on taxi-out.
     Checked on 2025: the bound holds for 100% of the 4-26% of departures where it applies, and it
     applies to 32% of taxi-outs over 40 min (60% at EGLL).
  2. Time since the runway last had an idle gap (busy period) measures a saturated queue.
  3. Some off-block times are the schedule, not an observation: |BLOCK - SCHED| <= 10 s for 4.8% of
     departures, 18% at LIRF (natural rate about 1%). For those rows the target is exactly
     take-off minus schedule, which is available in the ranking file. They are 4.8% of rows and 32%
     of within-airport squared deviation. A classifier P(fill) plus a regressor on non-fill rows,
     combined as `p * (take-off - schedule) + (1 - p) * E[y | not fill]`, is the RMSE-optimal mixture.
- Validation / model: as EXP-008 (same features B1 = EXP-008 +seq arm).
- RMSE all / clean / clean excl LIRF / LIRF all: B1 302.64 / 221.37 / 209.02 / 809.4; +stand 299.97 /
  217.32 / 204.67 / 810.5; +busy 303.93 / 221.33 / 208.86 / 817.0; +stand+busy 299.44 / 216.76 /
  203.95 / 810.8; **two-part fill on +stand+busy 255.91 / 215.16 / 202.96 / 570.8**. Fill classifier
  AUC 0.854 on Q4.
- Cost: about 75 s per regression arm, 125 s for the two-part model, 12 threads.
- Interpretation: stand occupancy is worth -4.1 s on clean flights; busy-period features add nothing
  once the sequence block is present. The two-part model is the largest single effect found so far:
  -43.5 s on all flights and 28 s better than EXP-007, mostly at LIRF (810 -> 571). It also helps
  clean flights slightly.
- Caveats: single split, single seed, fixed rounds, Q4 only. The fill regime is a property of how
  the data were recorded; the rule is learned from 2025 and applied identically to 2026, which the
  rules allow, but the write-up must say plainly that part of the gain is modelling a recording
  artefact.
- Decision: **continue, top priority**. Confirm on the January/July folds of the new protocol, then
  refine the fill classifier (per-airport, no-NM interaction, second-level jitter pattern).
- Script: `scripts/scouting/exp009_stand_busy_fill.py`.


## EXP-011 Baseline ladder under VAL-MONTH-V2 (2026-09-26)

- Purpose: re-establish the reference numbers under the new protocol (`docs/validation.md`).
- Configs: `v2-base` (LightGBM, base block), `v2-seqstand` (+ sequence and stand blocks),
  `v2-twopart` (two-part fill model on base+seq+stand). LightGBM lr 0.05, 255 leaves, 1500 rounds
  (800 for the fill classifier), seed 20260926, 12 threads.
- RMSE, pooled jan+jul (all / core <= 3 h / clean): base 482.31 / 309.49 / 245.22; seqstand
  476.60 / 297.08 / 235.49; **twopart 401.27 / 287.13 / 234.32**. Q4 fold: 305.76 / 244.14 / 225.85,
  300.54 / 235.39 / 216.46, 259.21 / 230.38 / 214.57.
- Cost: 50 s (base), 80 s (seqstand), 130 s (twopart) per fold.
- Interpretation:
  - The sequence and stand blocks give about -10 s on clean flights on every fold, confirming the
    Q4 scouting (EXP-008/009).
  - The two-part model is a large win on all flights in July and Q4. On January it is worse on
    core (234.4 vs 226.7). The cost comes from non-fill LIRF flights more than 2 h late: January
    2025 has an unusually low fill rate among those (7.5% vs 10-20% in other months, n = 159).
    Month-to-month variance of a rare event, not a bug, but the ranking file has 282 (Jan) and 659
    (Jul) such LIRF flights, so the calibration of P(fill) for very late flights is worth work.
  - The remaining all-flight error is dominated by a few NM-unmatched overnight records (e.g. an
    LFPG off-block the evening before take-off, 1,404 min). Two such flights hold half of LFPG's
    January squared error. Hence the `core` metric.
- Decision: `v2-twopart` is the reference for all latent/two-stage work.


## EXP-012 Two-stage neighbour state (estimated off-blocks), with ablation (2026-09-26)

- Hypothesis: the blanked off-blocks of all neighbours can be estimated; describing each flight's
  surroundings with those estimates recovers part of the EXP-008 oracle gain (-15 s clean).
- Derivation beforehand: if neighbour information is new, "neighbour features without the
  self-estimate" improves clean RMSE; if the gain is only stacking, "self-estimate only" carries it.
- Method (`src/prc/features/two_stage.py`, `scripts/protocol/build_two_stage.py`):
  - Stage 1 is a LightGBM regression on non-fill rows, 600 rounds. Training rows get 4-fold
    day-grouped out-of-fold estimates; all other rows get a model fitted on all training rows.
  - Estimated off-block = take-off - estimate.
  - Features: aircraft estimated to be taxiing out at my take-off and at my estimated off-block
    (all, and same runway); take-offs between my estimated off-block and my take-off; aircraft
    that pushed back after me but took off before me (overtaking); neighbours' mean estimated
    taxi-out (+-10 min, and previous 5 on the same runway); my own estimate.
- RMSE pooled jan+jul (all / core / clean): reference twopart 401.27 / 287.13 / 234.32;
  **+two-stage 398.74 / 279.48 / 229.77**; self-estimate only 405.61 / 287.67 / 234.53;
  neighbours without self 397.89 / 279.23 / 231.25. Q4: 259.65 / 224.78 / 209.77 (ref 214.57 clean).
- Uncertainty: paired day-block bootstrap on clean flights, jan+jul: +4.50 s [2.61, 6.96], 100% of
  resamples positive; Q4 +4.78 s [3.41, 6.35].
- Where: gain by same-runway traffic quintile (take-offs in the previous 30 min), low to high: 5.9,
  6.5, 7.4, 3.5, -1.2 s. It is not a peak-congestion effect; estimated queues at peak are poorest.
- Cost: stage 1 about 165 s per fold (5 LightGBM fits), stage 2 about 150 s per fold, 12 threads.
- Interpretation: the gain is genuine neighbour information (stacking alone gives nothing),
  capturing about a third of the oracle ceiling. Top features are the overtaking count,
  neighbours' estimated taxi and the self-estimate.
- Decision: **adopt**. Next: iterate the estimate (round 2), and learn the neighbourhood directly
  (EXP-010, neural set encoder).


## EXP-013 Two-stage round 2, and arrivals' taxi-in block (2026-09-26)

- Hypotheses:
  1. Iterating the estimate (round 2: re-estimate with round-1 neighbour features, rebuild the
     neighbourhood) sharpens the hidden-state reconstruction, a fixed point.
  2. Arrivals' taxi-in times are not blanked in ranking and directly observe surface congestion and
     disruption. New block `arr`: mean taxi-in and excess over the stand x runway median for
     arrivals in the previous 15/30/60 min, and arrivals taxiing in at my take-off. More than 99%
     coverage; excess taxi-in over 60 min correlates 0.23 with the target.
- RMSE pooled jan+jul (all / core / clean), reference EXP-012 `v2-twopart-2s` 398.74 / 279.48 / 229.77:
  - round 2 (`v2-twopart-2s-r2`): 385.81 / 285.50 / 231.24. January clean worsens 209.71 -> 215.72;
    July 244.92 -> 243.17 and Q4 209.77 -> 207.99 improve.
  - arr block without two-stage (`v2-twopart-arr`): 402.15 / 286.39 / 232.91 (-1.4 s clean vs
    `v2-twopart`).
  - arr block with two-stage (`v2-twopart-2s-arr`, stage 1 also sees arr): 391.77 / 280.52 / 230.09.
- Interpretation:
  - Round 2 fails on a few January disruption days (3, 6 and 10 January) at EHAM, LFPG, LTFM
    and EDDF (+8 to +18 s clean RMSE there). When stage 1 underestimates a whole abnormal day, a
    second round compounds the error instead of correcting it. Not robust: stop at round 1.
  - Arrivals' taxi-in carries about 1.4 s alone but is redundant with the two-stage block.
  - January's residual error concentrates on winter disruption days. That is the evidence gate for
    weather (next_steps item 6).
- Decision: keep `v2-twopart-2s` (round 1, no arr) as the reference. Keep the arr block available;
  re-test it with weather features.


## EXP-010 Learned flight-centred latent: joint window transformer, with no-context ablation (2026-09-26)

- Hypothesis: a network that reads every observed event around a departure (other take-offs,
  landings, in-blocks, stands, NM times) and jointly estimates the hidden off-blocks of all
  departures in the window learns neighbour information the hand features lack.
- Derivation beforehand:
  - If true: v2-twopart-nn improves clean RMSE with a day-block interval excluding 0; the network
    alone is competitive with the GBDT; the gain does not appear when attention is restricted to
    the token itself (no-context arm); it should be larger in congested quintiles.
  - If false: clean RMSE flat, or gain only on all flights (the fill head reshaping the tail); the
    no-context arm gives the same gain (plain stacking of a second model family).
  - The no-context arm is the discriminating test; the congestion quintiles are secondary.
- Design (`src/prc/latent_set/`, `scripts/latent_set/`):
  - Windows: one airport, 2 h, central 30-min slice plus 45 min each side, step 30 min, so every
    departure is central in exactly one window (145k windows, at most 216 tokens).
  - Tokens: every departure (take-off time, runway, stand, type, operator, NM/schedule offsets,
    the base/seq/stand hand features) and every arrival (landing, taxi-in, in-block, stand, runway,
    schedule and ARVT offsets). Built from `mask_departures` output only; tested.
  - Encoder: 4-layer pre-LN transformer, d 128, 8 heads, with learned additive attention biases
    per (type_i, type_j) pair for the signed time gap, same runway, and same stand with the signed
    stand-time gap (take-off vs in-block). About 0.9 M parameters.
  - Heads on departure tokens: 16-d bottleneck z, then mean, q10/q50/q90 and P(fill).
  - Loss: MSE (mean) + pinball (quantiles) on non-fill rows with the target clipped at 3600 s,
    plus 0.2 x BCE on fill over all labelled rows. Central departures weight 1, others 0.25.
  - Training: AdamW lr 1e-3, warmup + cosine over 16 epochs, batch 64 windows, bf16. Early
    stopping (patience 2) on 10% of the fitting days, never on the holdout.
- Stacking protocol:
  - Per fold, training days are split into 3 groups of contiguous 7-day blocks. Each group is
    predicted out-of-fold by a model fitted on the other two; the holdout month gets the mean of
    the 3 models.
  - z of models 2 and 3 is mapped onto model 1 by an affine least-squares fit on holdout inputs
    (R^2 0.87-0.90), then reduced to 8 PCA dims fitted on training rows.
  - Output: `data/processed/latent_set/{fold}.parquet`.
- RMSE, pooled jan+jul (all / core / clean); reference v2-twopart 401.27 / 287.13 / 234.32:
  - network alone, regression head: 564.87 / - / 234.36;
  - network alone with its own fill head: 450.33 / - / 229.77;
  - **v2-twopart-nn: 403.34 / 271.90 / 221.08**;
  - v2-twopart-nnpred (nn_pred + nn_pfill only): 399.38 / 273.12 / 221.82;
  - v2-twopart-nn-noctx (same network, attention to self only, lr 3e-4 because lr 1e-3
    diverged): 414.80 / 284.93 / 233.84;
  - v2-twopart-2s-nn (EXP-012 two-stage features + latent): 390.98 / 270.89 / 222.25.
- Per fold, clean (reference, then v2-twopart-nn):
  - jan: 210.68 -> 199.88; network alone 206.93.
  - jul: 251.96 -> 236.97; network alone 254.55.
  - q4: 214.57 -> 200.95; network alone 207.29.
  - Every airport improves on clean, pooled jan+jul: -9.5 s (EHAM) to -26.5 s (LTFM).
  - The network alone beats the GBDT on clean at 8 of 10 airports; it is worse at LIRF and EGLL.
- Uncertainty: paired day-block bootstrap, clean, ref minus new:
  - v2-twopart-nn: jan+jul +13.13 s [10.83, 16.17]; jan +10.80 [8.44, 13.16]; jul +14.80
    [11.14, 19.37]; q4 +13.61 [11.34, 16.08]; 100% of resamples positive.
  - no-context arm: +0.49 [-0.26, 1.23].
  - nn vs no-context: +12.71 [10.56, 15.51].
  - Adding two-stage features to nn: -1.11 [-2.38, -0.04] on clean (slightly worse).
- Congestion quintiles, clean jan+jul, delta RMSE (low to high):
  - `dep_between_aobt_to`: -9.6, -13.7, -12.3, -12.5, -17.5;
  - `rw_prev1800`: -12.5, -16.0, -16.3, -11.3, -10.2;
  - `rw_between_aobt_to`: -10.8, -13.3, -15.9, -11.9, -14.1.
  - The gain is broad, only mildly larger for the longest AOBT_3-to-take-off queues.
- Mechanism check (scratch, not kept): three airport-level arrival taxi-in means (the most obvious
  field the tokens have and the hand features lack) give only -1.4 s clean. That is not the
  explanation.
- All-flight caveat: jul all worsens (416.69 -> 430.94, LIRF 1218 -> 1312). Core (<= 3 h) improves
  on both folds, so this is the handful of multi-hour records, a tail-handling effect, not a
  latent effect.
- Cost:
  - Event/window build 12 s.
  - Context network: about 35 s per epoch, 6-9 epochs per model, 12-14 min per fold (3 models), so
    38 min for three folds. GPU peak about 5.6 GB, 8 CPU threads.
  - No-context arm: 24-25 min per fold.
  - Stacked two-part GBDT: about 260 s per fold on 8 threads.
- Interpretation:
  - The learned neighbourhood carries information the hand features and the EXP-012 two-stage
    estimates do not: -13 s clean on every fold, close to the EXP-008 oracle ceiling of -15 s.
  - The no-context arm rules out plain stacking. Same features, same heads, same OOF scheme, and
    the gain disappears.
  - The latent subsumes EXP-012: two-stage features add nothing on top.
  - The gain is not a peak-queue effect. The network reads airport state that matters at all
    congestion levels. What that state is remains open.
  - The z coordinates add little beyond nn_pred (0.7 s).
  - The context model early-stopped while the LR was still high (epochs 6-9), so it is probably
    under-trained.
- Decision: **adopt; mature**.
  - Build ranking windows with the saved encoding; final holdout-style outputs are the mean of the
    3 group models.
  - Try lower LR / longer schedule / 2-3 seeds; compare 2 h vs 3 h windows.
  - Add an explicit P(fill) calibration for LIRF, where the network alone is weak.
  - Look inside: attention-bias ablations (stand, runway) and which neighbour tokens get weight.
- 2026-09-26 follow-up (lead audit of EXP-010): no label leakage found. A token's label enters the loss
  only if its own day is a fitting day (`_weights`, `day_ok`), so boundary windows contribute inputs
  but never held-out labels. Holdout months are predicted by models fitted on fold training days
  only (3-day buffer kept); early stopping uses a slice of training days. Inputs come from
  `mask_departures` and the masked feature cache. One asymmetry to keep in mind: holdout rows get the
  mean of 3 group models, training rows a single out-of-fold model, so the stacked GBDT sees
  slightly better latent features at evaluation; the ranking pipeline must reproduce the same
  scheme. Decision: **adopt `v2-twopart-nn` as the reference** (jan+jul 403.3 / 271.9 / 221.1),
  replacing `v2-twopart-2s`.


## EXP-020 Fill-probability calibration, fill oracle, schedule-relative NM features (2026-09-26, lead)

(EXP-014 to EXP-019 are reserved for the EXP-010 hardening runs.)

- Questions:
  1. Is P(fill) of the two-part model calibrated, especially for LIRF flights more than 2 h late,
     where January looked costly (EXP-011)?
  2. How much would a perfect fill classifier be worth?
  3. Does an explicit NM-vs-schedule feature help the classifier?
- Method: `v2-twopart-nn-calib`, identical to `v2-twopart-nn`, saving `p_fill` and the regression
  part (reproduces 402.90 / 271.95 / 221.14). Oracle: `fill ? take-off - schedule : regression`.
- Calibration:
  - LIRF > 2 h late, mean predicted vs actual fill rate: jan 0.081 vs 0.075, jul 0.120 vs 0.110,
    q4 0.108 vs 0.121. Calibrated; the January cost in EXP-011 was variance of a rare event.
  - No-NM flights: calibrated too.
  - Globally the mid range is overconfident (p in 0.4-0.6: 0.48 predicted vs 0.34 actual).
- Oracle:
  - Core RMSE 271.95 -> 250.86 on jan+jul and 216.47 -> 198.31 on Q4: a 21 s ceiling.
  - On all flights the oracle is worse (417.67 vs 402.90): the mixture's upward pull accidentally
    helps the unpredictable overnight records.
  - A hard 0.5 threshold is worse than the mixture everywhere (core 281.06).
- Mechanism found at LIRF:
  - Fill probability is about 40% when the NM AOBT_3 is within +-60 s of the schedule, against
    5-8% when it is more than 15 min off.
  - Weak temporal clustering: 15% fill with no filled neighbours, 26% with many.
  - Reading: the schedule is recorded as off-block mostly for flights that pushed back roughly on
    time, not only for system gaps.
- New features (`schedrel` block): AOBT_3, EOBT_1 and LOBT minus schedule, and |AOBT_3 - schedule|.
  - `v2-twopart-nn-schedrel`: 402.79 / 271.44 / **219.59** (vs 402.90 / 271.95 / 221.14); Q4
    240.38 / 215.64 / 200.37.
  - Fill AUC 0.8717 -> 0.8723 (LIRF 0.850 -> 0.852).
- Interpretation: the classifier is near the limit of the available fields (AUC 0.87), so most of
  the 21 s oracle ceiling is probably out of reach with row-level information. The -1.5 s clean gain
  comes mainly from the regression part.
- Decision: adopt `schedrel` (cheap, above the noise floor on clean). The reference becomes
  `v2-twopart-nn-schedrel`.


## Observation: extreme disruption in the ranking file, January 2026 (2026-09-26, lead)

- Daily mean arrival taxi-in excess over the stand x runway median, observed in ranking without
  labels, reaches **1,457 s at EHAM on 2026-01-05** (499 s on 01-04, 342 s on 01-03). Departures fall
  to 208/day (normal about 600). LFPG (387 s, 01-07) and EGLL (245 s, 01-04) are also affected.
- The 2025 maximum is 338 s (EGLL 2025-07-06). The closest winter analog is LTFM 2025-02-23 (264 s),
  when the departure taxi-out median jumped from 963 to 2,934 s.
- Daily arrival excess correlates 0.32 with daily departure excess (2025, airport-days).
- Candidate v1 predictions for EHAM follow the NM AOBT_3 proxy on those days (medians 1,584-2,085 s
  vs take-off minus AOBT_3 medians 1,998-2,352 s), so the model is not blind to the event.
- Risk: systematic under-prediction on 2-7 January at EHAM (about 1,900 departures), because trees
  cannot extrapolate beyond 2025. Keep in mind when reading the leaderboard score. Candidate
  mitigations: an extrapolating residual model on the arrival-excess signal, validated on the LTFM
  February 2025 event held out; weather data (needs approval to download).


## EXP-014 Latent outputs for the ranking file (2026-09-26)

- Purpose: `data/processed/latent_set/final.parquet` for the submission, with the fold scheme.
- Method (`scripts/latent_set/run_final.py`, shared code `src/prc/latent_set/oof.py`):
  - 3 group models on all 2025 departure days (7-day blocks, 10% early-stopping slice), EXP-010
    settings: lr 1e-3, 16 epochs, patience 2.
  - 2025 rows get out-of-fold outputs; ranking rows get the mean of the 3 models.
  - z is aligned on ranking inputs (affine; R^2 in `experiments/v2-nn-alone/final_meta.json`), and
    the PCA is fitted on 2025 rows. Alignment R^2 is 0.86 and 0.87 (folds: 0.87-0.90).
  - Ranking tokens come from `mask_departures(load_ranking())` plus `features/ranking.parquet`,
    using the 2025 encoding (unseen categories -> 0).
  - 2025 and 2026 events are merged before windows are cut, so windows on 1 January 2026 read
    31 December 2025 as context. A test covers the file boundary.
- Checks (`scripts/latent_set/check_final.py`, `data/processed/latent_set/final_check.json`):
  - 2,429,888 rows, no duplicate ids, no NaN. Unknown-category share on ranking tokens: stand 1.0%
    (2025: 0.06%), operator 1.3% (0.9%), aircraft type 0.09%, runway 0.
  - All 344,841 ranking departures and all 2025 departures are covered.
  - Mean nn_pred on ranking vs 2025 OOF: +56 to +97 s at EDDF, EDDM, EHAM, LSZH and LTFM; within
    +-15 s at EGLL, LEBL, LEMD, LFPG and LIRF. This is consistent with the year-on-year AOBT_3
    drift, and not an encoding problem.
  - nn_pfill matches 2025 within 0.005 per airport.
- EHAM disruption, 2-7 January 2026:
  - Daily mean nn_pred is 1,093-2,115 s, against a 2025 worst-day mean target of 1,114 s. On
    2025 days, nn_mean tracks the daily mean target with correlation 0.99.
  - At the day level the network extrapolates: its daily means move with the AOBT_3 proxy.
  - At the flight level it saturates at the 3600 s training clip (max 3,650 s). 264 flights have
    proxy > 3600 s; only 143 have nn_pred > 3300 s.
  - The long tail of those days must come from the GBDT stage or a cap-aware rule, not from the
    network.
- Cost: 17 min GPU (3 models, 7-9 epochs each), 24 s event build.


## EXP-015 Latent training length and seeds (2026-09-26, derivation written before the runs)

- Question: the EXP-010 models early-stopped at epochs 6-9 with lr 1e-3, still in the high-LR
  phase. Is the latent under-trained?
- Arm `lr3e4`: lr 3e-4, cosine over 24 epochs, patience 4. Everything else as EXP-010.
- Derivation:
  - If under-training limits the latent: the validation RMSE of each group model drops by 2 s or
    more, and the stacked clean and core RMSE improve by more than the 0.7 s noise floor, on both
    jan and jul.
  - If not: the validation RMSE plateaus at the same level (about 190-200 s), and the stacked
    change stays within +-0.7 s. The cost then roughly doubles for nothing.
- Seeds: 2 extra training seeds of the retained setting, with the same day groups. The spread of
  the stacked clean RMSE is the noise floor of the latent itself. Expected spread 0.5-1.5 s. A
  larger spread means single-seed comparisons of latent variants cannot be trusted, and a seed
  average becomes a candidate.
- Result, `lr3e4` (jan+jul, stacked on base + seq + stand like EXP-010; all / core / clean):
  - 409.04 / 272.42 / 221.88, vs `v2-twopart-nn` 403.34 / 271.90 / 221.08.
  - Paired day bootstrap, clean, ref minus new: -0.81 s [-1.70, +0.23]. Slightly worse, within noise.
  - The models ran 14-18 epochs; the validation RMSE plateaued at the same level (199.3 / 193.9 /
    201.4 vs 197.6 / 190.1 / 199.8 on jan).
  - Network alone, clean: 235.32 vs 234.36.
  - Cost: about 2.5x (26-28 min per fold vs 12-14).
- Verdict: the latent is not under-trained. The early stop at epochs 6-9 marks a real plateau of the
  validation slice. Keep lr 1e-3 / patience 2. Seeds of that setting follow (pending).
- Seed design (fixed before running):
  - Training seeds 1 and 2 (init and shuffling) for the full model (`s1`, `s2`) and for `noarr`
    (`noarr_s1`, `noarr_s2`). The day groups are the same.
  - Latents use fp32 inference and are stacked on the current reference blocks.
  - Seed 20260926 is `latent_set_fp32` (full model) and `latent_set_noarr_fp32`.
  - The spread of 3 seeds per variant is the latent's noise floor. `noarr` is preferred if its
    seed mean is within that spread of the full model's (the lead's rule: 2.5x cheaper).
- Seed results, jan+jul, stacked on the reference blocks (all / core / clean):

  | latent | seed 20260926 | seed 1 | seed 2 | mean clean | range clean |
  |---|---|---|---|---|---|
  | full | 399.22 / 270.70 / 219.61 | 397.13 / 268.01 / 220.65 | 407.87 / 268.74 / 219.82 | 220.03 | 1.04 |
  | `noarr` | 407.83 / 271.16 / 220.38 | 378.83 / 267.78 / 219.65 | 401.87 / 268.99 / 219.84 | 219.96 | 0.73 |

  - The latent's seed noise on the stacked model is about 1 s clean (sd about 0.5), 2-3 s core and
    up to 29 s on all flights. That confirms that all-flight differences between single runs are
    meaningless.
  - Network alone, clean: full 234.4 (bf16 run) / 235.1 / 233.0; `noarr` 235.5 / 233.6 / 234.6.
  - `noarr` is equivalent to the full model within the seed spread (mean 219.96 vs 220.03).
- Seed averaging (not planned, suggested by the spread):
  - `nn_pred`, quantiles and `nn_pfill` are averaged over the 3 seeds; z comes from the first seed.
  - Stacked on jan+jul, full model: `seedavg` 401.07 / 268.31 / **218.63**; bootstrap vs seed
    20260926: clean +0.97 s [+0.49, +1.47], core +2.26 s [+0.14, +5.12]. That is 1.4 s better than
    the mean single seed, and it clears the noise floor on both core and clean.
  - `noarr` 3-seed average: 400.01 / 269.62 / **218.65**; clean +0.96 [+0.24, +1.66], core +1.21
    [-4.65, +6.51].
  - Network alone, clean: 231.6 (full average) and 232.1 (`noarr` average), vs about 234 single.
- Decision:
  - Adopt a 3-seed averaged latent.
  - Full and `noarr` are equivalent on clean; the full average is the only one significant on core.
    Given the lead's cost rule and the tie, prefer the `noarr` 3-seed average: 3 x 5 min per fold,
    and 3 x about 7 min for the final instead of 3 x 17. The full average remains the fallback if
    core matters more.
  - For the ranking file this needs final runs for seeds 1 and 2 (not done yet).


## EXP-016 Window length 3 h vs 2 h (2026-09-26, derivation written before the run)

- Arm `ctx60`: 60 min of context on each side instead of 45, EXP-010 training settings. Stacked on
  the current reference blocks with fp32 re-inference (`latent_set_ctx60_fp32`), compared with
  `v2-twopart-nn32-schedrel-off-wx`.
- Derivation:
  - If longer-range state matters (bank structure, arrivals more than 45 min before the take-off
    of long taxi-outs): clean improves by more than 0.7 s, mostly in the top quintile of
    `dep_between_aobt_to`.
  - If not: the change is within noise. EXP-010's gain was flat across congestion quintiles, so
    I expect no change, at about 1.5-2x the attention cost.
- Result (fp32 re-inference of the ctx60 checkpoints, stacked on the reference blocks, jan+jul):
  - 405.83 / 270.83 / 219.48, vs 399.22 / 270.70 / 219.61.
  - Bootstrap: clean +0.12 s [-0.52, +0.77], core +0.03 s [-5.21, +4.45].
  - Network alone clean 233.80, inside the single-seed range (233.0-235.1).
  - Cost: 15-16 min per fold (1.2x), 4.1 GB GPU with expandable segments (9 GB reserved without).
- Verdict: no gain, as derived. The relevant context is within 45 min. Keep 2 h windows.


## EXP-017 Latent mechanism ablations (2026-09-26, derivation written before the runs)

- Each arm removes one component from EXP-010 (not cumulative), with EXP-010 training settings;
  compare with `v2-twopart-nn` (jan+jul clean 221.08):
  - `nostand`: no stand-pair bias.
  - `norw`: no runway-pair bias.
  - `noarr`: no arrival tokens (departures only).
- Derivation:
  - Stand: the stand-occupancy bound is already in each departure token's hand features
    (`st_*`), but only against its own stand's last arrivals. If the pairwise bias carries the
    gain, `nostand` loses more than 0.7 s clean, concentrated at EGLL and LFPG (high bound
    coverage). If the gain lies elsewhere, the loss is within noise.
  - Runway: runway identity is also a token embedding, so attention can recover "same runway"
    from embedding products. I expect a small loss (under 1 s) unless the explicit pairing of
    time gap and runway is what matters.
  - Arrivals: the only field the hand features lack is the arrivals' taxi-in and in-block
    process. That hand block was worth -1.4 s (EXP-013). If arrivals are only that, `noarr` loses
    about 1-2 s. If the network reads more from arrivals (landing flow as a proxy for runway
    configuration, gate turnaround), it loses more. If `noarr` loses nothing, the latent is a
    departure-sequence model, and the mechanism is the joint reconstruction of neighbours'
    off-blocks.
- Results, jan+jul stacked (all / core / clean), and paired day bootstrap on clean vs
  `v2-twopart-nn` (221.08; ref minus new, negative = the ablation is worse):

  | arm | all | core | clean | bootstrap clean | network alone clean |
  |---|---|---|---|---|---|
  | EXP-010 (full) | 403.34 | 271.90 | 221.08 | - | 234.36 |
  | `nostand` | 404.76 | 271.80 | 221.08 | -0.03 [-0.75, +0.79] | 234.51 |
  | `norw` | 400.34 | 272.96 | 221.40 | -0.31 [-1.18, +0.55] | 234.20 |
  | `noarr` | 414.75 | 273.54 | 220.95 | +0.10 [-0.78, +1.14] | 235.49 |

  - Per-airport clean deltas are within about +-1.5 s, except LIRF (-1 to -6 s, i.e. better, on
    every arm, including the lr3e4 arm). LIRF is the least stable airport of the stack, so this is
    not attributable to any one component.
  - Without arrivals, the network alone is 1.1 s worse on clean (+3.3 s at EDDF, +3.8 s at LSZH),
    but the stacked model does not lose it: the GBDT recovers that part from the hand features.
  - All-flight RMSE moves by several seconds between arms (400-415) on the same clean. That is the
    tail re-shuffling, not a component effect.
- Interpretation:
  - None of the three explicit relations carries the gain. Stand-pair bias, runway-pair bias and
    arrival tokens are each individually redundant for the stacked model.
  - What remains is the core of the model: departures attending to departures, with the signed
    time-gap bias and each neighbour's own features (NM offsets, queue counts, stand bounds). The
    mechanism is the joint reconstruction of neighbours' off-blocks from their own evidence. That is
    the EXP-008 oracle quantity, estimated jointly instead of in two stages (EXP-012).
  - Caveat: the arms are single-seed, and the seed spread is not measured yet. "Redundant" means
    "within about 1 s", not zero.
- Decision: keep the full EXP-010 model as adopted, since no single removal gains. A departures-only
  model (`noarr`) is a 2.5x cheaper equivalent for the stack (5 min per fold instead of 12-14) and
  the natural base for further variants.


## EXP-021 February stress fold, and the AOBT_3 offset for extrapolation (2026-09-26, lead)

- Motivation: January 2026 contains a disruption at EHAM far outside 2025 (see the observation
  above), and trees cannot extrapolate. The only comparable 2025 event is LTFM snow in February.
- Stress fold `feb`: February 2025 held out, so the model never sees a snow day.
- Arms, all two-part on base + seq + stand + schedrel (no network, which has no feb outputs):
  - `v2-feb-noarr`: 272.64 / 266.10 / 224.47 (all / core / clean).
  - `v2-feb-arr` (+ arrivals' taxi-in): 274.44 / 267.26 / 226.28, no help.
  - `v2-feb-noarr-off` (regression boosted from take-off minus AOBT_3, clipped at 3 h, as
    `init_score`): **268.62 / 262.18 / 222.89**.
  - On LTFM 2025-02-23 (true mean 2,953 s), the mean prediction is 2,107 without the offset and 2,219
    with it. Both under-predict heavily; at LTFM AOBT_3 is a default value, so the offset can only
    help a little there.
- Main folds, `v2-twopart-nn-schedrel-off` vs `v2-twopart-nn-schedrel`: jan+jul 399.98 / 272.06 /
  220.15 vs 402.79 / 271.44 / 219.59; Q4 239.69 / 216.47 / 200.78 vs 240.38 / 215.64 / 200.37.
  Paired day bootstrap, clean, jan+jul: offset worse by 0.56 s [0.02, 1.07].
- Interpretation: a small, real cost on ordinary days against a gain on the stress fold. The offset
  matters where AOBT_3 is a genuine observation and the day is outside the training range, which is
  exactly EHAM on 2-7 January 2026, the case we cannot validate locally.
- Decision: use the offset in submission models, as insurance. Revisit after the leaderboard reading.


## EXP-022 Hourly weather block (Open-Meteo / ERA5) (2026-09-26, lead)

- Hypothesis: January error concentrates on winter disruption days (EXP-013), and January 2026 has an
  EHAM snowstorm, so weather (snowfall, frozen precipitation, icing conditions, wind) should reduce
  error on such days.
- Data: `scripts/external/fetch_weather.py`, hourly at aerodrome reference points, 2024-12-25 to
  2026-08-02, CC BY 4.0.
- Features (`wx` block, joined at the take-off hour): temperature, dew-point depression,
  precipitation, snowfall, snow depth, weather code, low cloud, wind speed/gust/direction; trailing
  3/12 h snowfall, 3 h precipitation, 6 h minimum temperature, 3 h icing hours, 6 h frozen hours.
- Signal check (2025, non-fill, target < 2 h): mean target 1,196 s with frozen precipitation in the
  previous 6 h vs 978 s otherwise. EHAM 2-7 Jan 2026 shows 4-7 cm snowfall per 12 h and 10-16 cm of
  snow on the ground.
- RMSE pooled jan+jul (all / core / clean):
  - `v2-twopart-nn-schedrel-off-wx`: **391.90 / 268.95 / 219.47**.
  - Reference `v2-twopart-nn-schedrel-off`: 399.98 / 272.06 / 220.15.
  - Q4: 241.34 / 214.81 / 199.44 vs 239.69 / 216.47 / 200.78.
  - Paired day bootstrap, clean: jan+jul +0.69 s [0.15, 1.20]; jan +1.00 s [0.46, 1.53].
  - July core improves by about 5 s (300.17 vs 305.31).
- Stress fold `feb` without the network (`v2-feb-noarr-off-wx` vs `v2-feb-noarr-off`): 267.96 / 262.02 /
  225.12 vs 268.62 / 262.18 / 222.89. The LTFM snow days stay heavily under-predicted (2025-02-23:
  mean prediction 2,237 vs truth 2,953). With the only LTFM snow episode held out, the trees have no
  example to learn from; EDDM and EHAM improve in February.
- Decision: adopt `wx`. The submission model becomes `v2-twopart-nn-schedrel-off-wx`. Extrapolation to
  the EHAM January 2026 storm remains the main unvalidated risk.


## EXP-018 CPU inference: reproduction without a GPU, and a CUDA/CPU divergence found (2026-09-26)

- Requirement: the organisers may not have a GPU. Training stays on GPU; inference and
  reproduction must run on CPU.
- Code:
  - `scripts/latent_set/infer.py`: inference only, from the saved checkpoints; `--device cpu|cuda`,
    `--amp auto|off`, `--scope full|target`.
    - `--amp auto`: bf16 autocast on CUDA, fp32 on CPU. There is never autocast on CPU.
    - `--scope full`: rebuilds the whole fold/final file. Each group model predicts its own
      out-of-fold days plus the target windows, then the same `assemble` (mean, z alignment, PCA)
      runs as in training.
  - `prc.latent_set.oof.assemble` is shared by training and inference;
    `prc.latent_set.train.model_from_state` rebuilds a model from a checkpoint.
- First comparison, jan holdout (153,706 departures, mean of 3 models, `nn_pred`):
  - saved training output vs GPU bf16 re-run: mean |diff| 0.0008 s, max 3.1 s (GPU is near
    deterministic);
  - GPU bf16 vs GPU fp32: mean 1.16 s, p99 4.8 s, max 17 s;
  - GPU fp32 vs CPU fp32: mean 0.57 s, p99 6.7 s, max **61 s**. Not a precision issue.
- Cause (layer-by-layer check): the attention time-gap buckets.
  - `dt / 60.0` is computed on CUDA as `dt * float32(1/60)`, a reciprocal multiply, so 600 s
    becomes 10.000001 min. CPU divides exactly (10.0).
  - On the 11 bucket edges that are whole minutes (5, 7, 10, ... 120 min), the two devices put the
    pair in different buckets.
  - Exact whole-minute gaps are common, because several airports log minute timestamps.
  - Only the bias differed (up to 4.2 logits); embeddings matched to 4e-6.
- Fix (`prc.latent_set.model.dt_bucket`): compute `dt * (float32(1) / float32(60))` explicitly.
  - This reproduces the CUDA training semantics bit for bit on every device (checked on every
    integer gap in +-20,000 s), so the trained checkpoints stay valid.
  - After the fix, CPU fp32 vs GPU fp32 differ by at most 8e-6 at every layer.
- `final.parquet` on CPU (`data/processed/latent_set/final_cpu.parquet`, 8 threads, fp32, full scope):
  - vs GPU fp32 (`latent_set_fp32/final.parquet`): `nn_pred` max |diff| 0.004 s (ranking rows
    0.0017 s), `nn_pfill` 8e-7, z 1e-5. Reproduced exactly.
  - vs the adopted `final.parquet` (GPU bf16): `nn_pred` mean 1.74 s, max 55 s (ranking rows: mean
    1.18 s, max 23 s). This is entirely the bf16 inference of the training run.
- bf16 vs fp32 for the stack:
  - The fold and final files were regenerated with fp32 inference from the same checkpoints
    (`data/processed/latent_set_fp32/`, GPU, 40 s per fold).
  - bf16 vs fp32 `nn_pred`: mean |diff| 1.6-1.8 s, max 34-55 s; mean signed diff +0.01 to +0.65 s.
  - Re-stacked `v2-twopart-nn-schedrel-fp32` vs `v2-twopart-nn-schedrel`: jan+jul 400.70 / 271.76 /
    219.75 vs 402.79 / 271.44 / 219.59; q4 238.78 / 215.73 / 200.40 vs 240.38 / 215.64 / 200.37.
  - Paired day bootstrap, clean: jan+jul -0.16 s [-0.58, +0.23]; q4 -0.02 s [-0.34, +0.31]. No
    difference.
  - Per flight, the stacked predictions move by 22 s on average (max 13,980 s on a tail row): the
    two-part GBDT is chaotic to a 1-2 s perturbation of one feature. This is a warning for
    reading any single-flight difference between runs.
- CPU wall time (8 threads, fp32, batch 64 windows, shared machine):
  - ranking file only (344,841 departures, 24,758 windows, 3 models): **157 s** (12 s event build +
    about 48 s per model);
  - full final reproduction including 2025 out-of-fold (2,429,888 rows): **469 s**;
  - one month holdout (jan, 3 models): 92 s.
  - GPU for comparison: 47 s full final.
- Recommendation:
  - Use fp32 inference on both sides: `data/processed/latent_set_fp32/{jan,jul,q4,final}.parquet`.
    Training features and ranking features are then produced by the same arithmetic, and a CPU-only
    reproduction matches them to 0.004 s.
  - Keep bf16 for training only; training in fp32 is not needed.
  - The adopted bf16 files stay unchanged; switching costs nothing measurable (bootstrap above).
  - Reproduction command:
    `uv run --group timesfm python scripts/latent_set/infer.py --target final --device cpu --out <path>`.


## EXP-019 Weather in the latent tokens (2026-09-26, derivation written before the run)

- Arm `wx`: every token (departures and arrivals) gets the 17 hourly `wx_*` values of its airport at
  the hour of its reference time, plus a missing flag.
  - Fixed physical scalings, no standardisation: temperatures / 10; log1p for precipitation and
    snowfall; log1p(cm) for snow depth; wind / 20.
  - Why: January 2026 lies outside the 2025 range (snow depth 16 cm vs 9 cm, 12 h snowfall 7.1 vs
    5.7 cm). Standardised inputs would hit the +-8 sd clamp, while these stay proportional.
- Folds: jan, jul and the `feb` stress fold (February 2025 held out: the only LTFM snow episode,
  2025-02-23 true mean 2,953 s). A no-weather baseline on `feb` is trained with the adopted setting
  (`latent_set_fp32/feb.parquet`).
- Stacking: all comparisons use the current reference blocks (base, seq, stand, schedrel, wx, offset
  aobt_proxy) with fp32 latents. The reference is `v2-twopart-nn32-schedrel-off-wx`.
- Derivation:
  - If weather in the tokens helps:
    - The network alone improves on jan but not on jul.
    - The gain concentrates on frozen-precipitation days (EHAM, EDDM, LSZH, LTFM).
    - On `feb`, the network alone under-predicts the LTFM snow days less.
    - The stacked gain is smaller than the network-alone gain, because the GBDT already has the
      `wx` block, likely below the noise floor on clean.
    - For January 2026 at EHAM, daily nn_pred rises on 2-7 January relative to the no-weather
      latent.
  - If not: the network alone is unchanged within its seed spread, and `feb` LTFM days are unchanged.
    The latent already reads a disrupted day from the event flow (take-off rate, AOBT_3 offsets,
    arrival taxi-in), so weather is redundant at the token level.
  - Neither outcome can fix the flight-level saturation at the 3600 s training clip seen in
    EXP-014; that is a target-definition choice, not an input one.
- Adoption rule: the stacked jan+jul change must exceed the seed spread (EXP-015) on core and clean,
  or `feb` must show a clear network-alone gain on snow days without a cost on jan/jul.
- Results, network alone (clean RMSE of `nn_pred`):
  - jan: 206.95 -> 204.72; jul: 254.55 -> 252.07; jan+jul pooled 232.00, vs single seeds 233.0-235.1
    without weather.
  - The jan gain is not on frozen-precipitation flights (266.4 -> 267.3, n = 6,279) but spread over
    other days and airports, and July gains as much. Weather acts as general conditioning (wind,
    convective days), not a snow effect, on these folds.
  - `feb` stress fold, 2 seeds each:
    - clean: no weather 232.81 / 232.76, weather 227.39 / 222.42;
    - frozen-precipitation flights (n = 8,605): 555 / 557 -> 523 / 490;
    - LTFM: 436 / 436 -> 417 / 392.
    - LTFM snow days, daily mean nn_pred vs clipped truth:
      - 2025-02-23 (truth 2,503): 2,030-2,063 -> 2,107-2,330;
      - 2025-02-19 (truth 2,397): 1,843-1,912 -> 1,956-2,065.
    - Robust over both seeds: weather tokens reduce the under-prediction on snow days by 100-300 s.
- Results, stacked (reference blocks, which already include the GBDT `wx` block):
  - jan+jul: 387.64 / 269.72 / 219.49 vs 399.22 / 270.70 / 219.61. Bootstrap clean +0.13 s [-0.70,
    +0.86], core +1.00 s [-3.97, +6.08]. No gain beyond the seed spread.
  - `feb`: no-weather latent 213.92 / 216.86 clean (2 seeds); weather latent 219.18 / 212.75.
    - Non-LTFM flights are identical (178.4-178.8).
    - LTFM swings 371-400 with the latent seed.
    - The GBDT does not reliably pass the network's better snow-day estimate through: on LTFM snow
      days the stacked outcome is decided by seed noise.
  - For reference, the latent itself on `feb` (no weather): stacked clean 213.9 / 216.9 vs 225.12
    without any latent (`v2-feb-noarr-off-wx`). Bootstrap +11.0 s [+7.9, +14.9]; LTFM 405.6 ->
    377.6-389.3.
    - The latent extrapolates to a held-out snow episode better than the trees, with or without
      weather.
- January 2026 EHAM (`data/processed/latent_set_wx/final.parquet`, one seed), daily mean nn_pred
  without -> with weather:
  - 2 Jan 1,093 -> 1,172; 3 Jan 1,756 -> 1,918; 4 Jan 1,887 -> 2,164; 5 Jan 2,114 -> 2,231;
    6 Jan 1,662 -> 1,721; 7 Jan 1,810 -> 1,918.
  - Ordinary days (1, 8, 10 Jan) are unchanged (+-15 s).
  - Direction consistent with `feb`; still below the AOBT_3 proxy (2,330-2,604 on 3-5 Jan).
- Cost: training about 15 min per fold (weather adds 18 inputs, no measurable slowdown); the final
  run took 20 min.
- Verdict:
  - Weather in the tokens is a real network-level gain exactly where it matters (snow days, the
    January 2026 risk), consistent over 2 seeds and 2 datasets (feb 2025, EHAM 2026).
  - It is not yet a stacked gain: the two-part GBDT absorbs or scrambles it, and single-seed stacked
    comparisons on snow days are dominated by latent seed noise.
  - Next (running): a 3-seed average of the departures-only latent with weather
    (`noarr_wx_s{20260926,1,2}` on jan, jul, feb). This tests whether the seed average, which
    removes the seed noise, lets the stack keep the snow-day gain.


## EXP-023 CatBoost regression part, and LightGBM + CatBoost blend (2026-09-26, lead)

- Hypothesis: a second tree algorithm (CatBoost, ordered boosting, native categoricals, symmetric
  trees) makes different errors; averaging reduces variance, especially the per-flight instability
  of the two-part mixture (EXP-018).
- Config `v2-cb-nn-schedrel-off-wx`: identical to the LightGBM submission model except the regression
  part (CatBoost depth 8, lr 0.08, 3,000 iterations, 8 threads, same AOBT_3 offset as a baseline).
  The fill classifier stays LightGBM. Blend `v2-blend-lgb-cb` = 50/50 average
  (`scripts/protocol/blend.py`); weights not tuned.
- RMSE jan+jul (all / core / clean):
  - LightGBM `v2-twopart-nn-schedrel-off-wx`: 391.90 / 268.95 / 219.47.
  - CatBoost alone: 376.62 / 259.06 / 216.94. Bootstrap vs LightGBM: clean +2.71 [-2.47, 5.51], core
    +9.89 [-0.64, 19.49]. Better on average but not reliably.
  - **Blend: 376.98 / 255.38 / 214.58.** Bootstrap vs LightGBM: clean +4.95 [3.43, 5.98], core +13.53
    [7.55, 19.89], 100% of resamples positive. Every airport improves on core, from EHAM +3.9 s to
    LIRF +46.2 s; LSZH +21.1.
- Cost: CatBoost about 33-37 min per fold on 8 threads (LightGBM two-part about 3.5 min).
- Decision: **adopt the blend** for submissions. Candidate = average of the LightGBM and CatBoost
  candidate files (`scripts/protocol/blend_submissions.py`), each with fp32 latents (EXP-018).


## EXP-024 Seed bagging of the LightGBM two-part model (2026-09-26, lead)

- Motivation: EXP-018 showed per-flight predictions of the stacked two-part model move by about 22 s on
  average when one feature is perturbed by 1-2 s, and re-inference in fp32 moved the all-flight RMSE
  by 16 s. Variance, not bias.
- Arms (fp32 latents, `latent_set_fp32/`):
  - single seed `v2-twopart-nn32-schedrel-off-wx`: jan+jul 407.60 / 271.17 / 219.80.
  - 3 seeds averaged (`seeds: [20260926, 7, 11]`, `SeedBag`) `-bag3`: **399.80 / 269.11 / 217.98**; Q4
    235.22 / 212.84 / 198.06 (single: 240.96 / 215.15 / 199.59).
- Paired day bootstrap, clean, bag vs single: +1.83 s [1.47, 2.18]. Mean per-flight |difference| 18 s.
- Blend with CatBoost (EXP-023; CatBoost fold predictions with bf16 latents), `v2-blend-lgbbag3-cb`:
  jan+jul 380.78 / 255.57 / 214.15.
- Cost: 3x LightGBM (about 11-14 min per fold on 8 threads).
- Decision: submissions use the 3-seed bag for the LightGBM half of the blend (candidate v4).


## EXP-025 Departures-only latent with weather, 3-seed average: choice and ranking files (2026-09-26)

- Question (the rule fixed in EXP-019 and agreed with the lead): does a 3-seed average of the
  departures-only latent with weather tokens (`noarr_wx`) beat the 3-seed average without weather
  (`noarr`)?
  - The bar: beyond noise on jan+jul clean or core, or clearly on `feb` snow days without a jan/jul
    cost.
  - Seeds 20260926, 1, 2. Heads are averaged; z comes from the first seed
    (`scripts/latent_set/seed_average.py`).
- Stacked on the reference blocks, `noarr_wx_avg3` vs `noarr` 3-seed average:
  - jan+jul (all / core / clean): 384.22 / 266.28 / 218.47 vs 400.01 / 269.62 / 218.65.
    - Bootstrap clean +0.18 s [-0.49, +0.83], core **+3.22 s [+0.08, +7.42]**.
    - vs the single-seed full model (`v2-twopart-nn-fp32`, 219.61): clean +1.11 [+0.40, +1.83],
      core +4.53 [-2.13, +10.87].
  - `feb` (3-seed average of each): 242.03 / 235.99 / 211.96 vs 241.90 / 235.76 / 213.81.
    - Bootstrap clean **+1.77 s [+0.15, +3.80]**, core -0.13 [-3.76, +3.25].
    - Frozen-precipitation flights 475.3 -> 462.1, LTFM 378.7 -> 370.2; other flights identical
      (176.45 vs 176.49).
  - Network alone, clean: jan+jul 231.15 (vs 232.09 without weather); feb 219.45 vs 226.80.
  - Seed averaging also stabilises `feb`: single-seed stacked clean ranged 212.8-219.2; the averages
    give 213.8 and 212.0.
- Decision: **`noarr_wx` 3-seed average**. It is equal on jan+jul clean, better on jan+jul core
  beyond the bootstrap interval, and better on the snow stress fold without a cost elsewhere.
- Files: `data/processed/latent_set_noarr_wx_avg3/{jan,jul,feb,q4,final}.parquet`.
  - Same scheme as all folds: 3 day-group models per seed, fp32 inference, out-of-fold for 2025
    training rows, the mean of the 3 group models for holdout/ranking rows, then the mean over 3
    seeds.
  - Per-seed sources: `latent_set_noarr_wx_s{20260926,1,2}/`. Checkpoints:
    `artefacts/latent_set/{fold}_noarr_wx_s*_g*.pt`.
  - Training flags: `--weather --no-arrivals --seed S`, lr 1e-3, 16 epochs, patience 2, 2 h windows.
- Final checks (`scripts/latent_set/check_final.py --tag _noarr_wx_avg3`):
  - 2,429,888 rows, same columns as the fold files, no duplicates, no NaN.
  - All 344,841 ranking departures and all 2025 departures are covered.
  - Mean nn_pred on ranking vs 2025 OOF: EDDF 922 vs 864, EDDM 882 vs 818, EGLL 1361 vs 1376, EHAM
    886 vs 789, LEBL 952 vs 958, LEMD 1025 vs 1021, LFPG 1001 vs 1016, LIRF 1110 vs 1098, LSZH 801
    vs 742, LTFM 1126 vs 1052.
  - The same pattern as EXP-014 (+58 to +97 s at EDDF, EDDM, EHAM, LSZH, LTFM; the year-on-year
    drift). nn_pfill matches 2025 within 0.005.
  - z alignment R^2 per seed 0.86-0.91.
- EHAM 1-10 January 2026, daily mean nn_pred:
  - EXP-014 single-seed full latent vs `noarr_wx_avg3` (q90 mean; AOBT_3 proxy):
    - 1 Jan 733 vs 731 (908; 892)
    - 2 Jan 1,093 vs 1,110 (1,387; 1,258)
    - 3 Jan 1,756 vs 1,863 (2,268; 2,330)
    - 4 Jan 1,887 vs 2,005 (2,434; 2,595)
    - 5 Jan 2,114 vs 2,049 (2,487; 2,604)
    - 6 Jan 1,662 vs 1,652 (2,001; 1,942)
    - 7 Jan 1,810 vs 1,844 (2,204; 2,111)
    - 8 Jan 750 vs 766 (950; 877)
    - 9 Jan 1,373 vs 1,402 (1,690; 1,534)
    - 10 Jan 1,116 vs 1,165 (1,384; 1,303)
  - Averaging dilutes the single-seed weather lift on the storm days (EXP-019 seed 20260926 full:
    +60 to +280 s). The average still rises on 3-4 Jan (+107, +118).
  - The per-flight maximum stays at the 3600 s clip (3,592-3,642). The network remains below the
    proxy on the worst days; the flight-level cap is a target choice (EXP-014).
- CPU reproducibility of this variant: seed 1 ranking-only inference on CPU (8 threads, fp32)
  matches the GPU training-run output to 0.002 s max (mean 8e-5 s), in 55 s for 344,841 departures.
  Departures-only is 3x faster than the full model on CPU as well.
- Cost:
  - Per seed: jan+jul+feb 15 min, q4 5 min, final 6 min GPU; 80 min in total for the 3 seeds.
  - The `noarr` (no weather) feb comparison seeds took another 18 min.


## EXP-026 Blend on the 3-seed departures-only + weather latent (candidate v5) (2026-09-27, lead)

- Change vs v4: the latent is `latent_set_noarr_wx_avg3` (EXP-025) instead of the single-seed full
  latent. LightGBM 3-seed bag + CatBoost regression, 50/50 blend, same blocks (base, seq, stand,
  schedrel, wx) and AOBT_3 offset.
- RMSE jan+jul (all / core / clean):
  - LightGBM bag 386.42 / 265.25 / 216.83 (Q4 233.39 / 212.59 / 197.55);
  - CatBoost 378.63 / 252.04 / 213.57;
  - **blend 376.79 / 252.29 / 212.57** (v4 blend: 380.78 / 255.57 / 214.15).
- Paired day bootstrap, v5 vs v4: clean +1.50 s [0.59, 3.03]; core +3.30 s [-2.09, 8.80].
- Decision: adopt; candidate v5 built (`candidate_blend-avg3-v5`), upload name `quick-quicksand_v3`.


## EXP-027 OpenStreetMap airport geometry block (2026-09-27, lead)

- Hypothesis: the model knows stand x runway pairs only as categories, so it has little to go on for
  rare or unseen pairs (4.6% of EDDM 2026 departures use stands unseen in 2025). A physical
  stand-to-threshold distance generalises across pairs.
- Block `geo` (EXP-027 decision entry): stand and threshold positions, distances, runway length,
  match-level flag. Coverage 96-100% of departures in train and ranking.
- Config `v2-lgbbag3-avg3-geo` vs `v2-lgbbag3-avg3` (LightGBM 3-seed bag, latent `noarr_wx_avg3`):
  - jan+jul 394.61 / 264.67 / 216.42 vs 386.42 / 265.25 / 216.83; Q4 231.54 / 212.60 / 197.22 vs
    233.39 / 212.59 / 197.55.
  - Paired day bootstrap, clean: jan+jul +0.40 s [0.17, 0.66]; Q4 +0.33 s [0.15, 0.51].
    Core: jan+jul +0.59 [-0.66, 1.67], Q4 0.
- By training count of the flight's stand x runway pair (clean, jan+jul): 1-10 prior flights 301.3
  -> 293.5 s (n = 2,607); 11-50: 269.6 -> 268.2; more than 200: 210.3 -> 210.2; unseen pairs
  (n = 429) unchanged.
- Interpretation: small, consistent, and located where the hypothesis says (rare pairs). Unseen pairs
  do not improve, probably because unseen stands are also unmatched in OSM or too few to learn from.
- Decision: adopt `geo` in the pipeline; no separate upload for it. The CatBoost half gets it at the
  next candidate build.


## EXP-028 Airline code and record-quality flags (2026-09-28, lead)

- Finding that motivated it: after the first leaderboard scores, the decomposition of the v5 fold
  error shows 60 flights with target > 3 h carry 55% of jan+jul squared error. Of the 202 such flights
  in 2025, 188 are at LIRF; 96% have no NM match and 90% are schedule fills. Among LIRF no-NM
  departures about half are fills, and the fill rate depends strongly on the carrier (ISR 98%, ETH
  92%, EXS 79% vs EZY 8%, EJU 10%, ITY 16%). The model could not see the carrier: the NM operator is
  anonymised and empty without an NM match.
- A duplicate-record hypothesis (the original schedule stored in a second record of the same flight)
  was checked and rejected: the other records are the same flight number on neighbouring days.
- Features (block `airline`):
  - `airline`: first 3 letters of FLIGHT_mvt, else the 2-character IATA code, which may contain a
    digit; 99.3% coverage.
  - Record-quality flags: `flight_no_odd` (garbled, e.g. WMTT9066, or missing flight number) and
    `actype_missing`. The 239 LIRF no-NM flights with such records are 74% fills.
- LightGBM 3-seed bag, jan+jul (all / core / clean), vs `v2-lgbbag3-avg3-geo` 394.61 / 264.67 / 216.42:
  `v2-lgbbag3-avg3-geo-al` 394.12 / 265.22 / 216.53. Q4 241.62 / 212.61 / 197.46 vs 231.54 / 212.60 /
  197.22. No aggregate change.
- Targeted check, single seed with fill probabilities saved (`v2-fillcheck-{noal,al}`):
  - fill AUC on LIRF no-NM rows 0.8525 -> 0.8749 (n = 397);
  - their RMSE 7,396 -> 6,806 s;
  - overall AUC unchanged (0.881 / 0.880).
- Interpretation: the feature does what it was designed for on the subset that matters, but that
  subset is about 400 flights per fold pair and its effect disappears in the all-flight noise.
- Decision: keep `airline` (no cost on core/clean, targeted gain). The main tail question stays open:
  see EXP-029.


## EXP-029 Latent training target clipped at 7,200 s instead of 3,600 s (2026-09-28, derivation written before the runs)

- Motivation: per-flight nn_pred saturates at 3,600-3,650 s on the EHAM snowstorm of 2-7 January
  2026, while the AOBT_3 proxy goes far higher (EXP-014, EXP-019, EXP-025).
  - The cap is the training clip `Y_CLIP`.
  - Raising it frees 2,296 non-fill 2025 training targets in (3,600, 7,200] s: 0.12% of non-fill
    rows. They are concentrated at LTFM (756), EGLL (665), LFPG (349) and LIRF (297), and 44 of
    them at EHAM.
- Arm `noarr_wx_clip7200`: the adopted variant (departures only, weather tokens), `--y-clip 7200`.
  - The MSE loss and the fill exclusion are kept; lr 1e-3, 16 epochs, patience 2.
  - The early-stopping metric then also scores against targets clipped at 7,200 s.
  - Seeds 20260926, 1, 2 on jan, jul and feb. Heads are averaged as for
    `latent_set_noarr_wx_avg3`; z comes from the first seed.
- Stacking: `configs/v2-lgbbag3-avg3-geo-al.yaml` with the latent directory swapped, compared with
  `experiments/v2-lgbbag3-avg3-geo-al` (jan+jul 394.1 / 265.2 / 216.5) using
  `scripts/latent_set/compare.py`. `feb` gets the same reference config for a fair pair.
- Derivation:
  - If the clip limits long-taxi and snow days:
    - Network alone: better on flights with a target of 3,600-7,200 s, and on the `feb` LTFM snow
      days (less under-prediction than the 2,415 / 2,247 s daily means against truths of 2,953 /
      2,766 s).
    - Stacked jan+jul: core improves beyond 2-3 s, clean unchanged within about 1 s.
    - EHAM 3-5 January 2026: daily nn_pred rises toward the AOBT_3 proxy (2,330-2,604 s), and the
      per-flight maximum passes 3,600 s.
  - If not:
    - The 2,296 freed targets are too rare to move the fit, or the MSE on them only adds variance.
    - Network-alone clean may worsen slightly (a heavier tail in the loss).
    - Stacked metrics unchanged within the seed noise, and a larger seed spread.
  - Mechanism caveat: the stacked GBDT has its own regression on aobt_proxy (the offset), so part of
    any network-level gain on long taxis may already be covered. A network-alone gain without a
    stacked gain would read as "redundant with the offset", not "useless".
- Adoption rule (lead): stacked jan+jul core or clean beats the reference beyond the noise (about
  1 s clean, 2-3 s core) with no clean cost; or a clear `feb` snow-day gain with no jan/jul cost.


## EXP-030 AOBT_3 relative to the airport-day default (2026-09-28, lead; derivation written before the run)

- Finding (label-free, ranking file):
  - At LTFM, NM's AOBT_3 is mostly take-off minus a default taxi time: 52% of a day's flights carry
    exactly the day's modal value (EDDM 23%, LIRF 14%, the others under 8%).
  - The default switches between periods: 2025 about 1,200 s in most months and 1,440 s in January,
    March and November; **January 2026 about 1,500 s, a value never seen in 2025**.
  - Actual LTFM taxi-out does not follow it: the 2025 monthly median stays at 910-1,014 s whatever
    the default (y - AOBT_3 proxy median -355 s in the 1,440 months, -240 s in the 1,200 months).
  - Candidate v5 predicts LTFM January 2026 daily medians of about 1,100-1,300 s on 1,500-s days
    against 950-1,040 s on 1,200-s days, without snow. Our median prediction for LTFM January 2026
    is 145 s above the January 2025 truth, while arrivals' taxi-in did not change (-9 s).
  - Mechanism: the AOBT_3 offset (EXP-021) and the raw proxy feature carry an NM parameter change
    into the predictions. Estimated leaderboard cost if about 15,000 flights are biased by about
    150 s: 1.5 s RMSE.
- Change:
  - features `px_day_default` (the airport-day modal AOBT_3 proxy, 10 s resolution), `px_dev`
    (deviation from it), `px_is_default`, `px_day_default_share`;
  - offset mode `aobt_proxy_dev`: at airports whose median daily default share exceeds 0.3
    (decided on the training rows, which selects LTFM), the boosting start point is the airport's
    median training target plus `px_dev`; elsewhere unchanged.
- Derivation:
  - On the 2025 folds the change should be neutral within noise, since training covers the 1,200
    and 1,440 regimes. A small clean gain at LTFM is possible.
  - The effect that matters is on 2026 and cannot be validated with labels. The check: after the
    change, LTFM January 2026 daily medians on 1,500-s days should fall to the level of the
    1,200-s days (about 950-1,050 s), except on the snow days 18-19 January.
- Adoption rule: neutral or better on jan+jul core and clean (beyond -0.7 s clean counts as a cost),
  and the 2026 LTFM check passes.
- Runs on the new deterministic cache (sorted rows, LightGBM `deterministic`), paired with a fresh
  reference `d-ref` and with the EXP-029 clip latent `d-clip`: the three share row order and seeds.
- 2026-09-28 results (deterministic cache; jan+jul all / core / clean; feb = stress fold):
  - `d-ref` 395.44 / 265.28 / 216.44; feb 237.54 / 232.26 / 209.56.
  - `d-pxdev` 392.01 / 265.01 / 216.22 (bootstrap clean +0.22 [-0.24, +0.65], core +0.28);
    **feb 272.87 / 268.23 / 247.54** (clean -36.0 [-68.4, -9.5]); LTFM feb core 427 -> 571.
  - Reason: on the 2025 LTFM snow days NM raised the day's default taxi time itself. The default
    shift carries real disruption information, and the deviation-from-default offset erased it.
    NM's default moves for two reasons, seasonal parameter changes (January 2026, mild weather,
    arrivals' taxi-in unchanged) and genuine disruption (snow); only weather separates them.
  - Decision: reject the `aobt_proxy_dev` offset. Next: keep the plain offset and add the `pxdev`
    features, so the trees can combine the default level with weather (`d-pxfeat`).

## EXP-029 follow-up (lead, 2026-09-28): clip 7,200 s result

- The run finished the 3 seeds on jan, jul and feb, then stopped early. The lead
  seed-averaged them (`latent_set_noarr_wx_clip7200_avg3`) and stacked them as `d-clip` on the
  deterministic cache, paired with `d-ref`.
- jan+jul 386.00 / 264.68 / 216.80 vs 395.44 / 265.28 / 216.44: clean -0.36 [-0.88, +0.12], core
  +0.44 [-3.61, +5.93]. feb 240.00 / 234.64 / 211.17 vs 237.54 / 232.26 / 209.56: clean -1.58 [-2.91,
  -0.52], core -2.33 [-4.97, -0.27].
- Decision: reject. The 2,296 freed targets add variance, not signal; feb (the snow stress fold)
  is worse.
- `d-pxfeat` (plain offset + `pxdev` features): jan+jul 391.50 / 264.99 / 216.06, clean +0.37 [0.13, 0.61],
  core +0.26 [-0.68, 1.33]; feb 237.33 / 231.92 / 209.83, neutral (clean -0.27 [-1.07, 0.49]).
  Adopted for its small clean gain; the snow information is preserved.
- 2026 check (full-year model, `candidate_d-pxfeat`): LTFM January 2026, snow days 1, 18 and 19
  excluded, median prediction on days with default >= 1,450 s: v5 1,160 s -> 1,152 s; days < 1,450 s:
  996 -> 993 s. **The inflation is not fixed**: the trees cannot map an unseen default value (1,500 s)
  to the known regimes.
- Open proposal (needs a team decision; not validatable with 2025 labels): snap small shifts of the
  airport-day default to the nearest 2025 default (1,500 -> 1,440 s when within 100 s), so that
  NM's January 2026 parameter change is treated like the 2025 1,440 regime, while snow-day defaults
  (1,736-1,797 s) stay unchanged. Evidence for: 2025 regime changes did not move actual taxi-out,
  arrivals' taxi-in unchanged, mild weather on those days. Stakes: about 13,000 flights, estimated
  +/-1.3 s leaderboard RMSE depending on which reading is right.
- **Correction, 2026-09-28 evening (lead): the EXP-030 premise was wrong; proposal withdrawn.**
  - At day level (modal proxy per airport-day), the 2025 LTFM defaults are {1,200 s, 1,500 s}. The
    monthly "1,440 s" values were mixtures of 1,200-s and 1,500-s days, so 1,500 s is not new in
    2026.
  - In 2025, LTFM taxi-out on 1,500-s days was really longer: daily median 1,075 s over 109 days vs
    911 s over 250 days on 1,200-s days, +164 s, consistent in every month. Arrivals' taxi-in was
    not longer (722 vs 780 s), so the default tracks a departure-specific operating mode, probably
    the runway configuration.
  - Therefore the +160 s predicted on January 2026 1,500-s days is expected behaviour, not a bias.
    The snap correction was implemented, then withdrawn before any use: under the
    2025 reference it changes nothing, and its rationale is refuted. Lesson: check regime effects
    at the level where the regime is defined (the day), not on monthly aggregates.
  - `pxdev` features stay adopted for their small measured gain (+0.37 s clean).


## EXP-031 Night campaign: Copernicus ERA5 vs Open-Meteo, full stack with the new features (2026-09-28/29, lead)

- Setup (one script, same deterministic cache except the weather columns; verified that only
  `wx_*` differ):
  - latent `noarr_wx` retrained on Copernicus weather, 3 seeds, folds jan/jul/feb/q4 and final
    (`latent_set_noarr_wx_cds_avg3`);
  - LightGBM 3-seed bag and CatBoost two-part models on blocks base, airline, seq, stand, schedrel,
    wx, geo, pxdev with the AOBT_3 offset, for both weather sources.
- RMSE jan+jul (all / core / clean):
  - `n-om-lgb` 395.31 / 265.61 / 216.12 (Q4 240.58 / 212.46 / 197.20);
  - `n-cds-lgb` 396.99 / 265.41 / 216.08 (Q4 244.50 / 212.17 / 197.37);
  - `n-om-cb` 382.57 / 250.44 / 212.79; `n-cds-cb` 392.54 / 264.60 / 212.63;
  - **`n-om-blend` 381.90 / 249.49 / 211.98**; `n-cds-blend` 387.23 / 254.90 / 212.01.
- Paired day bootstrap:
  - Copernicus vs Open-Meteo blend: clean -0.03 [-0.47, +0.39], core -5.21 [-12.85, +0.27];
  - new features (Open-Meteo) vs v5 `v2-blend-avg3`: clean +0.58 [+0.24, +0.99], core +2.68
    [-0.48, +6.11].
- Interpretation:
  - The weather source does not matter for LightGBM or on clean flights. The core gap comes from
    the CatBoost half alone (264.6 vs 250.4), so it is more likely CatBoost instability on tail rows
    than a weather effect.
  - By the rule fixed before the run (Copernicus unless worse beyond noise), Open-Meteo stays.
  - The Copernicus pipeline remains available (`PRC_WEATHER=cds`) if the Open-Meteo API terms
    become a problem.
- Decision: the new Open-Meteo blend is the reference and the next candidate (`configs/final-*.yaml`,
  `scripts/reproduce.sh models`). Follow-up: seed-bag the CatBoost half, since its single-seed
  variance appears to be the largest remaining noise source in core.


## EXP-032 How much of the leaderboard error is the fill tail? (2026-09-30, lead)

- Method: one two-part LightGBM (final-lgb blocks) fitted on all 2025 and applied to the ranking file.
  Per flight, the expected squared error from not knowing fill vs non-fill is
  p(1-p)(ms - reg)^2 (p = P(fill), ms = take-off - schedule, reg = non-fill regression).
- Ranking file: expected fill-uncertainty MSE 16,010 (84% at LIRF; 9,890 from the 5,283 no-NM
  flights). The leaderboard total is 309.95^2 = 96,067.
- Calibration on the 2025 folds: the realised MSE contribution of the uncertain rows (e > 1e5) is
  about 3x the expected term (jan 23,280 vs 7,394; jul 127,754 vs 47,772), because those rows also
  carry regression and overnight-record errors.
- Reading:
  - The tail is roughly half our leaderboard MSE (3 x 16k = 48k of 96k). The non-tail half
    corresponds to about 220 s RMSE.
  - The top team's total MSE (220.4^2 = 48,576) is about our non-tail half. So the gap to the top is
    almost entirely the tail, and our ordinary-flight accuracy is probably competitive.
  - Top teams likely discriminate fills much better than AUC 0.88, or have found a rule that makes
    these rows predictable.
- Action: independent forensic search for the missing signal (`docs/forensics_fill.md`,
  running).


## EXP-033 NM matching window: |off-block - LOBT| <= 60 min (2026-09-30 night, independent forensic analysis + lead)

- Source: `docs/forensics_fill.md` (independent forensic search, independent analysis, no access to other
  teams' work). Lead verification: over all 2,062,623 NM-matched 2025 departures,
  |BLOCK - LOBT_flt| lies in [-60.1, +60.1] min, 0 exceptions, at every airport. It reads as the
  organisers' join rule.
- Consequences (label-free on ranking):
  - matched flights: target in [take-off - LOBT - 3600, take-off - LOBT + 3600];
  - a fill is impossible when |schedule - LOBT| > 60 min (66,665 such 2025 rows, 0 fills).
- Implementation (`NMWindow` in `src/prc/models/gbdt.py`, built into `TwoPartFill`): P(fill) = 0
  when the fill is impossible; predictions of matched flights clipped to the window.
- A first version that replaced out-of-window predictions by the window centre hurt clean (+2.8 s)
  and core (+2.0 s): the violators pushed back near the window edge. Clipping to the nearest edge,
  post hoc on `n-om-blend` (jan+jul): **all 381.9 -> 365.8, core 249.5 -> 249.3, clean 212.0 ->
  211.8**.
- Ranking: 48 v4 predictions lie outside their window, almost all the gain on one LIRF flight
  (ITY340, take-off one day after its LOBT day: window [83,214, 90,426] s, v4 predicted 11,718 s).
  The analyst's lower bound of the leaderboard gain: MSE -14,939 (309.95 -> at most 284.8 s) if the
  rule holds in 2026.
- Full-rule validation: `w-lgb` (running).

## EXP-034 24-hour date artefact (D24) as a third mixture component (2026-09-30 night)

- Mechanism (forensics section 2): for some no-NM departures that left on a later local date than
  scheduled, the off-block is stored with the schedule's local date and the real time of day:
  target = 86,400 s + taxi. 13 LIRF cases in 2025 plus 1 at LFPG and 1 at LSZH; the model never
  predicted such values.
- Implementation (`D24` in `gbdt.py`, opt-in `d24: true`): for no-NM departures crossing a local
  date, q = P(D24) by LIRF vs other and local take-off hour bucket (0-3, 3-10, 10-24), learned from
  the training rows with 2 pseudo-observations toward 0. Prediction = q (86,400 + reg) +
  (1 - q) two-part prediction.
- Rates on all 2025: LIRF 0.008 / 0.100 / 0.429 by bucket; other airports 0. Ranking: 52 candidate
  flights, 6 with q > 0.1.
- Derivation: D24 targets exceed 3 h, so core and clean cannot move; only all can. Rows are few and
  all-flight noise is large, so judge on the rows concerned (LIRF crossing flights) and accept only
  if they improve without any core/clean change. The rates rest on about 20 rows: treat the
  leaderboard expectation (analyst: about -4,500 MSE) as an order of magnitude.


## EXP-035 Recovered NM record rules for no-NM departures (2026-09-30 early morning, lead) -> rejected as implemented

- Feature `rec_to_minus_lobt`: take-off minus the LOBT of the NM record recovered from the arrival
  row at the destination (same flight number and destination, landing 0-14 h after take-off,
  earliest). Coverage LIRF no-NM: 115 of 1,487 train rows, 42 of 383 ranking rows. The analyst
  found 216 / 71 with a looser link.
- Rules reproduced on LIRF 2025: A (|schedule - LOBT| <= 60 min) 76 rows, 0 fills; B (take-off - 15
  min within 45 min of LOBT) 23 rows, 91% fills.
- Implementation error: the rules were applied at all airports (A: P(fill) = 0; B: P(fill) >= 0.9),
  but they were established at LIRF only.
- `w24r-lgb` vs `w24-lgb`, the 300 changed rows' squared error: jan 1.97e7 -> 1.44e7, jul 9.70e8 ->
  1.15e9, **q4 2.28e8 -> 6.32e9** (q4 core 212.15 -> 238.46).
- Decision: reject as implemented; candidate v7b (built in parallel) discarded, v7 stays the
  recommendation. To do: restrict to LIRF (or learn per airport), re-validate. Expected gain is
  small (analyst: about 600 MSE on the ranking file).
- Lesson: a rule found on one airport's records must not be generalised without a per-airport check.
- Follow-up, rules restricted to LIRF (`w24rl-lgb`): 53 rows change; their squared error: jan 1.02e7
  -> 4.98e6, jul 9.18e8 -> 1.11e9, q4 2.03e8 -> 1.19e8; in total 1.13e9 -> 1.23e9. jan+jul core
  bootstrap -1.12 [-4.54, +0.86]. Mixed, net worse: not adopted. v7c (built in parallel) discarded.
  The rules are clean on their 2025 sample, but rule B's 0.9 floor looks too aggressive out of
  sample (July). Possible later variant: feed `rec_to_minus_lobt` to the classifier as a feature
  instead of hard rules.


## EXP-036 Fresh review: where the leaderboard gap is, and the fill mixture's two defects (2026-09-30, lead, new session)

- Leaderboard reading (no probing; our four scored uploads only):
  - LB MSE changes tracked local **core** MSE changes: v1 -> v5 -14.3k on the board vs -14.5k core
    locally; v5 -> v6 -3.5k vs core -1.4k while local all moved +3.9k. Fit LB = a * core + c:
    a = 1.0-1.1, c = 26-37k (tail plus any 2026-specific error).
  - So our ranking core costs about 62k MSE and the rest about 30k. The top team's total is 48.2k
    (219.6 s on 2026-09-30). **The top teams are better on core, not only on the tail**; EXP-032's
    "ordinary accuracy is competitive" does not hold.
  - Our policy: every upload is justified by a logged local result beforehand, and no upload is
    designed to learn about specific rows.
- Decomposition of `n-om-cb` jan+jul (core): a perfect fill classifier gives core 225.8 vs 250.4
  (-11.7k MSE). P(fill) is overconfident in the mid range (0.34 -> 0.24 actual, 0.55 -> 0.43, 0.75
  -> 0.59) and underconfident on very late LIRF no-NM rows (true fills of 16,000-45,000 s predicted
  at p = 0.05-0.3).
- Ordinary-flight residuals are per-flight: removing leave-one-out airport-hour means explains 3.5%
  of NM non-fill MSE (1.4k of 40.5k). Better congestion state has little room left.
- January 2026 predictions are 100-180 s above January 2025 truth at EDDF, EDDM, EHAM, LSZH, LTFM;
  Open-Meteo shows January 2026 much colder there (frozen-precipitation hours x2-3, LSZH icing
  27% -> 61%). Consistent with de-icing, not a drift bug. Not actionable without labels.
- Ruled out: timezone-shifted fills (no spikes at schedule +-1/2 h), off-block = inbound in-block
  (< 0.1%), fills to LOBT/EOBT/IOBT (no excess beyond the schedule fill).
- Lab (`scripts/lab/fill_lab.py`, one LightGBM seed, final-lgb blocks + latent, jan+jul;
  all / core / clean):

  | regression target | classifier | all | core | clean |
  |---|---|---|---|---|
  | raw (current) | unweighted (current) | 362.8 | 267.3 | 217.3 |
  | clipped at 3 h | unweighted | 369.0 | **242.5** | 216.3 |
  | non-fill > 3 h dropped | unweighted | 371.5 | 241.7 | 216.5 |
  | clipped at 3 h | 63 leaves | 358.0 | 244.4 | 216.0 |
  | clipped at 3 h | stake-weighted (ms - nn_pred)^2 | 346.5 | 245.4 | 217.1 |

  - The raw regression is contaminated by the 30-odd non-fill 2025 rows at 58,000-88,000 s (D24,
    overnight records): on no-NM non-fill core rows its RMSE is 1,202 s vs 585 s clipped. LIRF no-NM
    core MSE 18.9k -> 7.8k, other no-NM 5.3k -> 4.1k.
  - The contamination acted as an accidental tail model (raw reg predicts 20,000-38,000 s for late
    LIRF no-NM rows), so the clip loses on the 60 rows > 3 h (+17k MSE locally), net about zero on
    local all. Since the board tracks core, the clip is worth about -12k MSE there by itself, if the
    tail is handled explicitly instead of by contamination.
- Next: full-year out-of-fold parts (six two-month holdouts, `scripts/lab/year_oof.py`) with the
  clipped regression, then a stake-dependent calibration of P(fill) fitted out of fold
  (`scripts/lab/calib.py`), judged on 12 months of tail rows instead of 60.


## EXP-038 Full-year out-of-fold parts; stake-weighted fill classifiers and calibration (2026-09-30, lead)

- Why: jan+jul hold 60 rows > 3 h, too few to judge tail handling. Six two-month holdouts over 2025
  (3-day buffers) give out-of-fold parts for every 2025 departure: 202 rows > 3 h, 14 D24.
  `scripts/lab/year_oof.py` (final-lgb blocks, final latent file, regression clipped at 3 h, one
  LightGBM seed), output `experiments/lab-year/parts.parquet`; scoring `scripts/lab/calib.py`.
- Classifiers: `base` (current), `wsq` (weights min((ms - nn_pred)^2, 86,400^2), 63 leaves, 500
  rounds), `wabs` (weights |ms - nn_pred|). Calibrator: weighted logistic regression on logit(p) with
  segment and log-stake interactions, fitted leave-one-block-out.
- Full year, with the D24 zone rule of EXP-039 (all / core / clean; tail = MSE of rows > 3 h):

  | P(fill) | all | core | clean | tail |
  |---|---|---|---|---|
  | base | 260.6 | 216.2 | 201.4 | 21.2k |
  | wsq | 254.0 | 217.9 | 201.9 | 17.1k |
  | **mean of base and wsq** | 254.3 | **216.0** | 201.3 | 18.0k |
  | wsq, calibrated (cap 4 h) | 252.2 | 217.2 | 201.5 | 16.4k |
  | fill oracle | 224.7 | 193.9 | 187.3 | 12.9k |

- Reading: stake weighting moves error from the tail to core; averaging the two classifiers keeps
  the tail gain at no core cost. The fitted calibrator adds 2 s on all but costs 1 s core; not
  adopted (one more fitted component for little). 22 s of core remain locked in fill uncertainty
  (oracle), unreachable with the fields we have so far.
- Decision: mean of the unweighted and stake-weighted classifiers (`clf_stake: true`).


## EXP-039 The 24-hour date artefact is a rule (2026-09-30, lead)

- Full-year decomposition (EXP-038, clipped regression): the 14 D24 rows carry 17.4k of the 24.1k
  tail MSE, more than the 170 fills > 3 h (4.2k). At ranking scale one missed D24 row costs about
  7,500 MSE.
- Mechanism: the recorded off-block keeps the real local time of day and takes the calendar date
  nearest to the schedule. LIRF no-NM non-fills whose real off-block falls on a later local date
  than the schedule (real off-block of D24 rows = recorded + 24 h):
  - |scheduled time of day - off-block time of day| < 12 h: **11 of 11 are D24**;
  - larger gaps (night crossings): 23 normal, 1 D24 (ITY332);
  - matched LIRF flights (496 rows) and the other airports (984 rows in the same geometry): 0 D24.
  So in the zone the only uncertainty is fill versus D24; the hour-bucket table of EXP-034 mixed
  zone and non-zone rows.
- Rule (`D24Zone` in `src/prc/models/gbdt.py`, test `tests/test_d24_zone.py`): off-block estimated as
  take-off - clip(reg, 300, 1,500) s; in the zone, prediction = p ms + (1 - p)(86,400 + reg);
  LIRF no-NM later-date rows outside it get a 1/25 D24 residual.
- Full year: 25 zone rows (11 D24, 14 fills, nothing else). Zone rows' MSE 8.3k -> 3.5k (base
  classifier), all 266.2 -> 260.6, core unchanged. jan+jul with the clipped regression and the mean
  classifier: all 334.6 -> 308.9.
- Caveat: the rule was found on the same 2025 rows it is scored on. It has one free parameter (12 h),
  set by the mechanism rather than tuned, and it held 11/11 with no counter-example; but treat the
  size of its measured gain as optimistic.
- Ranking: 8 zone rows (1 in January, 7 in July), v7 predicted 58,000-120,000 s for them.


## EXP-037 Physics lab: target encodings and inbound rotation for the non-fill regression (2026-09-30, lead)

- Question: after EXP-036 showed ordinary-flight residuals are per flight, do recurring identities
  (stand x runway, flight number, airline x runway x hour, type x runway, destination) or the inbound
  aircraft's turnaround add information?
- Features (`scripts/lab/phys_feats.py`): out-of-fold target encodings (non-fill rows, target
  clipped at 3,600 s, smoothing 20, day-grouped 5-fold on training rows) and label-free rotation
  (latest arrival at the same stand, same airline prefix, in-block before take-off - 10 min;
  stand-only fallback: schedule slack, take-off minus inbound in-block, inbound lateness and taxi-in).
- One LightGBM seed, regression clipped at 3 h, jan+jul (`scripts/lab/phys_lab.py`); "ordinary" =
  NM-matched non-fill rows <= 2 h:

  | arm | ordinary | core | clean |
  |---|---|---|---|
  | reference | 205.12 | 242.58 | 216.34 |
  | + target encodings | 205.12 | 242.70 | 216.33 |
  | + rotation | 204.85 | 241.82 | 216.09 |
  | + both | 204.58 | 241.53 | 215.88 |

- Reading: target encodings add nothing (the trees and the latent already carry the identities);
  rotation about 0.3-0.5 s, at the noise floor. The regression is near saturation for this feature
  family. Not adopted now; rotation is a cheap label-free candidate to re-test with seeds if time
  allows.


## EXP-040 Candidate v8: clipped regression + stake-weighted fill + D24 rule, LightGBM bag + CatBoost (2026-09-30, lead)

- Configs `configs/v8-{lgb,cb}.yaml`: final-lgb blocks and latent, `yclip: 10800` (EXP-036),
  `clf_stake: true` (EXP-038), `d24: zone` (EXP-039), NM window as before. LightGBM 3-seed bag;
  CatBoost regression (classifiers LightGBM). Blend 50/50, not tuned (0.7 CatBoost would give core
  237.1 vs 237.6: not worth tuning on the evaluation folds).
- VAL-MONTH-V2 jan+jul (all / core / clean):

  | model | all | core | clean |
  |---|---|---|---|
  | v8 LightGBM bag | 308.41 | 241.92 | 215.04 |
  | v8 CatBoost | 304.58 | 237.70 | 212.05 |
  | **v8 blend** | **304.71** | **237.56** | **211.42** |
  | v7 blend (approx.) | 343.43 | 249.53 | 211.74 |
  | v6 blend (`n-om-blend`, 309.95 on the board) | 381.90 | 249.49 | 211.98 |

- Paired day bootstrap, v8 blend vs v7 (old minus new, > 0 = v8 better): core +11.7 s [6.4, 19.7],
  all +37.8 [11.0, 77.1], clean +0.32 [0.01, 0.62]; vs v6: core +11.6 [6.6, 18.4], clean +0.56 [0.21,
  0.92]. LightGBM half vs v7's LightGBM half (`w24-lgb`): core +23.4 [13.1, 39.3].
- Leaderboard expectation (v7 = 303.32, MSE 92.0k): core -5.8k MSE (the board has tracked core
  one-for-one); tail -19k locally on jan+jul, scaled to the ranking's lighter tail (x 0.55 from the
  v1-v7 fits) about -10k. About 76k MSE, **~275 s (plausible 265-290)**, rank about 49 of 207 on
  2026-09-30 (265 -> 34, 285 -> 69). The D24 part is lumpy: 8 ranking zone rows.


## EXP-041 LIRF no-NM departures more than 6 h late are fills; zone fill rate falls with lateness (2026-09-30, lead)

- Finding (full-year out-of-fold parts, EXP-038): LIRF no-NM departures more than 6 h late, outside
  the D24 zone, are fills in 37 of 38 cases (every two-month block: 6/6, 2/2, 5/5, 14/15, 6/6, 4/4);
  the one exception is an ordinary 1,078 s taxi. The classifier gave them 0.41-0.90 on average per
  block. At the other airports no-NM late flights are ordinary (fill rate 0).
- Mechanism (consistent with EXP-033): a delay of 6 h or more is always filed, so NM's LOBT follows
  the real off-block; a recorded real off-block would then match NM. A no-NM record that late
  therefore carries a substitute off-block (the schedule), or a D24 date.
- In the D24 zone the fill share falls with lateness: < 14 h 7/7 fills, 14-18 h 7/12, >= 18 h 1/7
  (later departures are recorded as D24 rather than filled).
- Rule (`lirf_late: true`, in `D24Zone`): late rows get p = max(p, rate), rate learned on training
  rows ((fills + 0.5)/(n + 1), 0.962 on all 2025); zone rows get the binned rate ((fills + 1)/(n + 2),
  0.89 / 0.54 / 0.22), replacing the classifier there.
- Full year, rates fitted leave-one-block-out, v8 parts (all / core; MSE of the rows concerned):
  all 253.10 -> 249.60, core 216.14 -> 216.51 (the one ordinary exception); late rows 1,884 -> 241;
  zone rows 2,169 -> 2,057.
- Ranking: 18 late rows; v8 already gives 15 of them 0.88-1.0, three get 0.23 / 0.41 / 0.69
  (Volotea and Ryanair, 9-11 h late). Expected gain about 3,400 MSE if they are fills at the 2025
  rate, about 7 s on the board; the downside if one is ordinary is about 3,000 MSE.
- Caveat: found by looking at the 2025 rows; the rates are fitted out of block, but the rule itself
  is in-sample. The mechanism, and the per-block stability, are the reasons to trust it.
- Leaderboard follow-up (2026-09-30): v9 = v8 + this rule scored 262.11 vs 261.81 (+154 MSE, 22 rows
  changed, possible swing +-5.5k). No gain on the board. Kept in the code (the full-year evidence is
  unchanged) but the next candidates should be judged on core, where many rows carry the gain.


## EXP-042 Regression hyper-parameters: feature subsampling and Huber loss (2026-09-30, lead)

- Setup: `scripts/lab/phys_lab.py`, one LightGBM seed, regression clipped at 3 h, final-lgb blocks and
  latent, jan+jul; "ordinary" = NM-matched non-fill rows <= 2 h; core and clean use a fixed fill
  probability (lab EXP-036 `p_base`) so arms differ only by the regression.

  | arm (vs 255 leaves, lr 0.05, 1,500 rounds, ff 0.8, L2) | ordinary | core | clean |
  |---|---|---|---|
  | reference | 205.12 | 242.58 | 216.34 |
  | 127 leaves / 511 leaves | 205.04 / 206.46 | 242.11 / 243.94 | 216.26 / 217.61 |
  | lr 0.025, 3,000 rounds | 204.35 | 241.95 | 215.63 |
  | feature_fraction 0.5 / 0.3 | 203.12 / 202.58 | 240.98 / 240.40 | 214.61 / 214.18 |
  | Huber alpha 900 (L2 elsewhere) | 203.99 | 240.56 | 215.39 |
  | ff 0.5 + Huber 900 (600 / 1,500) | 202.31 (202.39 / 202.81) | 239.23 (239.81 / 239.58) | 213.88 (214.05 / 214.34) |
  | **ff 0.5 + Huber 900 + lr 0.025** | **201.80** | **238.89** | **213.51** |

- Reading: the regression was over-fitting correlated features (subsampling helps) and chasing
  heavy-tailed residuals (the top 1% of ordinary rows carries 30% of their squared error, EXP-036);
  Huber with a 15-min transition trades a little bias for much less variance. Gains are spread over
  every airport, the kind the board tracks.
- Next: v10 = v9 with these regression settings in the LightGBM bag (`configs/v10-lgb.yaml`) and the
  CatBoost analogue (Huber delta 900, rsm 0.5, `configs/v10-cb.yaml`), full protocol, then final.
- v10 protocol (2026-09-30 evening), jan+jul all / core / clean:
  - `v10-lgb` (3-seed bag, ff 0.5 + Huber 900 + lr 0.025, 3,000 rounds): 302.78 / 238.63 / 212.68 vs
    `v8-lgb` 308.41 / 241.92 / 215.04; day bootstrap clean +2.36 [2.03, 2.72], core +3.25 [2.37, 4.44].
  - `v10-cb` (Huber delta 900, rsm 0.5): 302.09 / 237.63 / 212.32 vs `v8-cb` 304.58 / 237.70 / 212.05:
    neutral for CatBoost (symmetric trees already regularise; kept for its speed).
  - **v10 blend 50/50: 301.17 / 236.55 / 210.97** vs v8 304.71 / 237.56 / 211.42; clean +0.45 [0.17,
    0.71], core +0.99 [0.31, 1.88], all +3.5 [-0.6, 8.6]. The LightGBM gain is diluted: CatBoost
    carries most of the core accuracy. Weights 0.4-0.6 within 0.2 s: 50/50 kept.


## EXP-043 CatBoost regression settings (2026-09-30 night, lead)

- `scripts/lab/phys_lab.py` CatBoost arms, one seed, regression clipped at 3 h, fixed lab fill
  probability, jan+jul (ordinary / core / clean):
  - reference (depth 8, lr 0.08, 3,000 it.): 200.51 / 236.89 / 212.04 (LightGBM reference 205.12 /
    242.58 / 216.34; tuned LightGBM, EXP-042, 201.80 / 238.89 / 213.51);
  - depth 10: 200.19 / 236.80 / 211.83 (1.5x time);
  - lr 0.04, 6,000 it.: 200.11 / 236.49 / 211.70 (2x time);
  - l2_leaf_reg 10: 200.55 / 237.21 / 212.20.
- Reading: CatBoost is near its optimum for these features; the best arm gains 0.4 s core at twice
  the cost, below what the board can resolve (v10 lesson). Not adopted. CatBoost remains the stronger
  regression by about 1.3 s ordinary over the tuned LightGBM.


## EXP-044 Would strike calendars help? (2026-10-01, delegated check)

- Question: do we need data on social events such as strikes? Check on the full-year
  2025 out-of-fold parts (EXP-038) against verified public strike calendars (Italy, France, Spain,
  UK, Germany, Netherlands, Switzerland, Turkey). Full report not published.
- Ceiling of any day-level signal: removing leave-one-out airport-day mean residuals explains 0.94% of
  ordinary-flight MSE (-0.62 s RMSE); airport-hour 2.7%.
- Strike days (196 airport-days held in 2025) carry error in proportion to their flights (4.2% of
  between-day error for 4.9% of flights), are not over-represented among the anomalous days (1 of the
  top 25, 1.3 expected), nor among the LIRF fill-excess days (1 of the top 20, 1.0 expected).
  Called-off strikes (placebo) show a larger apparent effect than held ones.
- A strike flag fitted out of sample makes the model slightly worse; in-sample the most it could buy is
  about +28 s^2 MSE, about 0.05 s RMSE; on the ranking months (5 weak-to-strong LIRF strike days,
  0.64% of departures) about 0.02 s. Strikes that matter already appear as traffic collapses in the
  movement file.
- Licensing: no openly licensed, dated strike dataset covers 2025-2026 (Italian ministry calendar has
  no stated licence on its subdomain; its CC-BY open-data CSV stops in 2020).
- Decision: do not add strike data. Side observation: mean P(fill) runs low at every airport in the
  full-year parts (LIRF 0.158 vs 0.181 actual); a per-airport recalibration improves all but costs
  core, consistent with EXP-038; not pursued.


## EXP-045 Latent v2: airline token and Huber loss in the window network (2026-10-01, derivation before the run)

- Motivation: the latent network is the only component that ever moved ordinary flights a lot
  (EXP-010, -13 s clean). It lacks the carrier for no-NM flights (NM's operator is anonymised and empty
  without a match; the trees got `airline` in EXP-028), and its mean head uses squared error while
  Huber helped the trees by about 3 s core on ordinary flights (EXP-042).
- Change (opt-in, old checkpoints unaffected): `--airline` adds the carrier from the flight number as
  an eighth token field (399 carriers, 1.2% unknown in January 2025); `--huber 900` uses a Huber loss
  (delta 900 s) for the mean head. Departures only, weather tokens, 3 seeds (20260926, 1, 2), 3 day-group
  models each, lr 1e-3, 16 epochs, patience 2: the adopted EXP-025 setting otherwise. Seed average
  `latent_set_v2_avg3`; stacks `configs/v11-{lgb,cb}.yaml` = v10 with the latent swapped.
- Derivation:
  - If the changes help: the network alone (clean) improves beyond its 3-seed spread (about 1 s) on
    jan and jul; the stack improves core by more than 1.5 s and clean by more than 0.5 s, with the gain
    larger on no-NM and LIRF rows (airline) and on ordinary rows generally (Huber).
  - If not: network alone within its seed spread, stack within noise (clean 0.5 s, core 1.5 s).
  - Both changes ship together for time reasons; a gain cannot be split between them without a
    second run.
- Adoption rule: upload v11 only if jan+jul core improves by at least 2 s over v10 (the v9/v10 lesson).
  (Correction 2026-10-02: 2 s of core at 237 s is about 0.95k MSE, not 2.5k as first written.)
- Run: `scripts/night_v11.sh`, 2026-10-02 02:00-07:00 (night run), every step under a deadline.
- Results (night of 2026-10-01/02, 23:00-01:33, all steps clean):
  - Network alone, clean non-fill jan+jul: 209.12 vs 208.86 for `noarr_wx_avg3`. No change: Huber
    and the airline token do nothing for ordinary flights inside the network.
  - Stack: `v11-lgb` 302.85 / 237.61 / 212.93 (v10-lgb 302.78 / 238.63 / 212.68); `v11-cb` 302.12 /
    236.63 / 212.08 (v10-cb 302.09 / 237.63 / 212.32); **v11 blend 301.25 / 235.55 / 211.00** (v10
    301.17 / 236.55 / 210.97). Day bootstrap vs v10: core +1.03 [-0.64, +2.80], clean -0.02 [-0.52,
    +0.46]. The core gain sits on no-NM rows (RMSE 863 -> 845) and LIRF (477 -> 470), as expected from
    the airline token; ordinary NM rows unchanged (201.2 -> 201.4).
  - Verdict: the null outcome of the derivation; rule not met, v11 not uploaded.
- Follow-up: **the average of the v10 and v11 stacks** (two latents, 6 seeds, 4 GBMs): 300.44 / 235.16
  / 210.34; vs v10 clean +0.63 [0.38, 0.88], core +1.36 [0.40, 2.47]; vs v8 (board best) clean +1.08
  [0.75, 1.41], core +2.37 [1.47, 3.42], about -1.1k MSE: real but below what the board resolved in
  v9/v10. Kept as the pooled candidate (`candidate_v10v11.parquet`), not uploaded. A third stack on
  three fresh latent seeds (v12, `scripts/night_v12.sh`, 02:18-07:00) tests whether a third member
  adds more.
- Third stack (v12, latent v2 seeds 3/4/5, run 02:18-04:47): `v12-lgb` 303.24 / 238.69 / 212.70,
  `v12-cb` 302.57 / 237.58 / 212.23, blend 301.68 / 236.59 / 210.95. **Average of the three stacks
  (v10+v11+v12): 300.28 / 234.99 / 210.07**; vs v10+v11 clean +0.27 [0.14, 0.40], core +0.17 [-0.17,
  0.47]; vs v8 clean +1.34 [1.02, 1.67], core +2.57 [1.65, 3.56]. Diminishing returns: a fourth stack
  would add about 0.1 s. This ensemble is the freeze candidate (upload quick-quicksand_v9).


## EXP-046 Ceiling of neighbour reconstruction (2026-10-04, derivation written before the run)

- Context: the goal is to push the latent architecture further. The latent's job is to reconstruct
  what the hidden off-blocks of neighbouring departures imply for each flight. Before designing new
  architectures, measure the headroom: give the trees the TRUE off-blocks of all other departures
  (never the flight's own) and see how far ordinary-flight error falls.
- Oracle features (`scripts/lab/oracle_ceiling.py`): aircraft still taxiing at my take-off (all and
  same runway), pushbacks 5-30 min before my take-off, queue ahead (pushed back >= 10 min before my
  take-off but took off after me), neighbours' true taxi (previous five on my runway, all within
  +-10 min, the take-offs just before and after mine). One LightGBM, regression clipped at 3 h,
  ff 0.5 + Huber 900, jan+jul; metric = ordinary rows (NM-matched non-fill <= 2 h).
- Arms: ref (blocks + current latent), +oracle, oracle without latent, blocks only.
- Derivation:
  - Large headroom (ref -> +oracle below -10 s): the current latent captures only part of what
    neighbours carry; a better joint reconstruction (iterative estimation, queue-structured
    attention) has room. Push the architecture.
  - Small headroom (< 3 s): the latent already extracts most of it; the remaining error is each
    flight's own unobserved factors (start-up, route, holding, de-icing). A new architecture over
    neighbours cannot gain much; effort should go to flight-level signals or stop.
  - (blocks -> oracle_nolat) minus (blocks -> ref) = share of the neighbour ceiling the latent
    already captures.
- Results (ordinary rows, jan+jul, one LightGBM seed):

  | arm | RMSE |
  |---|---|
  | blocks only | 217.25 |
  | blocks + current latent (ref) | 202.33 |
  | blocks + oracle, no latent | 193.46 |
  | ref + full oracle | **189.48** |
  | ref + oracle neighbour taxi only | 194.44 |
  | ref + oracle counts / queue only | 192.10 |
  | ref + the same features computed from the latent's own estimates (nn_pred) | 201.96 |

- Reading:
  - The latent captures 63% of the neighbour ceiling (14.9 of 23.8 s); perfect neighbour knowledge
    would still give -12.9 s on top of it (about -4.7k MSE at board scale). Headroom is real.
  - The most valuable hidden quantity is the true surface state at my take-off (aircraft taxiing,
    pushbacks just before, queue ahead): -10.2 s alone.
  - Rebuilding those quantities explicitly from the latent's own per-flight estimates adds -0.4 s:
    the network already uses what its estimates imply. The gap is estimation quality, not missing
    structure, so explicit queue features or iterative two-stage passes on current estimates will not
    help (consistent with EXP-012/013).
- Decision: push the architecture where the ceiling says, by teaching the network the surface state
  directly: auxiliary heads predicting the oracle counts and neighbour taxi (computable from labels on
  training days only, never at inference). EXP-047.


## EXP-047 Surface-state auxiliary heads in the window network (2026-10-04 night, derivation before the runs)

- From EXP-046: the true surface state at each take-off is worth -10 to -13 s on ordinary flights on
  top of the latent, and rebuilding it from the network's estimates adds nothing. Hypothesis: a
  network trained to predict the surface state (training-only targets) builds a better internal
  picture of the queue and estimates every flight better.
- Implementation (`src/prc/latent_set/aux.py`, flag `--aux-weight`): 8 standardised targets per
  departure from other departures' labels (aircraft taxiing at my take-off, all and same runway;
  pushbacks in the previous 10 / 30 min; queue ahead; neighbours' taxi within +-10 min; previous and
  next take-off on my runway within 1 h). Head on the 16-d bottleneck z (which the trees receive).
  Valid only where every departure within +-3 h is labelled, and in training only for departures
  whose +-3 h span lies in the model's fitting days. Cross-checked against the EXP-046 oracle code
  (corr 1.000 / 0.9997). Base setting = latent v2 (EXP-045: departures, weather, airline, Huber).
- Plan: (A) screen aux weights 0.2 and 1.0 with one seed on jan+jul, network alone, against
  `latent_set_v2_s20260926` (same seed, no aux); (B) best weight with 3 seeds (20260926, 1, 2, the v11
  seeds: a paired comparison) + final, stacked as v13 = v11 configs with the latent swapped.
- Derivation:
  - If the hypothesis holds: network-alone ordinary RMSE improves by more than 1 s (single-seed noise
    about 1 s) on both folds; the stack's core improves by more than 1.5 s and clean by more than
    0.5 s vs v11; the gain is larger in busy periods (queue).
  - If not: network alone within noise (the head learns the targets but z keeps the same information
    for the main task), stack within noise. A strong weight may even hurt the main head.
- Adoption: a v13 stack that beats v11 beyond noise joins the ensemble; an upload only if the new
  ensemble beats quick-quicksand_v9's local core by >= 2 s.
- Results (2026-10-04 night): network alone, ordinary rows jan+jul, same seed: no aux 211.49, aux 0.2
  211.58, aux 1.0 211.73; busy periods unchanged (214.50 / 214.36 / 214.57). **Null**: teaching the
  network the surface state adds no information; the EXP-046 gap is the value of unobserved
  information, not a learning problem. The literature review
  agrees: published queue-feature gains all rely on knowing other aircraft's pushback times, and on a
  saturated runway the take-off sequence carries little about queue depth. Architecture line closed.


## EXP-048 Arrivals at adjacent stands as pushback evidence (2026-10-04, lead)

- Idea: a pushback blocks the taxilane behind neighbouring stands; arrivals heading there wait (excess
  taxi-in, observed in ranking). Features (`scripts/lab/adjacent_stands.py`): arrivals within 120 m
  (OSM stand coordinates) in-blocking in the 45 min before my take-off (count, max excess, timing,
  count waiting > 3 min) and my own stand's next arrival (excess, in-block minus my take-off).
- One LightGBM, blocks + latent, jan+jul ordinary rows: 202.41 -> 201.76 (-0.65 s), mostly from my own
  stand's next arrival; adjacent-stand features carry 0.1-0.7% of gain. Diffuse and small (about
  -250 MSE): not adopted (v9/v10 lesson).


## EXP-049 Targeted record rules (2026-10-04 night; independent review + lead verification)

- Source: an independent fresh-eyes review proposed six ideas; each was re-verified here
  on our own out-of-fold predictions. Rejected after verification: T - recovered AOBT_3 for no-NM
  departures (RMSE 715 s vs 459 s for our regression on 792 non-LIRF rows: 88% within 10 min but a
  heavy tail), adjacent stands (EXP-048).
- Adopted (`src/prc/models/postrules.py`, applied after the ensemble):
  - `lobt_floor`: matched departures with take-off - LOBT > 2 h: pred = max(pred, T - LOBT - 1,200).
    Full-year OOF: 111-143 rows, -188 MSE, 6/6 blocks; from 1.5 h it hurts (937 rows, 3/6 blocks).
  - `twin`: duplicate records at LFPG/EDDM (same flight, schedule, take-off within 60 s; one
    matched): matched twin = T - LOBT - 300. 2025: 90 twins, RMSE 2,451 -> 1,116 s; -206 MSE, 6/6.
  - `winter_nonm`: no-NM non-LIRF departures on airport-days with > 6% no-NM, frozen precipitation
    and minimum temperature < 3 C: pred x 1.29. Leave-one-day-out on the 6 frozen 2025 days: squared
    error -37%; non-frozen disruption days: +1% (no effect). 2025 no-NM vs matched mean taxi on those
    days: 2,007 vs 1,054 s (EHAM 01-05), 2,661 vs 1,811 (EDDM 12-24), about 2.15x at LTFM; exception
    EDDF 01-05 (0.84x). In the ranking our ensemble predicts no-NM BELOW matched on the EHAM storm
    days 01-04/05: the uplift goes in the 2025 direction. Weakest rule locally (-77 MSE full year).
- jan+jul three-stack ensemble: bundle -583 MSE all, -463 core (core 234.99 -> 234.01).
- Ranking: 945 rows change (857 winter, 93 floor, 15 twins); movement cost sum d^2/N = 1,769;
  expected net about -1.5k MSE if the 2025 ratios hold, about +1.8k in the worst case (winter ratio
  1.0 on all rows).

- Leaderboard (2026-10-04 22:00): the bundle on quick-quicksand_v9 scored 261.68 vs 260.77 (+478 MSE,
  movement cost 1,769). Not adopted. Lesson: a factor fitted on six 2025 winter days does not
  transfer to a storm four times beyond 2025; rules whose rows are dominated by out-of-range days
  carry most of the movement and the least evidence.


## EXP-050 LIRF forensics, second pass (2026-10-04/05 night, independent analysis + lead verification)

- Dedicated analysis of the Rome records (internal notes). Its first
  analysis (a year-wide join on destination country, a cross product) reached 49 GB and triggered the
  OOM that killed the desktop app at 22:03; rerun under a 10 GB cap.
- Findings, by value:
  1. **Turkish Airlines regime switch** (verified by the lead): from 2025-10-17 its LIRF departures are
     376/378 fills (the two others follow the 600 s early rule); its arrivals' substitute signature
     (in-block = schedule or landing + 10 min) rises from 15-23% (Jan-Sep 2025) to 77-79% (Nov-Dec);
     January 2026: 62% (July 2026: 25%, near baseline). Label-free in the ranking. The ensemble gives
     these January departures an implied P(fill) of about 0.3. Rule: P(fill) = 0.9 for THY LIRF
     January 2026 departures (154 rows), fill value = take-off - schedule (or 600 s if early).
     Expected about -250 MSE.
  2. A third substitute regime: some LIRF off-blocks shifted by the local UTC offset (-1 h winter,
     -2 h summer), always no-NM; 30 of 96 no-NM departures that took off 0-60 min after schedule in
     2025. About 30 ranking rows, about 140 MSE, medium confidence. Not implemented.
  3. Smaller signals for late no-NM LIRF departures (destination congestion, return leg) and AEZ
     arrival-missing rates: 100-600 MSE together, uncertain. Not implemented.
  - Agent's verdict, consistent with the lead's: the remaining fill gap is mostly per-flight recording
    failures and outages, not predictable from these fields.


## EXP-051 Fourth ensemble member: latent with arrival tokens (2026-10-05 night, lead)

- Latent v3full = departures + arrivals + weather + airline + Huber, seeds 6/7/8, run as a capped
  systemd service (peak 17.3 GB). Network alone, ordinary rows jan+jul: **207.94** (noarr_wx 208.86,
  v2 209.12, v2b 208.98): the best network so far, better on both months and on busy and quiet periods.
- Stack v15: LightGBM 302.33 / 238.09 / 212.45, CatBoost 301.24 / 236.27 / 211.65 (best CatBoost half),
  blend 300.60 / 235.69 / 210.61.
- Four-stack average (v10+v11+v12+v15): 299.83 / 234.59 / 209.76; vs three stacks clean +0.31 [0.22,
  0.42], core +0.41 [0.12, 0.69]. Adopted as the freeze candidate with the THY rule (v16).


## EXP-052 ADS-B surface tracks (adsb.lol) as a corrector on the ensemble (2026-10-05/06, lead)

- Source: adsb.lol daily archives (ODbL / CC0), January and July 2025 and 2026 (124 days, 360 GB on
  the external drive). Allowed by the organisers (forum answer, 2026-10-01). `scripts/adsb/`: extraction of
  points within 8 km of the 10 airports below 3,000 ft (`extract_day.py`), take-off detection and
  one-to-one matching on take-off time + callsign + type (`features.py`), stand positions from arrival
  tracks (`stands.py`: 1,700 stands, median spread 3-4 m; few at LIRF and LFPG).
- Features: time since first ground point of the final session, stops, first point position and
  speed, distance of the first point to the departure's own stand, time since leaving the stand
  (when seen there: EHAM 92% within 60 s of the true off-block, but only 25% of EHAM departures),
  surface congestion counts. Built from take-off time, airport, callsign, type, stand; never the
  off-block.
- Coverage (matched departures) 2025 -> 2026: EDDF 73 -> 58%, EDDM 57 -> 97%, EGLL 12 -> 71%, EHAM
  98 -> 98%, LEBL 81 -> 91%, LEMD 1 -> 73%, LFPG 6 -> 15%, LIRF 46 -> 26%, LSZH 71 -> 86%, LTFM 0.
- Corrector: LightGBM (Huber 900, 15 leaves, 500 x 0.02) on target (clip 3 h) - ens4 OOF base,
  trained and applied on matched departures only. VAL-MONTH-V2 primary folds on 58 of 62 days:
  base 301.70 / 234.44 / 210.03 -> 298.01 / 229.63 / 204.86 (all / core / clean); vs the same
  corrector without ADS-B, clean +5.3 [4.8, 5.8]. Clean by airport: EDDF 168 -> 155, EDDM 158 -> 140,
  EHAM 158 -> 137, LEBL 210 -> 200, LSZH 166 -> 152; stable in both folds. LIRF worse (jul +3.1).
- Transfer: generic corrector (no airport code, no coordinates) trained without the test airport
  gives almost the same gains (EDDF 156.9, EDDM 140.5, EHAM 138.5, LEBL 201.9, LSZH 156.3). Generic
  without congestion counts (receiver-dependent, drift-prone) equals the airport-specific one
  (non-LIRF core 195.40 vs 195.46). Gains hold in every bin of track depth (seen / base) outside
  LIRF, and the mean correction tracks the mean residual in each bin (calibrated); LIRF is the
  exception (base residual structure of fills and no-NM).
- Adopted: generic corrector without congestion counts, 3 seeds, applied to matched 2026 departures
  outside LIRF (`apply_ranking.py`). Risk: 2026 tracks are deeper at EGLL/LEMD/EDDM and shallower at
  EDDF than in 2025; depth bins argue the corrector stays calibrated.
- Board: quick-quicksand_v12 = 250.55 (v11 260.02): -4.8k MSE vs -2.4k expected; the 2026 coverage
  (EGLL, LEMD, EDDM) is higher than 2025's, so more rows are corrected.

## EXP-053 ADS-B corrector v2 and rejected extensions (2026-10-06 night, lead)

- Track-shape features (path length, straight distance from the first ground point to lift-off,
  mean ground speed, time since first reaching 10 kt, time since the end of the last stop >= 20 s):
  clean +0.25 [0.15, 0.33]. Capacity 31 leaves / 1000 rounds: +0.6 (63 and 127 leaves worse).
  Huber 150 instead of 900: +0.2. Together: 295.08 / 228.48 / 203.06 (v12 corrector 295.84 / 229.46 /
  204.11). Board: quick-quicksand_v13 = 249.60 (-475 MSE, as expected).
- CatBoost corrector (depth 7, Huber 150) averaged with the LightGBM one: clean +0.13 [0.06, 0.20],
  core +0.09. Kept for the freeze bundle (too small alone).
- Rejected: neighbour queue features from other departures' tracks (+0.03 s on matched rows; a
  corrector for unmatched departures from neighbours alone is worse, -0.13 s); NM timing offsets
  (LOBT, AOBT_3, EOBT, IOBT) in the corrector (-0.1 s); ground detection by speed < 40 kt (no extra
  matches: unmatched departures are first seen airborne, the receivers miss their ground phase).
- Remaining 2025 error (jan+jul, after the corrector): three LFPG tail rows 34% (2025-specific), LIRF
  matched rows 14% (fill uncertainty, shallow tracks), LTFM 8% (no coverage), unmatched EGLL/LFPG 15%.
- Determinism: match-score ties, row order and LightGBM threads made rebuilds differ by rms 7 s;
  fixed (sorted tie-breaks, `deterministic=True`). Two rebuilds are now byte-identical; the uploaded
  v13 is within that noise of the deterministic rebuild. The freeze upload will be built by the
  deterministic pipeline.
- Also rejected (2026-10-06 night): (1) training the corrector on Q4 2025 days: one test day
  (2025-11-12) shows coverage still 2025-like at EGLL (5%; 71% in 2026) and LFPG (19%), already
  2026-like only at LEMD (91%): estimated gain about 150 MSE for 3 h of downloads and a mixed base, not
  pursued; test files removed. (2) the corrector at LIRF restricted to NM-matched flights: worse in
  July 2025 (MSE 174.6k -> 177.0k on 11,620 rows; no LIRF tracks matched in January 2025), deep tracks
  neutral. LIRF's error is the record (fills), not the taxi: LIRF stays excluded.

## EXP-054 Arrival recording-regime features (2026-10-06, lead)

- At LIRF, across airline-months of 2025, the share of an airline's arrivals with substitute in-block
  times (in-block = schedule, or landing + 10 min) correlates 0.79 with its departures' fill share
  (month-to-month changes within an airline 0.41); about 0 at the nine other airports. Features
  (`scripts/regime/build.py`, label-free): same airline, airport and calendar month arrival substitute
  rate and count, airport-month rate, LIRF interaction.
- One member (v15-lgb): core 238.09 -> 236.27, both months better. Eight members: ensemble 234.59 ->
  233.16 core, LIRF all 527.7 -> 523.5; with the ADS-B corrector 228.42 -> 226.94.
- Board: quick-quicksand_v16 (with the calibrations) = 246.73 (v15 247.98).

## EXP-055 / EXP-056 Calibration to EUROCONTROL's published aggregates (2026-10-06, lead)

- EXP-055: monthly ATXOT taxi-out means (organiser-approved), implied 2026 mean = published 2026 + 2025
  gap; reliable only where the gap transfers between months within 7 s (EGLL, LEBL, EHAM, LIRF, LEMD).
  Board: quick-quicksand_v14 = 249.10 (v13 249.60).
- EXP-056: daily DLY_ALL_PRE_2 (team decision; equals the daily sum of max(0, off-block - schedule),
  corr 0.9997 in 2025). Per airport-day shift solved to reproduce it, half applied. 2025: core 228.48 ->
  227.57, both months and all airports better; uncertainty-aware variant (+0.07 s) not adopted.
  Board: quick-quicksand_v15 = 247.98.
- LTFM and LFPG forensics (independent analyses, internal notes): no record-level lever at either airport.

## EXP-056b/c Fill-aware daily calibration at LIRF (2026-10-07 night, lead)

- A schedule-filled record contributes 0 to the published daily sum of max(0, off-block - schedule); a
  real late off-block contributes T - S - taxi. At LIRF, per day, a common logit shift of the fill weights
  is solved to match the published sum (weights inferred from the prediction against the latent's
  non-fill estimate), applied at 0.75. 2025: LIRF all 522.5 -> 513.4 (uniform shift as reference), both
  months better; overall all 293.75 -> 292.51, core 226.04 -> 225.26. Worse at every other airport
  (their fills are not a substitute practice), so LIRF only. Board: quick-quicksand_v17 = 244.38 (v16
  246.73), about twice the local estimate.
- EXP-057 rejected: per-flight inbound substitute flag (same airline, same stand, 8 h): one member core
  236.27 -> 236.35; the monthly airline rate already carries it.
- EXP-056c rejected: exact fill probabilities from the models instead of inferred weights. CatBoost
  parts before the D24 logic: January 344 -> 320 but July 609 -> 667; LightGBM parts after the D24 logic:
  LIRF all 521.7 (lam 0.5) vs 513.2 for the inferred weights. `PRC_SAVE_PARTS` kept in run.py and
  make_submission.py for later use.

## EXP-058 February / June 2025 generalisation check (final-phase months, 2026-10-19)

- Whole-month holdouts never used for any tuning (`scripts/run_febjun.sh`, launched as a detached job): latents re-trained
  per fold (12 seeds), the eight regime-feature models (configs v1xr-*-fj), then the daily calibration to
  EUROCONTROL's published 2025 pre-departure delay (uniform half shift, fill-aware at LIRF). The monthly
  calibration and the ADS-B corrector are not scored (in-sample on 2025; no tracks for these months).
- Members, core feb / jun: LightGBM 225.3-225.5 / 219.9-220.1, CatBoost 229.2-232.9 / 219.6-220.2.
- Ensemble: all / core / clean 266.93 / 220.90 / 201.72 (feb core 224.86, jun core 217.73; LIRF all 662.5).
- + daily calibration: 265.24 / 218.90 / 200.84. + fill-aware at LIRF: 262.41 / 217.90 / 200.73 (feb core 220.63,
  jun core 215.73; LIRF all 646.7). Ensemble -> calibrated, core, paired day bootstrap +4.3 s [1.6, 7.4].
- Reading: the calibration gains found on January / July carry over to two unseen months, with the same
  ordering (daily helps, the fill-aware LIRF step helps most on all flights). The recipe was not tuned to the
  January / July board.
