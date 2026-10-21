# Forensics: where the schedule-fill tail comes from (2026-09-30)

Independent forensic pass on the fill tail (EXP-032 follow-up). All measurements on 2025 training
data with day-grouped folds unless stated; ranking-file figures use `quick-quicksand_v4` (leaderboard
309.95 s, MSE 96,067) as the reference. Every signal below is computed without reading departures'
`BLOCK_TIME_UTC_mvt` or `TAXITIME_SEC_mvt`. Scratch code:
(`feat.py`, `feat2.py`, `cv.py`, `q*.py`).

## Summary, ranked by expected leaderboard impact

| # | Finding | Kind | Ranking rows | Expected MSE gain on v4 |
|---|---------|------|--------------|-------------------------|
| 1 | NM match rule: a departure is matched only if its off-block is within 60 min of NM `LOBT_flt` | hard constraint | 48 violations in v4 | **>= 14,939 (guaranteed if the rule holds), RMSE 309.95 -> <= 284.8** |
| 2 | 24-hour date artefact: off-block stored as the schedule's local date + the actual local time of day | new tail regime | 15 candidates | about 4,500 (expected, small-sample rates) |
| 3 | No-NM fills are flights whose delay was already in the flight plan (IOBT/LOBT late) and whose AOBT was never recorded | mechanism | 32 decided by recovered NM data | about 600 |
| 4 | Inbound rotation slack (schedule minus inbound in-block) | fill feature | about 90% coverage | small: fill AUC on LIRF no-NM +0.008 |
| 5 | Arrival-side fills (in-block = schedule) by airline-day | fill feature | 93% coverage | small, ordinary flights only |

Findings 1 and 2 are not about discriminating fills better: they are about records whose target is
not what the two-part model assumes. Finding 1 alone, applied to v4, removes at least 15.5% of our
leaderboard MSE, almost all from a single LIRF flight.

## 1. The NM matching window: |off-block - LOBT| <= 60 min

### Evidence
- Over all 2,062,623 NM-matched 2025 departures, `|BLOCK - LOBT_flt|` never exceeds 60.1 min
  (median 5 min, q99 44 min, max 60.1 min). The same holds at every airport. `IOBT_flt` equals
  `LOBT_flt` in 99.4% of rows.
- Nothing comparable holds for EOBT_1, AOBT_3 or IOBT beyond that (AOBT_3 gap reaches 344 min).
- Read as: the organisers' left join keeps an NM flight only if the airport off-block is within
  +-60 min of LOBT.

### Consequences (all label-free on ranking)
1. **Matched rows: target in [T - LOBT - 3600, T - LOBT + 3600]** (T = take-off). 0 violations in
   2025. Clipping predictions to this window can only reduce the error.
2. **Matched rows with |SCHED - LOBT| > 60 min can never be fills**: 66,665 such 2025 rows, 0 fills.
3. **No-NM means the off-block is more than 60 min away from the flight's LOBT** (when the NM flight
   exists). Recovering the NM flight from the destination arrival row (next section) shows the no-NM
   off-blocks sit mostly at LOBT + 60 to 90 min: the delay passed the window.

### Measured effect of the clip (existing out-of-fold predictions)
- `n-om-blend` jan+jul all: **381.9 -> 365.8 s** (jul 408.8 -> 381.4, jan 345.7 -> 345.6), 65 rows
  changed; `n-om-lgb` q4 240.58 -> 240.06. Clean and core untouched in practice (0.03% of rows).
- Clipping to the window edge beats replacing by the window centre (367.1): the violators are
  flights that really pushed back near LOBT + 60 min.

### Ranking file
- 48 matched rows of v4 fall outside their window. Clipping them reduces the leaderboard SSE by at
  least 5.15e9, i.e. **MSE -14,939 or more (309.95 -> at most 284.8 s)**, because the truth is inside
  the window.
- One row carries 99% of it: LIRF `ITY340`, schedule 07-16 19:35, take-off 07-17 21:51, LOBT
  07-16 21:44. Window [83,214 s, 90,426 s]; v4 predicts 11,718 s. This is a matched instance of the
  24-hour artefact (section 2): true off-block about 07-17 21:35, recorded on 07-16. Predict about
  86,500 to 87,400 s (T - LOBT is 86,820 s).
- Other airports: 19 LFPG rows (e.g. `DAH1001`, `ACI501`, `DAH1543` predicted 1,400 to 3,400 s with
  a lower bound of 4,500 to 6,600 s), 11 EHAM, a few elsewhere.

## 2. The 24-hour date artefact (D24)

### Mechanism
- 13 LIRF 2025 departures (plus one each at LFPG and LSZH) have a target of 84,000 to 88,400 s. In
  every case the recorded off-block has the correct local time of day but the **local calendar date
  of the schedule**, while the real off-block happened on a later local date. Example: `ITYTY630`
  schedule 09-28 08:35 UTC, take-off 09-28 22:38 UTC (00:38 local on 09-29), off-block recorded
  09-27 22:18 UTC = 09-28 00:18 local. Target = 86,400 + taxi.
- The rule works in local time (Europe/Rome); UTC dates miss some cases.
- Such rows are all no-NM (the off-block is 24 h from LOBT) unless LOBT itself was filed on the
  schedule day and the flight left a day later, as for ranking `ITY340`.

### When it happens (LIRF, no-NM, take-off local date after the schedule local date; 2025)
| Take-off local time | Rows | D24 | Fill | Normal |
|---------------------|------|-----|------|--------|
| 00:00 to 03:00 | 131 | 1 | 97 | 33 |
| 03:00 to 10:00 | 18 | 2 | 16 | 0 |
| 10:00 onwards | 19 | 9 | 10 | 0 |

Fills dominate just after midnight; for flights leaving the next day in daytime, D24 is as likely as
a fill. The two-part model never predicts D24 (the regression is trained with targets below
3,600/7,200 s), so it predicts about p x ms, far from 87,000 s.

### Measured effect
- 2025, LIRF out-of-fold: a three-way mixture on these rows,
  `p24 x (86400 + reg) + (1 - p24) x current_pred` with p24 from the table (leave-one-out),
  **cuts LIRF SSE by 29%** (4.19e10 -> 2.14e10 on 168 rows); over all 2025 departures that is MSE
  -9,800 per year of data.
- Ranking: 52 candidate rows (LIRF, no-NM, crossing), 15 with take-off at or after 03:00 local.
  Expected gain with the 2025 rates, q^2 (v4 - 87,400)^2 summed: **about 4,500 MSE**. Largest
  contributors: `ITYTY674` 07-20, `WMTT3508` 01-03, `SVII3006` 07-17, `ITYTY672` 07-10, `NOSOS431`
  07-11. The rates rest on 19 rows (9 D24): treat the gain as an order of magnitude.

## 3. Why no-NM LIRF departures are half fills: the delay was in the flight plan

### Evidence
- At LIRF, among departures that took off more than 1.5 h late, fills are 60 to 100% no-NM, while
  non-fills are only about 8% no-NM. No-NM appears exactly when the recorded off-block is more than
  1 h from LOBT (finding 1).
- The NM flight of a no-NM LIRF departure can be recovered when its destination is one of the ten
  airports: the destination arrival row (same flight number, landing 0 to 14 h after take-off)
  carries the full NM record. 216 of 1,487 train rows, 71 of 383 ranking rows.
  - Fills: IOBT/LOBT already late, close to the actual off-block (e.g. `GTI8237` IOBT +190 min,
    AOBT +216 min). The delay was known when the flight plan was filed; the airport never recorded
    the AOBT and kept the schedule, so the off-block is far from LOBT and the match fails.
  - Non-fills: IOBT = schedule, AOBT 60 to 180 min late. No delay filed; the real off-block is
    recorded, more than 60 min after LOBT, and the match fails.
- Rules on the recovered rows (LIRF, 2025):
  - A: |SCHED - LOBT| <= 60 min -> fill impossible: **76 rows, 0 fills**.
  - B: take-off - 15 min within 45 min of LOBT -> **24 rows, 92% fills**.
  - otherwise: 26 rows, 46% fills.
- Ranking LIRF: 20 rows under A, 12 under B. Moving them (A to about 950 s, B towards ms) gains about
  600 MSE on v4.
- At other airports the same recovery gives the exclusion window (target outside
  [c - 3600, c + 3600], c = T - LOBT) for 254 ranking no-NM rows, mostly irrelevant because they are
  non-fills already.

### What this means for the classifier
For a no-NM LIRF departure the question "fill or not" is really "was a delay filed in the flight plan
before departure". Where the NM record cannot be recovered (80%), the best proxies are rotation
delays (section 4).

## 4. Rotation features (inbound aircraft)

- Inbound link: last arrival at the same stand with the same 3-letter airline whose in-block precedes
  take-off minus 10 min (20 h tolerance). Coverage 90.8% train, 90.3% ranking; a stand-only fallback
  covers 97%.
- Features: `rot_slack` = schedule - inbound in-block, `rot_slack_plan` = schedule - inbound
  `ARVT_1_flt`, `rot_inb_late`, `rot_slack_sched`, `rot_inb_to_to`, each also stand-only.
- Univariate (LIRF no-NM, late > 1 h): fill rate 64 to 82% when the inbound blocked in after the
  departure's schedule, 23% when it arrived more than 30 min before. Among matched late LIRF flights
  the same slack below -60 min gives LOBT late by > 60 min in 92 to 100% (reactionary delays get
  filed).
- LightGBM fill classifier, LIRF only, 5 day-grouped folds, 600 rounds, two seeds:
  - base (current cache) AUC LIRF no-NM 0.925 / 0.926, all LIRF 0.8685 / 0.8686;
  - + rotation: 0.933 / 0.934; + recovered-NM features: 0.933 / 0.934 / 0.935;
  - fill-uncertainty term (fixed regression, D24 rows excluded) 175k / 168k per row (base) vs
    159k / 165k (+rotation +recovered): a 3 to 6% reduction, inside seed noise on the tail.
- Negative: a transfer model P(LOBT late > 60) trained on matched rows reaches AUC 0.99 (0.92
  without take-off features) on matched rows but only 0.70 for fill on no-NM rows (selection bias).
  Adding it changed nothing.

## 5. Arrival-side fills and the "10-minute" fills

- LIRF arrivals are schedule-filled too: in-block = schedule for 10% of arrivals, but **only for
  arrivals that landed before schedule**. Late landers show a spike at exactly 10 min taxi-in
  (12,451 vs 6,586 at 9 min). Departures that took off before schedule show the mirror spike at
  exactly 10 min taxi-out (281 of 1,196). Reading: a missing actual block time becomes
  max(schedule, landing + 10 min) for arrivals and min(schedule, take-off - 10 min) for departures.
- These arrival artefacts also add noise to the arrival taxi-in features (`arr_ti_*`) at LIRF.
- Airline-day fill rate among early-landing arrivals (plus stand-day, airport-day, inbound fill
  flag): all-LIRF fill AUC 0.8685 -> 0.8716, logloss 0.3239 -> 0.3192, but no tail gain (no-NM AUC
  0.930, fill term flat). Useful for ordinary fills, not for the leaderboard gap.

## 6. Ruled out

- **MVT_ID order.** IDs come in one block per airport-month. At LIRF (and EDDF, EDDM, EGLL, LEBL,
  LEMD) rows are sorted by schedule, departures and arrivals in separate runs, then cut into chunks
  of about 113 rows inserted in shuffled order. Among equal schedules the order is random with respect
  to off-block, take-off, taxi and fill (concordance 0.49 to 0.51). Chunk fill rates look like the
  daily average. EHAM, LSZH, LFPG and LTFM use another, unexplained order that follows no timestamp;
  not pursued further.
- **Second-level jitter.** At minute-resolution airports the take-off is shifted by j and the
  off-block by -j (floor), with j uniform on [-6, 6) s: identical for fills and non-fills
  (|e| mean 3.00 vs 3.00). The only use is that the target equals 60k + 2j with j read from the
  take-off seconds, a noise of about 7 s RMS, negligible.
- **Null patterns and formats.** `AIRCRAFT_TYPE_mvt` is null for every LIRF no-NM row (it comes from
  NM there), so it carries no fill information within that subset. Flight-number format, weekday and
  month show only the airline effect already captured by `airline`.
- **Day-level clustering.** Weak (as in EXP-020); the airline-day arrival fill rate is its
  observable form and helps only ordinary fills.

## 7. How the top teams could be near 220 s

- v4 MSE 96,067. Clipping to the NM window: at most 81,128 (284.8 s), guaranteed if the matching
  rule holds in 2026 as it does on all 2 million matched 2025 rows. Expected D24 handling: about
  76,600 (276.8 s). Recovered-NM rules: about 76,000.
- The remaining gap to 48,576 is still about 27,000 MSE. v4 places large bets on 32 LIRF no-NM rows
  predicted above 20,000 s (sum of squared predictions equal to 184,000 MSE). Two or three of them
  being D24 or normal instead of fills is enough to explain the rest; the top teams may simply hold
  fewer of those losses, or have the D24 and window logic plus better calibration on this subset.
- A single ranking row is worth 14,800 MSE: leaderboard gaps of 20 to 40 s can come from a handful of
  flights. Any team that noticed the 60-min matching rule, or the 24-hour dates, gains tens of seconds
  at once.

## 8. What to implement, ranked

1. **Post-processing clip for matched rows (do first; zero risk on 2025):**
   `lo = max(0, T - LOBT - 3606)`, `hi = T - LOBT + 3606` (seconds), `pred = clip(pred, lo, hi)`
   when `LOBT_flt` is present. Also force P(fill) = 0 when |SCHED - LOBT| > 3,606 s. On ranking set
   `ITY340` (MVT_ID 213147416) to about 87,000 s rather than the edge value. Note that a second
   `ITY340` (MVT_ID 213149623, matched, LOBT 07-17 20:05) left the same evening: 213147416 is the
   07-16 rotation that left 26 h late.
2. **D24 regime:** for no-NM departures whose local take-off date is after the local schedule date,
   use a three-way mixture `p_fill x ms + p24 x (86400 + reg) + (1 - p_fill - p24) x reg`, with p24
   by local take-off time (about 0.01 before 03:00, 0.11 from 03:00 to 10:00, 0.47 after 10:00 at
   LIRF; zero elsewhere for lack of cases). Keep the regression clip; do not let the regressor learn
   the D24 rows.
3. **Recovered NM record for no-NM departures** from the destination arrival row: rules A
   (P(fill) = 0) and B (P(fill) about 0.9), plus the exclusion window. Add `rec_lobt_vs_sched`,
   `rec_to_vs_lobt` as features.
4. **Rotation block** (`rot_*`, section 4) in the fill classifier; retrain and judge on the LIRF
   no-NM subset with several seeds, since the all-flight metric cannot see it.
5. **Arrival fill block** (`arrfill_*`, `inb_afill`) for ordinary fills; expect a clean-flight effect
   at most.

Adoption check for 1 and 2 on VAL-MONTH-V2: jan+jul all should drop by about 16 s from the clip
alone (measured on `n-om-blend`); core and clean should barely move since only 0.03% of rows change
(not measured separately here).
