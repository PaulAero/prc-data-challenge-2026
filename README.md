# Taxi-out time prediction: PRC Data Challenge 2026 (team quick-quicksand)

Code and documentation of team **quick-quicksand** (Paul PROVOST, co-authored with Claude, Anthropic) for the
[EUROCONTROL PRC Data Challenge 2026](https://ansperformance.eu/study/data-challenge/): predict the
taxi-out time (take-off minus recorded off-block) of departures at ten European airports, scored by RMSE.

The full method is in **[`docs/method.md`](docs/method.md)**; every experiment, kept or rejected, is in
[`docs/experiment_log.md`](docs/experiment_log.md) (EXP-001 to EXP-058).

![Architecture of the solution](docs/architecture.svg)

## In short

The score is decided by how off-block times are *recorded*, not only by how aircraft taxi. The solution
combines:

1. **Record forensics.** About 5% of departures (18% at Rome Fiumicino) carry the schedule instead of a
   real off-block; the NM flight match only exists when the off-block is within 60 min of the last filed
   off-block time; some Rome records are stored 24 h early; recording regimes switch by airline and
   month, and the switch is visible in the same airline's arrivals.
2. **A two-part model:** P(record = schedule) x (take-off - schedule) + (1 - P) x a regression trained on
   targets clipped at 3 h; LightGBM and CatBoost on queue, stand, NM, weather and geometry features, plus
   a learned transformer latent over 2-hour airport movement windows ([zoom](docs/latent.svg)); eight
   models averaged.
3. **Open ADS-B surface tracks** (adsb.lol): stand positions learned from arrival tracks, then a corrector
   of the ensemble residual from each departure's visible ground track.
4. **Calibration to EUROCONTROL's published aggregates** (daily pre-departure delay, monthly taxi-out
   averages), fill-aware at Rome.

## Results

![How the score came down](docs/progress.svg)

| | RMSE (s) |
| --- | --- |
| Public leaderboard (January + July 2026), best upload `quick-quicksand_v17` | 244.38 |
| Local validation, January + July 2025 whole-month holdouts, ordinary flights (<= 3 h) | 225.3 |
| Generalisation check, February + June 2025 holdouts never used for tuning, ordinary flights (no ADS-B tracks for these months) | 217.9 |

The final-phase file (January, February, June and July 2026) was produced with the same recipe
(`scripts/final_phase.sh`). It was uploaded after the deadline because of an internet outage and is ranked
as not eligible for the prize.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/). The official files go in `data/raw/` (12 monthly training
files, `ranking.parquet`, `submitting.parquet`).

```bash
uv sync --group timesfm --group weather
WITH_OOF=1 scripts/reproduce.sh   # stages: check external features latent models oof adsb calib
```

- A GPU is recommended for training the latent networks (inference runs on CPU).
- The ADS-B stage downloads about 3.4 GB per day of adsb.lol archives (124 days); set `ADSB_ROOT` to a disk
  with about 400 GB free (default `data/adsb`). Downloads use a single rate-limited stream.
- Uploading needs the team's MinIO keys in `.env` (see `.env.example`); keys are never committed.

Any stage can be run alone, e.g. `scripts/reproduce.sh models`. The validation protocol is described in
[`docs/validation.md`](docs/validation.md).

## Layout

- `src/prc/`: data loading, feature blocks, two-part models, validation protocol, latent network.
- `scripts/protocol/`: features, validation runs, final models, blending, calibration steps.
- `scripts/latent_set/`: training and inference of the transformer latent.
- `scripts/adsb/`: ADS-B download, extraction, stand map, track features, corrector.
- `scripts/regime/`: arrival recording-regime features. `scripts/lab/`: experiments behind the log.
- `configs/`: model configurations (`v1{0,1,2,5}r-{lgb,cb}` are the final members).
- `apps/explorer/`: a local data explorer (Streamlit).

## Data credits

- Official challenge data: EUROCONTROL PRC Data Challenge 2026, via the OpenSky Network.
- Weather: Open-Meteo historical weather API, CC BY 4.0, "Weather data by Open-Meteo.com"
  (<https://open-meteo.com/>), based on Copernicus ERA5. Official ERA5 / ERA5-Land time series from
  the Copernicus Climate Data Store are also fetched (`scripts/external/fetch_weather_cds.py`, CC BY
  4.0, "Contains modified Copernicus Climate Change Service information").
- ADS-B surface tracks: adsb.lol daily archives (<https://github.com/adsblol/globe_history_2025>,
  `globe_history_2026`), ODbL 1.0 (repository) and CC0 (archive contents). Use confirmed by the
  organisers under the open-data rule.
- Taxi-out and pre-departure delay aggregates: source EUROCONTROL (ansperformance.eu); use of the published
  aggregates confirmed by the organisers.
- Airport geometry: (c) OpenStreetMap contributors, ODbL 1.0 (<https://www.openstreetmap.org/copyright>).
  Derived databases published from it are offered under ODbL.

## Licence

GNU General Public License v3.0 (see `LICENSE`).
