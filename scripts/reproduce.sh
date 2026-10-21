#!/usr/bin/env bash
# Reproduce the submission from the official files in data/raw/.
#
# Usage: scripts/reproduce.sh [stage ...]      (default: all stages in order)
# Stages:
#   check     official files present and the template matches the ranking departures
#   external  open external data: weather (Open-Meteo, CC BY 4.0) and airport geometry
#             (OpenStreetMap, ODbL); skipped when already downloaded
#   features  feature cache (data/processed/features/)
#   latent    learned flight-centred latents (GPU recommended for training; inference runs on CPU):
#             A = departures + weather, seeds 20260926/1/2 -> data/processed/latent_set_noarr_wx_avg3/
#             B = A + airline token + Huber loss, seeds 20260926/1/2 -> latent_set_v2_avg3/
#             C = B with seeds 3/4/5 -> latent_set_v2b_avg3/
#             D = B with arrival tokens kept, seeds 6/7/8 -> latent_set_v3full_avg3/
#   models    on each latent, a LightGBM (3-seed bag) and a CatBoost two-part model fitted on all
#             2025 and applied to the ranking file (configs v10 / v11 / v12 / v15), the equal average
#             of the four stacks (configs v1xr: with the arrival regime features, EXP-054), then the THY LIRF
#             January rule (EXP-050) -> data/submissions/candidate_v20base.parquet
#   oof       out-of-fold predictions of the eight models on January and July 2025 (whole-month
#             holdouts, latents re-trained per fold) -> experiments/v1{0,1,2,5}-{lgb,cb}/predictions.parquet
#   adsb      adsb.lol surface tracks (ODbL / CC0; about 3.4 GB per day, 124 days, set ADSB_ROOT),
#             stand positions, per-departure track features, then a LightGBM corrector of the
#             ensemble residual trained on the January/July 2025 out-of-fold predictions and applied
#             to matched 2026 departures outside LIRF (EXP-052/053)
#             -> data/submissions/candidate_v20a.parquet
#   calib     calibration to EUROCONTROL's published daily pre-departure delay and monthly taxi-out
#             averages (EXP-055/056, fill-aware at LIRF) -> data/submissions/candidate_v21.parquet (quick-quicksand_v17)
#
# Environment: POLARS_MAX_THREADS (default 12). Model configs: configs/v1{0,1,2}-{lgb,cb}.yaml. Validation of the same configs: scripts/protocol/run.py configs/<id>.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."
export POLARS_MAX_THREADS="${POLARS_MAX_THREADS:-12}"
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
LATENT_FLAGS=(--weather --no-arrivals)
V2_FLAGS=(--weather --no-arrivals --airline --huber 900)
V3_FLAGS=(--weather --airline --huber 900)  # arrival tokens kept: run as a capped unit (peak about 19 GB)

stage_check() {
  ls data/raw/training_2025-*.parquet data/raw/ranking.parquet data/raw/submitting.parquet >/dev/null
  uv run python - <<'PY'
import polars as pl
r = pl.read_parquet("data/raw/ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP")
t = pl.read_parquet("data/raw/submitting.parquet")
assert set(r["MVT_ID_mvt"].to_list()) == set(t["MVT_ID_mvt"].to_list()), "template != ranking departures"
print(f"check ok: {t.height} departures to predict")
PY
}

stage_external() {
  ls data/external/weather/openmeteo_*.parquet >/dev/null 2>&1 || uv run python scripts/external/fetch_weather.py
  ls data/external/osm/*.json >/dev/null 2>&1 || uv run python scripts/external/fetch_osm.py
  # EUROCONTROL published aggregates (ansperformance.eu; source: EUROCONTROL): monthly taxi-out (ATXOT)
  # and daily pre-departure delay, 2025 and 2026
  mkdir -p data/external/pru
  for y in 2025 2026; do
    for f in taxi_out_additional_time all_pre_departure_delays; do
      [ -s "data/external/pru/${f}_$y.csv" ] || curl -sSL --fail --limit-rate 3M \
        -o "data/external/pru/${f}_$y.csv" "https://www.eurocontrol.int/performance/data/download/csv/${f}_$y.csv"
    done
  done
}

stage_features() {
  uv run python scripts/protocol/build_features.py
  uv run python scripts/regime/build.py  # arrival recording-regime features (EXP-054)
}

latent_family() {  # latent_family <out tag> <seed tag prefix> <seeds...> -- <flags...>
  local out="$1" pre="$2"; shift 2
  local seeds=() tags=()
  while [ "$1" != "--" ]; do seeds+=("$1"); shift; done; shift
  for s in "${seeds[@]}"; do
    uv run --group timesfm python scripts/latent_set/run_final.py "$@" --seed "$s" --tag "${pre}$s" --overwrite
    [ "${WITH_OOF:-0}" = 1 ] && uv run --group timesfm python scripts/latent_set/run_folds.py "$@" \
      --seed "$s" --tag "${pre}$s" --folds jan jul
    tags+=("${pre}$s")
  done
  local folds=(final); [ "${WITH_OOF:-0}" = 1 ] && folds=(jan jul final)
  uv run python scripts/latent_set/seed_average.py --out "$out" --tags "${tags[@]}" --folds "${folds[@]}"
}

stage_latent() {
  latent_family _noarr_wx_avg3 _noarr_wx_s 20260926 1 2 -- "${LATENT_FLAGS[@]}"
  latent_family _v2_avg3 _v2_s 20260926 1 2 -- "${V2_FLAGS[@]}"
  latent_family _v2b_avg3 _v2b_s 3 4 5 -- "${V2_FLAGS[@]}"
  latent_family _v3full_avg3 _v3full_s 6 7 8 -- "${V3_FLAGS[@]}"
}

stage_models() {
  for v in v10r v11r v12r v15r; do
    uv run python scripts/protocol/make_submission.py "configs/$v-lgb.yaml"
    uv run python scripts/protocol/make_submission.py "configs/$v-cb.yaml"
    uv run python scripts/protocol/blend_submissions.py "$v" \
      "data/submissions/candidate_$v-lgb.parquet" "data/submissions/candidate_$v-cb.parquet"
  done
  # pair blends first, then their average (the roundings of the uploaded file), then the THY rule
  uv run python scripts/protocol/blend_submissions.py ens4r data/submissions/candidate_v10r.parquet \
    data/submissions/candidate_v11r.parquet data/submissions/candidate_v12r.parquet data/submissions/candidate_v15r.parquet
  uv run python scripts/protocol/apply_thy.py data/submissions/candidate_ens4r.parquet v20base \
    data/processed/latent_set_v3full_avg3/final.parquet
}

stage_oof() {  # needs the fold latents: run the latent stage with WITH_OOF=1 first
  for v in v10r v11r v12r v15r; do
    uv run python scripts/protocol/run.py "configs/$v-lgb.yaml"
    uv run python scripts/protocol/run.py "configs/$v-cb.yaml"
  done
}

stage_adsb() {  # heavy download: one stream capped at 6 MB/s (a shared line saturates near 12 MB/s)
  export ADSB_ROOT="${ADSB_ROOT:-data/adsb}"
  uv run python scripts/adsb/fetch_all.py
  uv run python scripts/adsb/stands.py
  uv run python scripts/adsb/features.py
  ADSB_BASE_SUFFIX=r uv run python scripts/adsb/apply_ranking.py data/submissions/candidate_v20base.parquet v20a --cb
}

stage_calib() {  # EUROCONTROL aggregates: daily pre-departure delay (EXP-056), then monthly taxi-out (EXP-055)
  uv run python scripts/protocol/apply_daily.py data/submissions/candidate_v20a.parquet v21d  # LIRF: fill-aware
  uv run python scripts/protocol/apply_pru.py data/submissions/candidate_v21d.parquet v21
}

stages=("$@")
[ ${#stages[@]} -eq 0 ] && stages=(check external features latent models oof adsb calib)
for s in "${stages[@]}"; do
  echo "=== stage: $s ($(date +%H:%M:%S))"
  "stage_$s"
done
echo "done: data/submissions/candidate_v21.parquet"
