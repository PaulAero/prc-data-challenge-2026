#!/usr/bin/env bash
# Final phase (Jan + Feb + Jun + Jul 2026). Before running, point data/raw/ranking.parquet
# and data/raw/submitting.parquet to the organisers' final_ranking / final_submitting files (use a separate
# working copy: data/processed is rebuilt). Same recipe as
# quick-quicksand_v17 (candidate_v21): no ADS-B tracks exist here for February and June, so the corrector
# only touches matched January / July departures.
set -euo pipefail
cd "$(dirname "$0")/.."
export POLARS_MAX_THREADS=12 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
step() { echo "=== $1 $(date +%H:%M:%S)"; }
step features;  uv run python scripts/protocol/build_features.py
step regime;    uv run python scripts/regime/build.py
infer_family() {  # infer_family <out tag> <seed tag prefix> <seeds...> -- <flags...>
  local out="$1" pre="$2"; shift 2; local seeds=() tags=()
  while [ "$1" != "--" ]; do seeds+=("$1"); shift; done; shift
  for s in "${seeds[@]}"; do
    mkdir -p "data/processed/latent_set${pre}$s"
    uv run --group timesfm python scripts/latent_set/infer.py --target final --device cuda --scope full "$@" \
      --tag "${pre}$s"
    mv "data/processed/latent_set${pre}$s/final_cuda.parquet" "data/processed/latent_set${pre}$s/final.parquet"
    tags+=("${pre}$s")
  done
  uv run python scripts/latent_set/seed_average.py --out "$out" --tags "${tags[@]}" --folds final
}
step latent
infer_family _noarr_wx_avg3 _noarr_wx_s 20260926 1 2 -- --weather --no-arrivals
infer_family _v2_avg3 _v2_s 20260926 1 2 -- --weather --no-arrivals --airline --huber 900
infer_family _v2b_avg3 _v2b_s 3 4 5 -- --weather --no-arrivals --airline --huber 900
infer_family _v3full_avg3 _v3full_s 6 7 8 -- --weather --airline --huber 900
step models
for v in v10r v11r v12r v15r; do
  for m in lgb cb; do uv run python scripts/protocol/make_submission.py "configs/$v-$m.yaml"; done
  uv run python scripts/protocol/blend_submissions.py "$v" data/submissions/candidate_$v-lgb.parquet data/submissions/candidate_$v-cb.parquet
done
uv run python scripts/protocol/blend_submissions.py ens4r data/submissions/candidate_v10r.parquet \
  data/submissions/candidate_v11r.parquet data/submissions/candidate_v12r.parquet data/submissions/candidate_v15r.parquet
uv run python scripts/protocol/apply_thy.py data/submissions/candidate_ens4r.parquet v20base data/processed/latent_set_v3full_avg3/final.parquet
step adsb
uv run python scripts/adsb/stands.py
uv run python scripts/adsb/features.py
ADSB_BASE_SUFFIX=r uv run python scripts/adsb/apply_ranking.py data/submissions/candidate_v20base.parquet v20a --cb
step calib
uv run python scripts/protocol/apply_daily.py data/submissions/candidate_v20a.parquet v21d
uv run python scripts/protocol/apply_pru.py data/submissions/candidate_v21d.parquet v21
cp data/submissions/candidate_v21.parquet data/submissions/quick-quicksand_final.parquet
step check
uv run python - <<'PY'
import polars as pl
s = pl.read_parquet("data/submissions/quick-quicksand_final.parquet"); t = pl.read_parquet("data/raw/submitting.parquet")
assert s.columns == t.columns and s.height == t.height and s["MVT_ID_mvt"].equals(t["MVT_ID_mvt"]), "template mismatch"
assert s["TAXITIME_SEC_mvt"].null_count() == 0
print("final file ok:", s.height, "rows, min", s["TAXITIME_SEC_mvt"].min(), "max", s["TAXITIME_SEC_mvt"].max())
PY
step done
