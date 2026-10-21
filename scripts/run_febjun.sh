#!/usr/bin/env bash
# February / June 2025 generalisation check (final-phase months), run as a detached job:
#
#   systemd-run --user --unit=prc-febjun -p MemoryMax=32G -p MemorySwapMax=0 \
#     --working-directory=$HOME/IdeaProjects/prc-data-challenge --setenv=PATH=$PATH \
#     bash -c "scripts/run_febjun.sh > logs/febjun.log 2>&1"
#
# Follow with: tail -f logs/febjun.log   Stop with: systemctl --user stop prc-febjun
# Needs no internet. About 9-10 h (GPU latents ~5 h, eight models ~4 h). Resumable: finished steps are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
export POLARS_MAX_THREADS=12 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
step() { echo "=== $1 $(date '+%F %H:%M:%S')"; }
LATENT_FLAGS=(--weather --no-arrivals)
V2_FLAGS=(--weather --no-arrivals --airline --huber 900)
V3_FLAGS=(--weather --airline --huber 900)
family() {  # family <out tag> <seed tag prefix> <seeds...> -- <flags...>
  local out="$1" pre="$2"; shift 2; local seeds=() tags=()
  while [ "$1" != "--" ]; do seeds+=("$1"); shift; done; shift
  for s in "${seeds[@]}"; do
    if [ -s "data/processed/latent_set${pre}$s/feb.parquet" ] && [ -s "data/processed/latent_set${pre}$s/jun.parquet" ]; then
      echo "skip latent ${pre}$s"; else
      step "latent ${pre}$s"
      uv run --group timesfm python scripts/latent_set/run_folds.py "$@" --seed "$s" --tag "${pre}$s" --folds feb jun --overwrite
    fi
    tags+=("${pre}$s")
  done
  uv run python scripts/latent_set/seed_average.py --out "$out" --tags "${tags[@]}" --folds feb jun
}
step regime
uv run python scripts/regime/build.py
for f in feb jun; do cp data/processed/regime/jan.parquet "data/processed/regime/$f.parquet"; done  # label-free, same for all folds
family _noarr_wx_avg3 _noarr_wx_s 20260926 1 2 -- "${LATENT_FLAGS[@]}"
family _v2_avg3 _v2_s 20260926 1 2 -- "${V2_FLAGS[@]}"
family _v2b_avg3 _v2b_s 3 4 5 -- "${V2_FLAGS[@]}"
family _v3full_avg3 _v3full_s 6 7 8 -- "${V3_FLAGS[@]}"
for v in v10r v11r v12r v15r; do
  for m in lgb cb; do
    if [ -s "experiments/$v-$m-fj/predictions.parquet" ]; then echo "skip $v-$m-fj"; continue; fi
    step "model $v-$m-fj"; uv run python scripts/protocol/run.py "configs/$v-$m-fj.yaml"
  done
done
step eval
uv run python scripts/lab/febjun_eval.py
step done
