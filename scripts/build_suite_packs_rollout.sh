#!/usr/bin/env bash
# Build blind + entropy-guided W4A4 packs from a REAL-ROLLOUT observation
# pool (record_libero_rollout_obs_pi05.py's obs_pool_rollout.pt, 60 samples
# spanning real mid-task states) instead of the static-opening-frame pool
# build_suite_packs.sh uses. Same downstream recipe otherwise (official
# flow: PaliGemma GPTQ + Expert RTN, group-64 kernel format, rotation NOT
# folded).
#
# The "blind" arm here is deliberately NOT pool[:10] -- unlike the old
# task-cycling pool, the rollout pool is ordered
# [task0_ep0_frac.2, task0_ep0_frac.5, task0_ep0_frac.8, task0_ep1_..., task1_...],
# so a naive pool[:10] would grab ~2 tasks' worth of frames instead of one
# frame per task. Blind10 is built by taking the first frame of episode 0
# for each of the 10 tasks (grouped by first-occurrence of each unique
# `prompt`), preserving the "one representative frame per task" property
# the historical blind baseline has (see MECHANISM_CN.md section 2).
#
# Usage: build_suite_packs_rollout.sh <libero_suite_name> <cache_subdir_name> <build_gpu>
set -euo pipefail

SUITE="$1"
NAME="$2"
BUILD_GPU="$3"

ROOT=/private/xzq/EntropyCalib-QVLA
OPENPI_PY=/private/xzq/openpi/.venv/bin/python
CHECKPOINT=/private/xzq/openpi/checkpoints/pi05_libero/pi05_independent_pytorch_l20_seed42/30000
CACHE="$ROOT/cache/$NAME"
LOG="$CACHE/logs"
mkdir -p "$CACHE" "$LOG"

POOL="$CACHE/obs_pool_rollout.pt"
if [ ! -f "$POOL" ]; then
  echo "Missing $POOL -- run record_libero_rollout_obs_pi05.py first" >&2
  exit 1
fi

step() { echo "=== [$(date +%H:%M:%S)] [$NAME/rollout] $* ===" | tee -a "$LOG/pipeline_rollout.log"; }

cd "$ROOT"

step "1/6 scoring entropy on the rollout pool (60 samples x 6 noise draws)"
env CUDA_VISIBLE_DEVICES=$BUILD_GPU "$OPENPI_PY" scripts/compute_entropy_labels_pi05.py \
  --checkpoint "$CHECKPOINT" --obs-path "$POOL" \
  --num-noise-samples 6 --output "$CACHE/entropy_rollout.npz" >> "$LOG/r01_entropy.log" 2>&1

step "2/6 selecting entropy-guided subset"
env CUDA_VISIBLE_DEVICES=$BUILD_GPU "$OPENPI_PY" scripts/select_entropy_calibration_obs.py \
  --obs-path "$POOL" --entropy-cache "$CACHE/entropy_rollout.npz" \
  --num-samples 10 --output-obs-path "$CACHE/obs_entropy10_rollout.pt" >> "$LOG/r02_select.log" 2>&1

step "2b/6 building task-balanced blind subset (one frame per task, episode 0)"
"$OPENPI_PY" -c "
import torch
pool = torch.load('$POOL', weights_only=False)
seen = {}
for e in pool:
    if e['prompt'] not in seen:
        seen[e['prompt']] = e
blind = list(seen.values())[:10]
assert len(blind) == 10, f'expected 10 unique tasks, got {len(blind)}'
torch.save(blind, '$CACHE/obs_baseline10_rollout.pt')
print(len(blind), 'samples,', len(seen), 'unique tasks seen')
" >> "$LOG/r02_select.log" 2>&1

step "3/6 building 4 packs (paligemma/expert x baseline/entropy) on GPU $BUILD_GPU"
cd /private/xzq/Omega-QVLA-official-baseline
env CUDA_VISIBLE_DEVICES=$BUILD_GPU "$OPENPI_PY" "$ROOT/scripts/build_pi05_official_pack.py" \
  --component expert --checkpoint "$CHECKPOINT" --obs-path "$CACHE/obs_baseline10_rollout.pt" \
  --output "$CACHE/packs_rollout/expert_baseline/quantized.pt" --max-samples 10 \
  > "$LOG/r03_expert_baseline.log" 2>&1
env CUDA_VISIBLE_DEVICES=$BUILD_GPU "$OPENPI_PY" "$ROOT/scripts/build_pi05_official_pack.py" \
  --component expert --checkpoint "$CHECKPOINT" --obs-path "$CACHE/obs_entropy10_rollout.pt" \
  --output "$CACHE/packs_rollout/expert_entropy/quantized.pt" --max-samples 10 \
  > "$LOG/r03_expert_entropy.log" 2>&1
env CUDA_VISIBLE_DEVICES=$BUILD_GPU "$OPENPI_PY" "$ROOT/scripts/build_pi05_official_pack.py" \
  --component paligemma --checkpoint "$CHECKPOINT" --obs-path "$CACHE/obs_baseline10_rollout.pt" \
  --output "$CACHE/packs_rollout/paligemma_baseline/quantized.pt" --max-samples 10 \
  > "$LOG/r03_paligemma_baseline.log" 2>&1
env CUDA_VISIBLE_DEVICES=$BUILD_GPU "$OPENPI_PY" "$ROOT/scripts/build_pi05_official_pack.py" \
  --component paligemma --checkpoint "$CHECKPOINT" --obs-path "$CACHE/obs_entropy10_rollout.pt" \
  --output "$CACHE/packs_rollout/paligemma_entropy/quantized.pt" --max-samples 10 \
  > "$LOG/r03_paligemma_entropy.log" 2>&1
cd "$ROOT"

step "4/6 merging packs"
cd /private/xzq/Omega-QVLA-official-baseline
"$OPENPI_PY" -m tools.merge_packs --out "$CACHE/packs_rollout/merged_baseline/quantized.pt" \
  "$CACHE/packs_rollout/paligemma_baseline/quantized.pt" "$CACHE/packs_rollout/expert_baseline/quantized.pt" \
  >> "$LOG/r04_merge.log" 2>&1
"$OPENPI_PY" -m tools.merge_packs --out "$CACHE/packs_rollout/merged_entropy/quantized.pt" \
  "$CACHE/packs_rollout/paligemma_entropy/quantized.pt" "$CACHE/packs_rollout/expert_entropy/quantized.pt" \
  >> "$LOG/r04_merge.log" 2>&1
cd "$ROOT"

step "5/6 converting to group-64 packed INT4 (rotation NOT folded)"
cd /private/xzq/Omega-QVLA
"$OPENPI_PY" -m tools.pack_pi05_groupwise_int4 \
  --input "$CACHE/packs_rollout/merged_baseline/quantized.pt" \
  --output "$CACHE/packs_rollout/merged_baseline_group64_rot/quantized.pt" \
  --group-size 64 >> "$LOG/r05_convert.log" 2>&1
"$OPENPI_PY" -m tools.pack_pi05_groupwise_int4 \
  --input "$CACHE/packs_rollout/merged_entropy/quantized.pt" \
  --output "$CACHE/packs_rollout/merged_entropy_group64_rot/quantized.pt" \
  --group-size 64 >> "$LOG/r05_convert.log" 2>&1
cd "$ROOT"

step "6/6 PACK BUILD COMPLETE for $NAME (rollout-based pools)"
