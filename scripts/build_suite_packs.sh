#!/usr/bin/env bash
# Build blind + entropy-guided W4A4 packs (PaliGemma GPTQ + Expert RTN,
# group-64 kernel format, rotation NOT folded -- the corrected recipe) for
# one LIBERO task suite. Usage:
#   build_suite_packs.sh <libero_suite_name> <cache_subdir_name> <build_gpu>
# e.g. build_suite_packs.sh libero_object object 6
#
# Read-only against Omega-QVLA / Omega-QVLA-official-baseline / openpi;
# writes only under EntropyCalib-QVLA/cache/<cache_subdir_name>.
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

NVIDIA_GL_ENV=(
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  __EGL_VENDOR_LIBRARY_FILENAMES=/private/xzq/nvidia-gl-570/usr/share/glvnd/egl_vendor.d/10_nvidia.json
  LD_LIBRARY_PATH="/private/xzq/nvidia-gl-570/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
  LIBERO_CONFIG_PATH=/private/xzq/libero_config
)
LIBERO_PYTHONPATH=/private/xzq/openpi/third_party/libero

step() { echo "=== [$(date +%H:%M:%S)] [$NAME] $* ===" | tee -a "$LOG/pipeline.log"; }

cd "$ROOT"

step "1/6 recording 50-sample obs pool ($SUITE)"
cd /private/xzq/Omega-QVLA
env "${NVIDIA_GL_ENV[@]}" PYTHONPATH="$LIBERO_PYTHONPATH" "$OPENPI_PY" -m tools.record_libero_obs_for_pi05 \
  --task-suite-name "$SUITE" --num-samples 50 \
  --output "$CACHE/obs_pool50.pt" >> "$LOG/01_record.log" 2>&1
cd "$ROOT"

step "2/6 scoring entropy (50 samples x 6 noise draws)"
env CUDA_VISIBLE_DEVICES=$BUILD_GPU "$OPENPI_PY" scripts/compute_entropy_labels_pi05.py \
  --checkpoint "$CHECKPOINT" --obs-path "$CACHE/obs_pool50.pt" \
  --num-noise-samples 6 --output "$CACHE/entropy.npz" >> "$LOG/02_entropy.log" 2>&1

step "3/6 selecting entropy-guided + blind subsets"
env CUDA_VISIBLE_DEVICES=$BUILD_GPU "$OPENPI_PY" scripts/select_entropy_calibration_obs.py \
  --obs-path "$CACHE/obs_pool50.pt" --entropy-cache "$CACHE/entropy.npz" \
  --num-samples 10 --output-obs-path "$CACHE/obs_entropy10.pt" >> "$LOG/03_select.log" 2>&1
"$OPENPI_PY" -c "
import torch
pool = torch.load('$CACHE/obs_pool50.pt', weights_only=False)
torch.save(pool[:10], '$CACHE/obs_baseline10.pt')
print(len(pool[:10]), 'samples')
" >> "$LOG/03_select.log" 2>&1

step "4/6 building 4 packs (paligemma/expert x baseline/entropy) on GPU $BUILD_GPU"
cd /private/xzq/Omega-QVLA-official-baseline
env CUDA_VISIBLE_DEVICES=$BUILD_GPU "$OPENPI_PY" "$ROOT/scripts/build_pi05_official_pack.py" \
  --component expert --checkpoint "$CHECKPOINT" --obs-path "$CACHE/obs_baseline10.pt" \
  --output "$CACHE/packs/expert_baseline/quantized.pt" --max-samples 10 \
  > "$LOG/04_expert_baseline.log" 2>&1
env CUDA_VISIBLE_DEVICES=$BUILD_GPU "$OPENPI_PY" "$ROOT/scripts/build_pi05_official_pack.py" \
  --component expert --checkpoint "$CHECKPOINT" --obs-path "$CACHE/obs_entropy10.pt" \
  --output "$CACHE/packs/expert_entropy/quantized.pt" --max-samples 10 \
  > "$LOG/04_expert_entropy.log" 2>&1
env CUDA_VISIBLE_DEVICES=$BUILD_GPU "$OPENPI_PY" "$ROOT/scripts/build_pi05_official_pack.py" \
  --component paligemma --checkpoint "$CHECKPOINT" --obs-path "$CACHE/obs_baseline10.pt" \
  --output "$CACHE/packs/paligemma_baseline/quantized.pt" --max-samples 10 \
  > "$LOG/04_paligemma_baseline.log" 2>&1
env CUDA_VISIBLE_DEVICES=$BUILD_GPU "$OPENPI_PY" "$ROOT/scripts/build_pi05_official_pack.py" \
  --component paligemma --checkpoint "$CHECKPOINT" --obs-path "$CACHE/obs_entropy10.pt" \
  --output "$CACHE/packs/paligemma_entropy/quantized.pt" --max-samples 10 \
  > "$LOG/04_paligemma_entropy.log" 2>&1
cd "$ROOT"

step "5/6 merging packs"
cd /private/xzq/Omega-QVLA-official-baseline
"$OPENPI_PY" -m tools.merge_packs --out "$CACHE/packs/merged_baseline/quantized.pt" \
  "$CACHE/packs/paligemma_baseline/quantized.pt" "$CACHE/packs/expert_baseline/quantized.pt" \
  >> "$LOG/05_merge.log" 2>&1
"$OPENPI_PY" -m tools.merge_packs --out "$CACHE/packs/merged_entropy/quantized.pt" \
  "$CACHE/packs/paligemma_entropy/quantized.pt" "$CACHE/packs/expert_entropy/quantized.pt" \
  >> "$LOG/05_merge.log" 2>&1
cd "$ROOT"

step "6/6 converting to group-64 packed INT4 (rotation NOT folded -- corrected recipe)"
cd /private/xzq/Omega-QVLA
"$OPENPI_PY" -m tools.pack_pi05_groupwise_int4 \
  --input "$CACHE/packs/merged_baseline/quantized.pt" \
  --output "$CACHE/packs/merged_baseline_group64_rot/quantized.pt" \
  --group-size 64 >> "$LOG/06_convert.log" 2>&1
"$OPENPI_PY" -m tools.pack_pi05_groupwise_int4 \
  --input "$CACHE/packs/merged_entropy/quantized.pt" \
  --output "$CACHE/packs/merged_entropy_group64_rot/quantized.pt" \
  --group-size 64 >> "$LOG/06_convert.log" 2>&1
cd "$ROOT"

step "PACK BUILD COMPLETE for $NAME"
