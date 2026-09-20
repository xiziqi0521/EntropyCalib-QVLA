#!/usr/bin/env bash
# End-to-end entropy-guided vs blind calibration comparison on LIBERO's
# "Long" task suite (libero_10) -- the suite most sensitive to quantization
# quality for this checkpoint per prior reports, and with more trials/task
# (5) than the earlier libero_object smoke run (1), for real statistical
# power. Runs fully unattended: safe to close the laptop after launching.
#
# Everything here is either read-only against the two source repos (via
# sys.path imports) or writes only under EntropyCalib-QVLA/cache -- no
# original repo is modified.
set -euo pipefail

ROOT=/private/xzq/EntropyCalib-QVLA
OPENPI_PY=/private/xzq/openpi/.venv/bin/python
CHECKPOINT=/private/xzq/openpi/checkpoints/pi05_libero/pi05_independent_pytorch_l20_seed42/30000
SUITE=libero_10
CACHE="$ROOT/cache/long"
LOG="$CACHE/logs"
mkdir -p "$CACHE" "$LOG"

NVIDIA_GL_ENV=(
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  __EGL_VENDOR_LIBRARY_FILENAMES=/private/xzq/nvidia-gl-570/usr/share/glvnd/egl_vendor.d/10_nvidia.json
  LD_LIBRARY_PATH="/private/xzq/nvidia-gl-570/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
  LIBERO_CONFIG_PATH=/private/xzq/libero_config
)
LIBERO_PYTHONPATH=/private/xzq/openpi/third_party/libero
NO_PROXY_ENV=(-u ALL_PROXY -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1)

step() { echo "=== [$(date +%H:%M:%S)] $* ===" | tee -a "$LOG/pipeline.log"; }

cd "$ROOT"

# --- 1. Record a 50-sample LIBERO obs pool for the Long suite -------------
step "1/8 recording 50-sample obs pool (libero_10)"
cd /private/xzq/Omega-QVLA
env "${NVIDIA_GL_ENV[@]}" PYTHONPATH="$LIBERO_PYTHONPATH" "$OPENPI_PY" -m tools.record_libero_obs_for_pi05 \
  --task-suite-name "$SUITE" --num-samples 50 \
  --output "$CACHE/obs_pool50.pt" >> "$LOG/01_record.log" 2>&1
cd "$ROOT"

# --- 2. Score entropy on the pool ------------------------------------------
step "2/8 scoring entropy (50 samples x 6 noise draws)"
"$OPENPI_PY" scripts/compute_entropy_labels_pi05.py \
  --checkpoint "$CHECKPOINT" --obs-path "$CACHE/obs_pool50.pt" \
  --num-noise-samples 6 --output "$CACHE/entropy.npz" >> "$LOG/02_entropy.log" 2>&1

# --- 3. Build the two 10-sample calibration subsets ------------------------
step "3/8 selecting entropy-guided subset"
"$OPENPI_PY" scripts/select_entropy_calibration_obs.py \
  --obs-path "$CACHE/obs_pool50.pt" --entropy-cache "$CACHE/entropy.npz" \
  --num-samples 10 --output-obs-path "$CACHE/obs_entropy10.pt" >> "$LOG/03_select.log" 2>&1

step "3b/8 building blind (first-10) baseline subset"
"$OPENPI_PY" -c "
import torch
pool = torch.load('$CACHE/obs_pool50.pt', weights_only=False)
torch.save(pool[:10], '$CACHE/obs_baseline10.pt')
print(len(pool[:10]), 'samples')
" >> "$LOG/03_select.log" 2>&1

# --- 4. Build PaliGemma + Expert packs for both, sequentially on GPU 6 ----
# GPUs 1-4 are occupied by other tenants' VLLM workloads on this shared box
# (not ours to touch); GPU 0 has leftover context from this session. GPU 6
# was free at pipeline-launch time, so everything below pins to it and runs
# sequentially rather than risk resource contention via parallelism.
BUILD_GPU=6
step "4/8 building 4 packs (paligemma/expert x baseline/entropy) sequentially on GPU $BUILD_GPU"
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

# --- 5. Merge into two 252-layer packs -------------------------------------
step "5/8 merging packs"
cd /private/xzq/Omega-QVLA-official-baseline
"$OPENPI_PY" -m tools.merge_packs --out "$CACHE/packs/merged_baseline/quantized.pt" \
  "$CACHE/packs/paligemma_baseline/quantized.pt" "$CACHE/packs/expert_baseline/quantized.pt" \
  >> "$LOG/05_merge.log" 2>&1
"$OPENPI_PY" -m tools.merge_packs --out "$CACHE/packs/merged_entropy/quantized.pt" \
  "$CACHE/packs/paligemma_entropy/quantized.pt" "$CACHE/packs/expert_entropy/quantized.pt" \
  >> "$LOG/05_merge.log" 2>&1
cd "$ROOT"

# --- 6. Convert both to group-64 packed INT4 (kernel-consumable, and the
#        path already confirmed NOT to hit the runtime-rotation bug) -------
step "6/8 converting to group-64 packed INT4"
cd /private/xzq/Omega-QVLA
"$OPENPI_PY" -m tools.pack_pi05_groupwise_int4 \
  --input "$CACHE/packs/merged_baseline/quantized.pt" \
  --output "$CACHE/packs/merged_baseline_group64/quantized.pt" \
  --group-size 64 --fold-duquant-rotations >> "$LOG/06_convert.log" 2>&1
"$OPENPI_PY" -m tools.pack_pi05_groupwise_int4 \
  --input "$CACHE/packs/merged_entropy/quantized.pt" \
  --output "$CACHE/packs/merged_entropy_group64/quantized.pt" \
  --group-size 64 --fold-duquant-rotations >> "$LOG/06_convert.log" 2>&1
cd "$ROOT"

GPTQ_INCLUDE='.*paligemma_with_expert\.(paligemma\.model\.language_model|gemma_expert\.model)\.layers\.[0-9]+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*'
GPTQ_EXCLUDE='(?:^|\.)(vision_tower|vision_model|embeddings|embed_tokens|norm|layernorm|lm_head)(?:\.|$)'

run_eval() {
  local label="$1" pack="$2" outdir="$3"
  step "starting server for $label"
  cd /private/xzq/Omega-QVLA-official-baseline
  pkill -9 -f "scripts/openpi_inference_service.py" 2>/dev/null || true
  sleep 3
  env "${NO_PROXY_ENV[@]}" PYTHONPATH=/private/xzq/Omega-QVLA CUDA_VISIBLE_DEVICES=$BUILD_GPU \
    GR00T_GPTQ=1 GR00T_GPTQ_PATH="$pack" \
    GR00T_GPTQ_INCLUDE="$GPTQ_INCLUDE" GR00T_GPTQ_EXCLUDE="$GPTQ_EXCLUDE" \
    GR00T_GPTQ_WBITS_DEFAULT=4 GR00T_GPTQ_ABITS=4 GR00T_GPTQ_MISSING=error \
    GR00T_OMEGA_W4A4=1 GR00T_W4A4_GROUP_SIZE=64 \
    "$OPENPI_PY" scripts/openpi_inference_service.py \
    --model-path "$CHECKPOINT" --data-config pi05_libero --port 8100 \
    > "$LOG/server_${label}.log" 2>&1 &
  local server_pid=$!
  until grep -q "Creating OpenPI websocket server\|Traceback" "$LOG/server_${label}.log" 2>/dev/null; do
    sleep 10
  done
  if grep -q "Traceback" "$LOG/server_${label}.log"; then
    echo "SERVER FAILED for $label -- see $LOG/server_${label}.log" | tee -a "$LOG/pipeline.log"
    return 1
  fi
  step "running eval for $label (5 trials/task, libero_10 -- this is the long one)"
  cd /private/xzq/Omega-QVLA-official-baseline
  mkdir -p "$outdir"
  env "${NO_PROXY_ENV[@]}" "${NVIDIA_GL_ENV[@]}" PYTHONPATH="$LIBERO_PYTHONPATH" \
    "$OPENPI_PY" -m examples.Libero.eval.run_libero_eval \
    --task-suite-name "$SUITE" --num-trials-per-task 5 \
    --policy-backend openpi_ws --port 8100 --max-steps-profile openpi --replan-steps 5 \
    --no-save-videos --log-dir "$outdir/logs" --summary-json "$outdir/summary.json" \
    > "$LOG/eval_${label}.log" 2>&1
  kill -9 "$server_pid" 2>/dev/null || true
  sleep 3
  step "$label done: $(python3 -c "import json; d=json.load(open('$outdir/summary.json')); print(d['total_successes'], '/', d['total_episodes'], '=', d['total_success_rate'])" 2>/dev/null || echo 'summary missing')"
}

# --- 7. Eval blind-calibration pack on Long, 5 trials/task -----------------
step "7/8 blind-calibration eval"
run_eval "baseline" "$CACHE/packs/merged_baseline_group64/quantized.pt" "$CACHE/eval_baseline"

# --- 8. Eval entropy-guided pack on Long, 5 trials/task --------------------
step "8/8 entropy-guided eval"
run_eval "entropy" "$CACHE/packs/merged_entropy_group64/quantized.pt" "$CACHE/eval_entropy"

step "PIPELINE COMPLETE"
python3 -c "
import json
b = json.load(open('$CACHE/eval_baseline/summary.json'))
e = json.load(open('$CACHE/eval_entropy/summary.json'))
print(f\"Baseline (blind):   {b['total_successes']}/{b['total_episodes']} = {b['total_success_rate']:.1%}\")
print(f\"Entropy-guided:     {e['total_successes']}/{e['total_episodes']} = {e['total_success_rate']:.1%}\")
" | tee -a "$LOG/pipeline.log"
