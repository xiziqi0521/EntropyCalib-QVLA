#!/usr/bin/env bash
# Full-scale (50 trials/task) entropy-guided-then-blind eval for one suite's
# already-built merged_{baseline,entropy}_group64_rot packs (produced by
# build_suite_packs.sh). Mirrors run_long_suite_pipeline.sh's run_eval(), but
# generalized over suite/cache/gpu/port and using the corrected
# (non-rotation-folded) group64_rot packs at 50 trials/task instead of 5.
#
# Usage: run_suite_full_eval.sh <suite> <cache_name> <gpu> <port>
# e.g.   run_suite_full_eval.sh libero_spatial spatial 6 8102
set -euo pipefail

SUITE="$1"
NAME="$2"
GPU="$3"
PORT="$4"

ROOT=/private/xzq/EntropyCalib-QVLA
OPENPI_PY=/private/xzq/openpi/.venv/bin/python
CHECKPOINT=/private/xzq/openpi/checkpoints/pi05_libero/pi05_independent_pytorch_l20_seed42/30000
CACHE="$ROOT/cache/$NAME"
LOG="$CACHE/logs"
mkdir -p "$LOG"

NVIDIA_GL_ENV=(
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  __EGL_VENDOR_LIBRARY_FILENAMES=/private/xzq/nvidia-gl-570/usr/share/glvnd/egl_vendor.d/10_nvidia.json
  LD_LIBRARY_PATH="/private/xzq/nvidia-gl-570/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
  LIBERO_CONFIG_PATH=/private/xzq/libero_config
)
LIBERO_PYTHONPATH=/private/xzq/openpi/third_party/libero
NO_PROXY_ENV=(-u ALL_PROXY -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1)

GPTQ_INCLUDE='.*paligemma_with_expert\.(paligemma\.model\.language_model|gemma_expert\.model)\.layers\.[0-9]+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*'
GPTQ_EXCLUDE='(?:^|\.)(vision_tower|vision_model|embeddings|embed_tokens|norm|layernorm|lm_head)(?:\.|$)'

step() { echo "=== [$(date +%H:%M:%S)] [$NAME] $* ===" | tee -a "$LOG/pipeline.log"; }

run_eval() {
  local label="$1" pack="$2" outdir="$3"
  step "starting server for $label (port $PORT, gpu $GPU)"
  env "${NO_PROXY_ENV[@]}" PYTHONPATH=/private/xzq/Omega-QVLA CUDA_VISIBLE_DEVICES=$GPU \
    GR00T_GPTQ=1 GR00T_GPTQ_PATH="$pack" \
    GR00T_GPTQ_INCLUDE="$GPTQ_INCLUDE" GR00T_GPTQ_EXCLUDE="$GPTQ_EXCLUDE" \
    GR00T_GPTQ_WBITS_DEFAULT=4 GR00T_GPTQ_ABITS=4 GR00T_GPTQ_MISSING=error \
    GR00T_OMEGA_W4A4=1 GR00T_W4A4_GROUP_SIZE=64 \
    "$OPENPI_PY" /private/xzq/Omega-QVLA-official-baseline/scripts/openpi_inference_service.py \
    --model-path "$CHECKPOINT" --data-config pi05_libero --port "$PORT" \
    > "$LOG/server_${label}_full50.log" 2>&1 &
  local server_pid=$!
  until grep -q "Creating OpenPI websocket server\|Traceback" "$LOG/server_${label}_full50.log" 2>/dev/null; do
    sleep 10
  done
  if grep -q "Traceback" "$LOG/server_${label}_full50.log"; then
    echo "SERVER FAILED for $label -- see $LOG/server_${label}_full50.log" | tee -a "$LOG/pipeline.log"
    return 1
  fi
  step "running eval for $label (50 trials/task, $SUITE)"
  cd /private/xzq/Omega-QVLA-official-baseline
  mkdir -p "$outdir"
  env "${NO_PROXY_ENV[@]}" "${NVIDIA_GL_ENV[@]}" PYTHONPATH="$LIBERO_PYTHONPATH" \
    "$OPENPI_PY" -m examples.Libero.eval.run_libero_eval \
    --task-suite-name "$SUITE" --num-trials-per-task 50 \
    --policy-backend openpi_ws --port "$PORT" --max-steps-profile openpi --replan-steps 5 \
    --no-save-videos --log-dir "$outdir/logs" --summary-json "$outdir/summary.json" \
    > "$LOG/eval_${label}_full50.log" 2>&1
  kill -9 "$server_pid" 2>/dev/null || true
  sleep 3
  step "$label done: $(python3 -c "import json; d=json.load(open('$outdir/summary.json')); print(d['total_successes'], '/', d['total_episodes'], '=', d['total_success_rate'])" 2>/dev/null || echo 'summary missing')"
}

step "entropy-guided full-scale eval (first, per stated order)"
run_eval "entropy" "$CACHE/packs/merged_entropy_group64_rot/quantized.pt" "$CACHE/eval_entropy_full50"

step "blind-calibration full-scale eval (second)"
run_eval "baseline" "$CACHE/packs/merged_baseline_group64_rot/quantized.pt" "$CACHE/eval_baseline_full50"

step "SUITE $NAME FULL EVAL COMPLETE"
python3 -c "
import json
b = json.load(open('$CACHE/eval_baseline_full50/summary.json'))
e = json.load(open('$CACHE/eval_entropy_full50/summary.json'))
print(f\"Baseline (blind):   {b['total_successes']}/{b['total_episodes']} = {b['total_success_rate']:.1%}\")
print(f\"Entropy-guided:     {e['total_successes']}/{e['total_episodes']} = {e['total_success_rate']:.1%}\")
" | tee -a "$LOG/pipeline.log"
