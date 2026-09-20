#!/usr/bin/env bash
# Run only the blind-calibration full-scale (50 trials/task) eval for a suite
# whose entropy-guided arm is already done and whose merged_baseline_group64_rot
# pack already exists (built earlier by build_suite_packs.sh). Mirrors
# run_suite_full_eval.sh's run_eval() but skips the entropy arm.
#
# Usage: run_suite_blind_only.sh <suite> <cache_name> <gpu> <port>
# e.g.   run_suite_blind_only.sh libero_10 long 0 8100
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

PACK="$CACHE/packs/merged_baseline_group64_rot/quantized.pt"
OUTDIR="$CACHE/eval_baseline_full50"

step "starting server for baseline (port $PORT, gpu $GPU)"
env "${NO_PROXY_ENV[@]}" PYTHONPATH=/private/xzq/Omega-QVLA CUDA_VISIBLE_DEVICES=$GPU \
  GR00T_GPTQ=1 GR00T_GPTQ_PATH="$PACK" \
  GR00T_GPTQ_INCLUDE="$GPTQ_INCLUDE" GR00T_GPTQ_EXCLUDE="$GPTQ_EXCLUDE" \
  GR00T_GPTQ_WBITS_DEFAULT=4 GR00T_GPTQ_ABITS=4 GR00T_GPTQ_MISSING=error \
  GR00T_OMEGA_W4A4=1 GR00T_W4A4_GROUP_SIZE=64 \
  "$OPENPI_PY" /private/xzq/Omega-QVLA-official-baseline/scripts/openpi_inference_service.py \
  --model-path "$CHECKPOINT" --data-config pi05_libero --port "$PORT" \
  > "$LOG/server_baseline_full50.log" 2>&1 &
server_pid=$!
until grep -q "Creating OpenPI websocket server\|Traceback" "$LOG/server_baseline_full50.log" 2>/dev/null; do
  sleep 10
done
if grep -q "Traceback" "$LOG/server_baseline_full50.log"; then
  echo "SERVER FAILED for baseline -- see $LOG/server_baseline_full50.log" | tee -a "$LOG/pipeline.log"
  exit 1
fi

step "running eval for baseline (50 trials/task, $SUITE)"
cd /private/xzq/Omega-QVLA-official-baseline
mkdir -p "$OUTDIR"
env "${NO_PROXY_ENV[@]}" "${NVIDIA_GL_ENV[@]}" PYTHONPATH="$LIBERO_PYTHONPATH" \
  "$OPENPI_PY" -m examples.Libero.eval.run_libero_eval \
  --task-suite-name "$SUITE" --num-trials-per-task 50 \
  --policy-backend openpi_ws --port "$PORT" --max-steps-profile openpi --replan-steps 5 \
  --no-save-videos --log-dir "$OUTDIR/logs" --summary-json "$OUTDIR/summary.json" \
  > "$LOG/eval_baseline_full50.log" 2>&1
kill -9 "$server_pid" 2>/dev/null || true
sleep 3
step "baseline done: $(python3 -c "import json; d=json.load(open('$OUTDIR/summary.json')); print(d['total_successes'], '/', d['total_episodes'], '=', d['total_success_rate'])" 2>/dev/null || echo 'summary missing')"
