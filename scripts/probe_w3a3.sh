#!/usr/bin/env bash
# Kernel-free W3A3 accuracy probe: build a 3-bit-weight/3-bit-activation
# calibration pack using the SAME "official flow" recipe (PaliGemma GPTQ +
# Expert RTN) and existing calibration observations, but DO NOT run
# tools.pack_pi05_groupwise_int4 (it hard-rejects anything but W4:
# `if bits != 4: raise ValueError(...)`, since the deployed CUDA kernel only
# has an INT4 packing/unpacking format).
#
# Instead this evaluates directly against the plain merged pack.
# GptqLinear's runtime forward() already has a bit-width-agnostic fallback:
# omega_w4a4_backend.can_run() hard-requires weight_bits==4 and act_bits==4,
# so at W3A3 it returns False and the layer automatically falls back to its
# pure-PyTorch simulation path -- weights are already GPTQ/RTN-solved at
# 3 bits (baked into the dense `weight_res_q` values), and activations are
# quantized on the fly via `fake_quantize_groupwise_sym(x, 3, group_size)`
# (duquant_preprocess.py) -- the documented "reference layout for the
# eventual CUDA/checkpoint representation". This measures real 3-bit
# quantization error with no speedup (same latency as fp16, since nothing
# is actually packed into 3-bit storage) -- purpose is accuracy-only, to
# decide whether investing in real W3A3 kernel work (non-trivial: no native
# INT3 tensor-core instruction, needs bit-shuffling tricks) is worth it.
#
# Usage: probe_w3a3.sh <suite> <cache_name> <gpu> <port> [w_bits] [a_bits]
set -euo pipefail

SUITE="$1"
NAME="$2"
GPU="$3"
PORT="$4"
WBITS="${5:-3}"
ABITS="${6:-3}"

ROOT=/private/xzq/EntropyCalib-QVLA
OPENPI_PY=/private/xzq/openpi/.venv/bin/python
CHECKPOINT=/private/xzq/openpi/checkpoints/pi05_libero/pi05_independent_pytorch_l20_seed42/30000
CACHE="$ROOT/cache/$NAME"
LOG="$CACHE/logs"
PACKDIR="$CACHE/packs_w${WBITS}a${ABITS}"
mkdir -p "$LOG" "$PACKDIR"

# Reuse the existing static blind10 calibration obs (already recorded for
# every suite earlier this session) -- this probe is about bit-width, not
# about re-litigating calibration sample selection.
OBS="$CACHE/obs_baseline10.pt"
if [ ! -f "$OBS" ]; then
  echo "Missing $OBS -- expected the existing blind10 calibration obs for $NAME" >&2
  exit 1
fi

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

step() { echo "=== [$(date +%H:%M:%S)] [$NAME/W${WBITS}A${ABITS}-probe] $* ===" | tee -a "$LOG/pipeline_w3a3.log"; }

step "1/3 building W${WBITS}A${ABITS} expert+paligemma packs (blind10 calibration)"
cd /private/xzq/Omega-QVLA-official-baseline
env CUDA_VISIBLE_DEVICES=$GPU "$OPENPI_PY" "$ROOT/scripts/build_pi05_official_pack.py" \
  --component expert --checkpoint "$CHECKPOINT" --obs-path "$OBS" \
  --output "$PACKDIR/expert/quantized.pt" --max-samples 10 --w-bits "$WBITS" --a-bits "$ABITS" \
  > "$LOG/w3a3_expert.log" 2>&1
env CUDA_VISIBLE_DEVICES=$GPU "$OPENPI_PY" "$ROOT/scripts/build_pi05_official_pack.py" \
  --component paligemma --checkpoint "$CHECKPOINT" --obs-path "$OBS" \
  --output "$PACKDIR/paligemma/quantized.pt" --max-samples 10 --w-bits "$WBITS" --a-bits "$ABITS" \
  > "$LOG/w3a3_paligemma.log" 2>&1

step "2/3 merging (no group64 conversion -- kernel path requires W4, skipped on purpose)"
"$OPENPI_PY" -m tools.merge_packs --out "$PACKDIR/merged/quantized.pt" \
  "$PACKDIR/paligemma/quantized.pt" "$PACKDIR/expert/quantized.pt" \
  >> "$LOG/w3a3_merge.log" 2>&1
cd "$ROOT"

step "3/3 running eval (5 trials/task, fake-quantize simulation path, no kernel speedup)"
env "${NO_PROXY_ENV[@]}" PYTHONPATH=/private/xzq/Omega-QVLA CUDA_VISIBLE_DEVICES=$GPU \
  GR00T_GPTQ=1 GR00T_GPTQ_PATH="$PACKDIR/merged/quantized.pt" \
  GR00T_GPTQ_INCLUDE="$GPTQ_INCLUDE" GR00T_GPTQ_EXCLUDE="$GPTQ_EXCLUDE" \
  GR00T_GPTQ_WBITS_DEFAULT="$WBITS" GR00T_GPTQ_ABITS="$ABITS" GR00T_GPTQ_MISSING=error \
  GR00T_OMEGA_W4A4=1 GR00T_W4A4_GROUP_SIZE=64 \
  "$OPENPI_PY" /private/xzq/Omega-QVLA-official-baseline/scripts/openpi_inference_service.py \
  --model-path "$CHECKPOINT" --data-config pi05_libero --port "$PORT" \
  > "$LOG/w3a3_server.log" 2>&1 &
server_pid=$!
until grep -q "Creating OpenPI websocket server\|Traceback" "$LOG/w3a3_server.log" 2>/dev/null; do
  sleep 10
done
if grep -q "Traceback" "$LOG/w3a3_server.log"; then
  echo "SERVER FAILED -- see $LOG/w3a3_server.log" | tee -a "$LOG/pipeline_w3a3.log"
  exit 1
fi
OUTDIR="$CACHE/eval_w${WBITS}a${ABITS}_smoke5"
mkdir -p "$OUTDIR"
cd /private/xzq/Omega-QVLA-official-baseline
env "${NO_PROXY_ENV[@]}" "${NVIDIA_GL_ENV[@]}" PYTHONPATH="$LIBERO_PYTHONPATH" \
  "$OPENPI_PY" -m examples.Libero.eval.run_libero_eval \
  --task-suite-name "$SUITE" --num-trials-per-task 5 \
  --policy-backend openpi_ws --port "$PORT" --max-steps-profile openpi --replan-steps 5 \
  --no-save-videos --log-dir "$OUTDIR/logs" --summary-json "$OUTDIR/summary.json" \
  > "$LOG/w3a3_eval.log" 2>&1
kill -9 "$server_pid" 2>/dev/null || true
sleep 3
step "DONE: $(python3 -c "import json; d=json.load(open('$OUTDIR/summary.json')); print(d['total_successes'],'/',d['total_episodes'],'=',d['total_success_rate'])" 2>/dev/null || echo 'summary missing')"
