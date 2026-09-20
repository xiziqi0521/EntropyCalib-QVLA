# EntropyCalib-QVLA

Entropy-guided calibration sampling for W4A4 quantization of pi0.5.
This is a **new, standalone project** — it does not modify any of the
repos it draws from (`Omega-QVLA`, `Omega-QVLA-official-baseline`,
`Omega-QVLA-official-runs`, `DemoSpeedup`, `openpi`); everything here
imports them read-only via `sys.path` or reads their data files.

## The idea

W4A4 quantization splits into two independent stages: **calibration**
(feed the model some real observations, record activation statistics from
that forward pass) and **quantization execution** (GPTQ/RTN solve,
DuQuant rotation, the packed-INT4 kernel). This project only touches the
first stage. The quantization math itself is byte-for-byte the existing
Omega-QVLA code, unforked.

Calibration today picks its input observations blindly (first N samples).
DemoSpeedup's core algorithm — query a policy for multiple stochastic
action-chunk samples per observation, turn the resulting predictive-entropy
trace into a binary precision/redundant label via HDBSCAN — identifies
which observations the policy is confident/precise about (low entropy,
"precision-critical") versus which are redundant (high entropy). Since
quantization noise hurts precision-critical timesteps the most,
this project scores a *larger* obs pool with that signal and biases the
final calibration subset toward precision-critical observations instead of
picking blindly, at the same sample budget. Ported from DemoSpeedup: the
entropy estimator and HDBSCAN labeling. New: the entropy-guided sample
selection itself, and the pi0.5-specific plumbing to drive it.

## The actual W4A4 recipe (found via investigation, not assumed)

The user's own real "252-layer, good accuracy" pi0.5 result is the
**"official flow"**: PaliGemma LLM (126 Linear layers) quantized via plain
GPTQ, Gemma Expert (126 Linear layers) via per-step RTN — *not* the
DuQuant/ATM runtime path this project initially (incorrectly) targeted.
Two things had to be found/fixed before entropy-guided calibration could
even be tested against this recipe:

1. **Missing external patch.** Omega-QVLA's calibration hooks (and the
   `GptqLinear` runtime's per-step activation-scale dispatch) all gate on
   `gr00t.quantization.dit_step_context.get_current_dit_step()`. That
   context is only ever entered by GR00T's own model code — openpi's pi0.5
   denoising loop has no knowledge of it, so without a patch every
   calibration run silently captures zero activations. The fix
   (`entropy_calib/pi05_dit_step_patch.py`, vendored from
   `/private/xzq/Omega-QVLA/tools/pi05_dit_step_patch.py`) monkey-patches
   `PI0Pytorch.sample_actions` to enter `set_dit_quant_step(t)` per
   denoising iteration, matching whatever loop shape the installed openpi
   actually has (it had drifted from a `while` loop to a fixed
   `for _ in range(num_steps)` loop since the original patch was written;
   `apply_patch()` sanity-checks this and raises loudly rather than
   silently producing another empty pack).
2. **Rotation must NOT be folded into the weights before INT4 packing.**
   `tools/pack_pi05_groupwise_int4.py --fold-duquant-rotations` bakes
   DuQuant's rotation into the weight matrix, then re-quantizes — this
   double-quantizes an already-INT4 rotated weight and measurably hurts
   accuracy. The user's own validated packs (checked directly against
   `/private/xzq/Omega-QVLA-official-runs/step30000/packs/.../packed_group64/`)
   keep `duquant_rotation_blocks` in the record and apply the rotation to
   *activations* at runtime via the kernel's `rotate_input_blocks_wmma` —
   confirmed by reading `Omega-QVLA`'s `gptq_layers.py` kernel-path
   forward. Converting **without** `--fold-duquant-rotations` fixed a real
   accuracy regression (a run with folding measured ~0% on one config;
   the correct conversion recovered results in the same ballpark as the
   user's own historical numbers).

Loading the user's own historical `packed_group64` pack file through this
project's eval harness reproduced results in the same ballpark as their
recorded 84.2% (a decisive check that the harness itself has no bug).

## Layout

```
entropy_calib/
  entropy_utils.py              # ported KDE entropy estimator (DemoSpeedup)
  labeling.py                   # ported HDBSCAN precision/redundant labeling (DemoSpeedup)
  sample_selection.py           # entropy-guided calibration index selection
  multi_sample_policy_pi05.py   # multi-seed stochastic query adapter for openpi's Policy.infer()
  pi05_dit_step_patch.py        # vendored dit_step_context patch (see above) -- required before any
                                 # pi0.5 calibration build, GPTQ or RTN
  multi_sample_policy.py        # GR00T/Gr00tPolicy variant -- unverified, no GR00T checkpoint on disk
scripts/
  compute_entropy_labels_pi05.py       # score an obs pool, cache entropy + HDBSCAN labels
  select_entropy_calibration_obs.py    # filter a pool down to the entropy-guided subset
  build_pi05_official_pack.py          # apply the dit_step patch, then call Omega-QVLA-official-baseline's
                                        # own build_pi05_a2lite_gptq_perstep.py::main() unmodified, for
                                        # either --component expert (RTN) or paligemma (GPTQ)
  run_pi05_inference_service_patched.py  # same patch applied before serving, for eval-time correctness
  run_long_suite_pipeline.sh           # end-to-end: record pool -> entropy score -> select -> build both
                                        # sides x {baseline,entropy} -> merge -> convert -> eval, unattended
  calibrate_atm_pi05_entropy.py, compute_entropy_labels.py, build_gptq_weights_entropy.py
                                        # earlier exploration (ATM/DuQuant runtime path, GR00T path) --
                                        # superseded by the official-flow recipe above; kept for reference,
                                        # not part of the validated pipeline
  smoke_test_synthetic.py              # no-checkpoint sanity test of entropy_calib's pure logic
```

## Setup

```bash
/opt/conda/bin/uv pip install --python /private/xzq/openpi/.venv/bin/python hdbscan
```

pi0.5's own venv (`/private/xzq/openpi/.venv`) already has torch/scipy and
is used for everything -- no separate env needed. `omega_w4a4_cuda` (the
user's custom W4A4 kernel) had to be rebuilt against this venv's torch
(`cd /private/xzq/Omega-QVLA && /private/xzq/openpi/.venv/bin/python
setup_w4a4.py build_ext --inplace` after clearing the stale
`build/lib.linux-x86_64-cpython-311` — the original .so was linked against
a different torch and failed to import).

LIBERO simulation (for recording obs pools and for eval rollouts) also
runs from the openpi venv: `LIBERO_CONFIG_PATH=/private/xzq/libero_config`,
`PYTHONPATH=/private/xzq/openpi/third_party/libero`, plus the NVIDIA EGL
vars for headless MuJoCo rendering (`MUJOCO_GL=egl`,
`__EGL_VENDOR_LIBRARY_FILENAMES=.../nvidia-gl-570/.../10_nvidia.json`,
matching `LD_LIBRARY_PATH`) -- reused from the earlier DemoSpeedup
reproduction work. A stray proxy (`ALL_PROXY`/`HTTP_PROXY`) on this box
also hijacks the eval client's `localhost` websocket connection to the
inference server; unset it (`env -u ALL_PROXY -u HTTP_PROXY ...`).

## Usage

See `scripts/run_long_suite_pipeline.sh` for the full, currently-correct
sequence (record pool -> score entropy -> select subset -> build
PaliGemma+Expert packs for both blind and entropy-guided subsets -> merge
-> convert to group-64 **without** `--fold-duquant-rotations` -> eval both
through the kernel-backed `GptqLinear` runtime). Run it with
`OMEGA_QVLA_ROOT`/`CACHE`/`SUITE` adjusted for a different task suite --
it currently targets `libero_10` (Long).

## Results (2026-09-17/18, Long suite / libero_10, checkpoint
`pi05_independent_pytorch_l20_seed42/30000`, 10-sample calibration budget)

| Config | Episodes | Success rate |
|---|---:|---:|
| FP16 (user's historical, 500 ep) | 500 | 91.0% |
| Blind calibration, official flow (user's historical, 500 ep) | 500 | 84.2% |
| **Blind calibration (this project, corrected pipeline)** | 50 | **80.0%** |
| **Entropy-guided calibration (this project)** | 50 | **92.0%** |

Entropy-guided beat blind by 12pp and edged out the historical 500-episode
blind number, on the same corrected (rotation-preserving, dit_step-patched)
pipeline. Per-task breakdown shows the gap concentrated on tasks where FP16
itself succeeds easily but blind-calibration quantization struggles (e.g.
task 9, "yellow/white mug in microwave": FP16 100%, blind 40%, entropy
80%) -- consistent with entropy-guided calibration protecting exactly the
precision-critical timesteps quantization noise otherwise damages. One task
(8, "both moka pots on stove") stayed low under both calibration strategies
*and* under FP16 (33% at n=3) -- an inherently hard task for this
checkpoint, not a quantization artifact.

Caveat: n=50 per config is enough to see a real, not-obviously-noise signal
(one-sided z ≈ 1.7 for the 12pp gap) but not enough to call it statistically
airtight, and this is one task suite / one checkpoint / one calibration
budget. Repeating on Spatial/Goal/Object before treating this as a general
result is the natural next step.

## Status

- `entropy_calib/` pure logic verified against synthetic data
  (`scripts/smoke_test_synthetic.py`) and against the real checkpoint.
- Full pipeline (entropy scoring -> selection -> PaliGemma GPTQ + Expert
  RTN pack build -> merge -> group-64 kernel packing -> LIBERO eval)
  verified end-to-end on Long/`libero_10`, both directions of the
  comparison, plus a decisive cross-check against the user's own historical
  pack file to rule out an eval-harness bug.
- GR00T-N1.5 path (`multi_sample_policy.py`, `compute_entropy_labels.py`,
  `build_gptq_weights_entropy.py`) is implemented but not run against a
  real checkpoint -- parked per user request, no GR00T checkpoint on disk.
- Not yet done: Spatial/Goal/Object suites at the same scale; a genuine
  multi-GPU-sharded harness to make a full 4-suite x {blind,entropy} x
  500-episode run (the real replication scale) tractable in wall-clock
  time -- currently ~1.5-2h per 50-episode/suite run on a single GPU,
  serialized because GPUs 1-4 on this box are occupied by another tenant's
  workload.
